"""Phase 5: Deep run of the best configs (30 epochs)"""
import sys, os, time, json, torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir('/home/pesu-rf/team23')
from src.config import DEVICE, set_seeds
from src.models.gating_network import GatingNetwork
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from scripts.ablation.run_ablation import angular_similarity, load_data

def train_and_eval_deep(data, config, epochs=30):
    set_seeds(42)
    loss_name = config.get('loss', 'pairwise_margin')
    n_neg = config.get('n_neg', 10)
    margin = config.get('margin', 0.1)
    hard_neg = config.get('hard_neg', True)
    temperature = config.get('temperature', 0.07)
    lr = config.get('lr', 1e-3)
    tau_audio = config.get('tau_audio', 0.5)
    tau_text = config.get('tau_text', 1.0)
    tau_visual = config.get('tau_visual', 1.0)

    gating_net = GatingNetwork(dropout=0.3).to(DEVICE)
    optimizer = torch.optim.AdamW(gating_net.parameters(), lr=lr, weight_decay=1e-5)

    num_train = len(data['train_queries'])
    num_candidates = len(data['common_all'])
    gt_indices = torch.tensor([data['vid_to_idx'][data['train_vids'][i]] for i in range(num_train)], device=DEVICE)

    best_loss = float('inf')
    all_weights = []

    for epoch in range(epochs):
        t0_epoch = time.time()
        gating_net.train()
        total_loss = torch.tensor(0.0, device=DEVICE)
        num_batches = 0
        indices = torch.randperm(num_train)
        for q_start in range(0, num_train, 32):
            q_end = min(q_start + 32, num_train)
            batch_len = q_end - q_start
            if batch_len < 16: continue
            batch_idx = indices[q_start:q_end]
            q_clip = data['q_clip_train'][batch_idx].float().to(DEVICE)
            q_clap = data['q_clap_train'][batch_idx].float().to(DEVICE)
            weights = gating_net(q_clip)
            all_sim_v = torch.zeros(batch_len, num_candidates, device=DEVICE)
            all_sim_a = torch.zeros(batch_len, num_candidates, device=DEVICE)
            all_sim_t = torch.zeros(batch_len, num_candidates, device=DEVICE)
            for vs in range(0, num_candidates, 500):
                ve = min(vs + 500, num_candidates)
                vb_v = data['vid_mat'][vs:ve].float().to(DEVICE)
                vb_a = data['aud_mat'][vs:ve].float().to(DEVICE)
                vb_t = data['txt_mat'][vs:ve].float().to(DEVICE)
                all_sim_v[:, vs:ve] = angular_similarity(q_clip, vb_v, tau_visual)
                all_sim_a[:, vs:ve] = angular_similarity(q_clap, vb_a, tau_audio)
                all_sim_t[:, vs:ve] = angular_similarity(q_clip, vb_t, tau_text)
                del vb_v, vb_a, vb_t
            sim_gated = (weights[:, 0:1] * all_sim_v +
                        weights[:, 1:2] * all_sim_t +
                        weights[:, 2:3] * all_sim_a * 0.8)
            batch_gt = gt_indices[batch_idx]
            from scripts.ablation.run_ablation import LOSS_FUNCTIONS
            loss = LOSS_FUNCTIONS[loss_name](sim_gated, batch_gt,
                margin=margin, n_neg=n_neg, hard=hard_neg, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gating_net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        gating_net.eval()
        with torch.no_grad():
            w_test = gating_net(data['q_clip_train'].float().to(DEVICE))
            avg_w = w_test.mean(dim=0).cpu().numpy()
        all_weights.append(avg_w.tolist())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}  Loss={avg_loss:.4f}  w=[{avg_w[0]:.3f},{avg_w[1]:.3f},{avg_w[2]:.3f}]  Time={time.time()-t0_epoch:.0f}s", flush=True)

    gating_net.eval()
    sim_v = angular_similarity(data['q_clip_test'].float().to(DEVICE), data['vid_mat_test'].float().to(DEVICE), tau_visual).to(DEVICE)
    sim_t = angular_similarity(data['q_clip_test'].float().to(DEVICE), data['txt_mat_test'].float().to(DEVICE), tau_text).to(DEVICE)
    sim_a = angular_similarity(data['q_clap_test'].float().to(DEVICE), data['aud_mat_test'].float().to(DEVICE), tau_audio).to(DEVICE)
    with torch.no_grad():
        gate_w = gating_net(data['q_clip_test'].float().to(DEVICE))
    sim_gated = (gate_w[:, 0:1] * sim_v +
                 gate_w[:, 1:2] * sim_t +
                 gate_w[:, 2:3] * sim_a * 0.8)
    metrics = evaluate_retrieval(sim_gated, data['test_vids'], data['common_test'], ks=[1, 5, 10])
    text_metrics = evaluate_retrieval(sim_t, data['test_vids'], data['common_test'], ks=[1, 5, 10])
    vis_metrics = evaluate_retrieval(sim_v, data['test_vids'], data['common_test'], ks=[1, 5, 10])
    aud_metrics = evaluate_retrieval(sim_a, data['test_vids'], data['common_test'], ks=[1, 5, 10])
    return {
        'r1': metrics['R@1'], 'r5': metrics['R@5'], 'r10': metrics['R@10'],
        'text_r1': text_metrics['R@1'], 'text_r5': text_metrics['R@5'], 'text_r10': text_metrics['R@10'],
        'vis_r1': vis_metrics['R@1'], 'vis_r5': vis_metrics['R@5'], 'vis_r10': vis_metrics['R@10'],
        'aud_r1': aud_metrics['R@1'], 'aud_r5': aud_metrics['R@5'], 'aud_r10': aud_metrics['R@10'],
        'final_weights': [float(avg_w[0]), float(avg_w[1]), float(avg_w[2])],
        'weight_history': all_weights,
    }

if __name__ == '__main__':
    from types import SimpleNamespace
    from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH,
                            AEMS_AUDIO_EMBEDDINGS_PATH, AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE)
    args = SimpleNamespace(
        seed=42,
        manifest=AEMS_MANIFEST_PATH,
        video_embeds=AEMS_VID_EMBEDDINGS_PATH,
        audio_embeds=AEMS_AUDIO_EMBEDDINGS_PATH,
        text_embeds_train=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="train"),
        text_embeds_test=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"),
    )
    data = load_data(args)
    configs = [
        {'name': 'PM_hard_m0.1_n10_30e', 'loss': 'pairwise_margin', 'margin': 0.1, 'n_neg': 10, 'hard_neg': True, 'temperature': 0.07, 'lr': 1e-3},
        {'name': 'PM_hard_m0.1_n100_30e', 'loss': 'pairwise_margin', 'margin': 0.1, 'n_neg': 100, 'hard_neg': True, 'temperature': 0.07, 'lr': 1e-3},
        {'name': 'PM_hard_m0.2_n10_30e', 'loss': 'pairwise_margin', 'margin': 0.2, 'n_neg': 10, 'hard_neg': True, 'temperature': 0.07, 'lr': 1e-3},
        {'name': 'PM_hard_m0.1_n10_lr1e4_30e', 'loss': 'pairwise_margin', 'margin': 0.1, 'n_neg': 10, 'hard_neg': True, 'temperature': 0.07, 'lr': 1e-4},
        {'name': 'MNRL_t0.01_n100_30e', 'loss': 'mnrl', 'temperature': 0.01, 'n_neg': 100, 'lr': 1e-3},
    ]
    all_results = []
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"CONFIG: {cfg['name']}")
        print(f"{'='*60}")
        t0 = time.time()
        results = train_and_eval_deep(data, cfg, epochs=30)
        elapsed = time.time() - t0
        r = {
            'name': cfg['name'], 'config': cfg,
            'results': results,
            'time_s': elapsed
        }
        all_results.append(r)
        print(f"  R@1={results['r1']:.4f}  R@5={results['r5']:.4f}  R@10={results['r10']:.4f}")
        print(f"  Time: {elapsed:.0f}s")
        with open(f"outputs/ablations/phase5_{cfg['name']}.json", 'w') as f:
            json.dump(r, f, indent=2)

    print(f"\n{'='*60}")
    print("PHASE 5 SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<35} {'R@1':>6} {'R@5':>6} {'R@10':>6}  Weights")
    print(f"{'-'*35} {'-'*6} {'-'*6} {'-'*6}  {'-'*20}")
    for r in sorted(all_results, key=lambda x: -x['results']['r1']):
        res = r['results']
        w = res['final_weights']
        print(f"{r['name']:<35} {res['r1']:6.4f} {res['r5']:6.4f} {res['r10']:6.4f}  [{w[0]:.3f},{w[1]:.3f},{w[2]:.3f}]")

    with open("outputs/ablations/phase5_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to outputs/ablations/phase5_*.json")
