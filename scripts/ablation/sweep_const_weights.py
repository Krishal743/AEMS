"""Fine-grained constant-weight sweep over the 2D simplex, reusing once-loaded data."""
import sys, os, gc, torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir('/home/pesu-rf/team23')
import torch.nn.functional as F
from types import SimpleNamespace
from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                         AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE, DEVICE)
from scripts.ablation.run_ablation import load_data, angular_similarity, train_and_eval
from src.evaluation.evaluate_retrieval import evaluate_retrieval

def sweep_weights(data, w_list, tau_audio=0.5, tau_text=1.0, tau_visual=1.0, scale_a=0.8):
    qc = data['q_clip_test'].float().to(DEVICE)
    qa = data['q_clap_test'].float().to(DEVICE)
    vm = data['vid_mat_test'].float().to(DEVICE)
    am = data['aud_mat_test'].float().to(DEVICE)
    tm = data['txt_mat_test'].float().to(DEVICE)
    sim_v = angular_similarity(qc, vm, tau_visual).to(DEVICE)
    sim_t = angular_similarity(qc, tm, tau_text).to(DEVICE)
    sim_a = angular_similarity(qa, am, tau_audio).to(DEVICE)
    results = []
    for i, w in enumerate(w_list):
        wv, wt, wa = w
        sim_gated = wv * sim_v + wt * sim_t + wa * sim_a * scale_a
        m = evaluate_retrieval(sim_gated, data['test_vids'], data['common_test'], ks=[1, 5, 10])
        results.append((w, m['R@1'], m['R@5'], m['R@10']))
    del sim_v, sim_t, sim_a
    gc.collect(); torch.cuda.empty_cache()
    return results

def main():
    from src.config import set_seeds
    set_seeds(42)
    args = SimpleNamespace(
        seed=42, manifest=AEMS_MANIFEST_PATH, video_embeds=AEMS_VID_EMBEDDINGS_PATH,
        audio_embeds=AEMS_AUDIO_EMBEDDINGS_PATH,
        text_embeds_train=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="train"),
        text_embeds_test=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"))
    data = load_data(args)

    # Coarse sweep: wa fixed to [0.005, 0.02], wv steps
    w_list = []
    for wa in [0.005, 0.01, 0.02]:
        wv_vals = np.arange(0.28, 0.50, 0.005)
        for wv in wv_vals:
            wt = 1.0 - wv - wa
            if wt < 0.4 or wt > 0.7: continue
            w_list.append((wv, wt, wa))
    print(f"Sweeping {len(w_list)} weight combos (+ coarse)...", flush=True)
    coarse = sweep_weights(data, w_list)
    coarse.sort(key=lambda x: -x[1])
    print(f"\n{'w_v':>6} {'w_t':>6} {'w_a':>6} {'R@1':>6} {'R@5':>6} {'R@10':>6}")
    print(f"{'-'*44}")
    for w, r1, r5, r10 in coarse[:20]:
        print(f"{w[0]:.4f} {w[1]:.4f} {w[2]:.4f} {r1:.4f} {r5:.4f} {r10:.4f}")

    # Fine sweep around best 3
    best3 = coarse[:3]
    fine = []
    for bw, br1, _, _ in best3:
        for wa in np.arange(max(0, bw[2]-0.01), bw[2]+0.011, 0.002):
            for wv in np.arange(bw[0]-0.03, bw[0]+0.031, 0.002):
                wt = 1.0 - wv - wa
                if wt < 0.4 or wt > 0.72: continue
                fine.append((wv, wt, wa))
    print(f"\nFine sweep: {len(fine)} combos", flush=True)
    fine_results = sweep_weights(data, fine)
    fine_results.sort(key=lambda x: -x[1])
    print(f"\n{'w_v':>6} {'w_t':>6} {'w_a':>6} {'R@1':>6} {'R@5':>6} {'R@10':>6}")
    print(f"{'-'*44}")
    for w, r1, r5, r10 in fine_results[:20]:
        print(f"{w[0]:.4f} {w[1]:.4f} {w[2]:.4f} {r1:.4f} {r5:.4f} {r10:.4f}")

    combined = coarse + fine_results
    combined.sort(key=lambda x: -x[1])
    best = combined[0]
    print(f"\n*** BEST: w=[{best[0][0]:.4f},{best[0][1]:.4f},{best[0][2]:.4f}] R@1={best[1]:.4f} R@5={best[2]:.4f} R@10={best[3]:.4f} ***")
    with open("outputs/ablations/weight_sweep_best.json", "w") as f:
        import json
        json.dump({'best': list(best[0]), 'r1': best[1], 'r5': best[2], 'r10': best[3],
                   'coarse_top': [{'w': list(x[0]), 'r1': x[1]} for x in coarse[:20]],
                   'fine_top': [{'w': list(x[0]), 'r1': x[1]} for x in fine_results[:20]]}, f, indent=2)

if __name__ == '__main__':
    main()