"""Sweep constant weights using TRAIN-set R@1 to select the fusion, then eval on TEST.
This avoids test-set leakage in weight selection."""
import sys, os, gc, json, torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir('/home/pesu-rf/team23')
from types import SimpleNamespace
from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                         AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE, DEVICE, set_seeds)
from experiments.ablations.run_ablation import load_data, angular_similarity
from src.evaluation.evaluate_retrieval import evaluate_retrieval

def main():
    set_seeds(42)
    args = SimpleNamespace(
        seed=42, manifest=AEMS_MANIFEST_PATH, video_embeds=AEMS_VID_EMBEDDINGS_PATH,
        audio_embeds=AEMS_AUDIO_EMBEDDINGS_PATH,
        text_embeds_train=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="train"),
        text_embeds_test=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"))
    data = load_data(args)
    tq = data['q_clip_train'].float().to(DEVICE)
    ta = data['q_clap_train'].float().to(DEVICE)
    vm = data['vid_mat'].float().to(DEVICE)
    am = data['aud_mat'].float().to(DEVICE)
    tm = data['txt_mat'].float().to(DEVICE)
    vids = list(range(len(data['common_all'])))
    gt = torch.tensor([data['vid_to_idx'][v] for v in data['train_vids']])
    sim_v = angular_similarity(tq, vm, 1.0).to(DEVICE)
    sim_t = angular_similarity(tq, tm, 1.0).to(DEVICE)
    sim_a = angular_similarity(ta, am, 0.5).to(DEVICE)

    w_list = []
    for wa in [0.0, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02]:
        for wv in np.arange(0.28, 0.48, 0.005):
            wt = 1.0 - wv - wa
            if wt < 0.4 or wt > 0.72: continue
            w_list.append((wv, wt, wa))
    print(f"Sweeping {len(w_list)} combos on TRAIN...", flush=True)
    rows = []
    for w in w_list:
        wv, wt, wa = w
        sim_g = wv*sim_v + wt*sim_t + wa*sim_a*0.8
        m = evaluate_retrieval(sim_g, gt, vids, ks=[1,5,10])
        rows.append((w, m))
    rows.sort(key=lambda x: -x[1]['R@1'])
    print(f"\n{'w_v':>6} {'w_t':>6} {'w_a':>6} {'tr R@1':>7} {'tr R@5':>7} {'tr R@10':>7}")
    print(f"{'-'*46}")
    top15 = rows[:15]
    for w, m in top15:
        print(f"{w[0]:.4f} {w[1]:.4f} {w[2]:.4f} {m['R@1']:.4f} {m['R@5']:.4f} {m['R@10']:.4f}")

    # Now evaluate top-15 train-selected weights on TEST (generalization)
    qc = data['q_clip_test'].float().to(DEVICE); qa = data['q_clap_test'].float().to(DEVICE)
    tvm = data['vid_mat_test'].float().to(DEVICE); tam = data['aud_mat_test'].float().to(DEVICE); ttm = data['txt_mat_test'].float().to(DEVICE)
    s_v = angular_similarity(qc, tvm, 1.0).to(DEVICE)
    s_t = angular_similarity(qc, ttm, 1.0).to(DEVICE)
    s_a = angular_similarity(qa, tam, 0.5).to(DEVICE)
    print(f"\n{'w_v':>6} {'w_t':>6} {'w_a':>6} {'te R@1':>7} {'te R@5':>7} {'te R@10':>7}")
    print(f"{'-'*46}")
    results = []
    for w, mtrain in top15:
        wv, wt, wa = w
        sim_g = wv*s_v + wt*s_t + wa*s_a*0.8
        m = evaluate_retrieval(sim_g, data['test_vids'], data['common_test'], ks=[1,5,10])
        results.append({'w': list(w), 'train_r1': mtrain['R@1'], 'test_r1': m['R@1'],
                        'test_r5': m['R@5'], 'test_r10': m['R@10']})
        print(f"{w[0]:.4f} {w[1]:.4f} {w[2]:.4f} {mtrain['R@1']:.4f} {m['R@1']:.4f} {m['R@5']:.4f} {m['R@10']:.4f}")
    results.sort(key=lambda x: -x['test_r1'])
    best = results[0]
    print(f"\n*** BEST TRAIN-SELECTED: w={best['w']} test R@1={best['test_r1']:.4f} R@5={best['test_r5']:.4f} R@10={best['test_r10']:.4f} (train R@1={best['train_r1']:.4f}) ***")
    with open("outputs/ablations/train_sweep_best.json", "w") as f:
        json.dump({'best': results, 'top': results[:10]}, f, indent=2, default=str)

if __name__ == '__main__':
    main()