"""Focused runs: constant net tethered toward train-selected optimum [0.30,0.69,0.01]."""
import sys, os, json
sys.path.insert(0, '/home/pesu-rf/team23')
os.chdir('/home/pesu-rf/team23')
from types import SimpleNamespace
from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                         AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE)
from scripts.ablation.run_ablation import load_data, train_and_eval

args = SimpleNamespace(
    seed=42, manifest=AEMS_MANIFEST_PATH, video_embeds=AEMS_VID_EMBEDDINGS_PATH,
    audio_embeds=AEMS_AUDIO_EMBEDDINGS_PATH,
    text_embeds_train=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="train"),
    text_embeds_test=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"))
data = load_data(args)

configs = []
for reg in [0.01, 0.1, 1.0]:
    for m in [0.2, 0.1]:
        configs.append({
            'name': f"C5r{reg}m{m}",
            'loss': 'pairwise_margin', 'margin': m, 'n_neg': 10, 'hard_neg': True,
            'constant_weights': True, 'weight_reg': reg, 'lr': 1e-3,
            'init_weights': [0.30, 0.69, 0.01], 'target_weights': [0.30, 0.69, 0.01],
            'epochs': 30})
configs.append({
    'name': "C5_lr5e-4_reg0.1",
    'loss': 'pairwise_margin', 'margin': 0.2, 'n_neg': 10, 'hard_neg': True,
    'constant_weights': True, 'weight_reg': 0.1, 'lr': 5e-4,
    'init_weights': [0.30, 0.69, 0.01], 'target_weights': [0.30, 0.69, 0.01],
    'epochs': 30})

out = []
for cfg in configs:
    print(f"\n{'='*70}\n>>> {cfg['name']}  (reg={cfg.get('weight_reg')}, margin={cfg.get('margin')})", flush=True)
    res = train_and_eval(args, data, cfg)
    out.append({'name': cfg['name'], 'cfg': cfg, 'results': res})
    print(f">>> DONE {cfg['name']}: R@1={res['metrics']['R@1']:.4f} w={res['avg_weights']}",
          flush=True)

with open('outputs/ablations/phase_c5_tethered.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print("\nSAVED_ALL")