"""Multi-seed validation of the best tethered-constant configs."""
import sys, os, json
sys.path.insert(0, '/home/pesu-rf/team23')
os.chdir('/home/pesu-rf/team23')
from types import SimpleNamespace
from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_CLAP_AUDIO_EMBEDDINGS_PATH,
                         AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE)
from experiments.ablations.run_ablation import load_data, train_and_eval

args = SimpleNamespace(
    seed=42, manifest=AEMS_MANIFEST_PATH, video_embeds=AEMS_VID_EMBEDDINGS_PATH,
    audio_embeds=AEMS_CLAP_AUDIO_EMBEDDINGS_PATH,
    text_embeds_train=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="train"),
    text_embeds_test=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"))
data = load_data(args)

BASE = {'loss': 'pairwise_margin', 'n_neg': 10, 'hard_neg': True, 'lr': 1e-3,
        'constant_weights': True, 'init_weights': [0.30, 0.69, 0.01],
        'target_weights': [0.30, 0.69, 0.01], 'weight_reg': 0.1, 'epochs': 30}

configs = []
for seed in [123, 456, 789]:
    for margin in [0.1, 0.2]:
        c = dict(BASE); c['margin'] = margin; c['seed'] = seed
        c['name'] = f"MS_seed{seed}_m{margin}"
        configs.append(c)

out = []
for cfg in configs:
    args.seed = cfg['seed']
    print(f"\n>>> {cfg['name']}", flush=True)
    res = train_and_eval(args, data, cfg)
    out.append({'name': cfg['name'], 'seed': cfg['seed'], 'margin': cfg['margin'],
                'r1': res['metrics']['R@1'], 'r5': res['metrics']['R@5'],
                'r10': res['metrics']['R@10'], 'w': res['avg_weights']})
    print(f">>> DONE {cfg['name']}: R@1={res['metrics']['R@1']:.4f} w={res['avg_weights']}", flush=True)

with open('outputs/ablations/multiseed_tethered.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print("\nSEED RUNS:", out)
print("SAVED_ALL")