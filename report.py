"""
Text report — numerical summaries for all simulation plots.
Run this to view numerical results without viewing images.
"""

import numpy as np
from sim import run, N_QUESTIONS, N_SEEDS, P_ADV_LLAMA_8B, BE_2V3, BE_3V2, BE_3V7, BE_1V4

steps = np.arange(1, N_QUESTIONS + 1)


def summarize(r, label, resolution, p_adv):
    h_end = r['honest_mean'][-1]
    h_std = r['honest_std'][-1]
    m_end = r['mal_mean'][-1]
    m_std = r['mal_std'][-1]
    acc_end = r['acc_mean'][-1]
    acc_std = r['acc_std'][-1]
    h_trend = "UP" if h_end > 100 else "DOWN"
    m_trend = "UP" if m_end > 100 else "DOWN"
    honest_wins = h_end > m_end
    print(f"  [{label} | {resolution} | p_adv={p_adv}]")
    print(f"    Honest   : end={h_end:.1f} ± {h_std:.1f}  trend={h_trend}")
    print(f"    Malicious: end={m_end:.1f} ± {m_std:.1f}  trend={m_trend}")
    print(f"    Gap (honest-mal): {h_end - m_end:+.1f}")
    print(f"    Accuracy : {acc_end:.3f} ± {acc_std:.3f}  (chance=0.500)")
    print(f"    Honest wins? {honest_wins}")
    print()


print("=" * 70)
print("PLOT 1 — Break-Even Curve")
print(f"Theoretical break-even for 2v3, p_honest=0.70: {BE_2V3:.4f}")
print("=" * 70)
P_ADV_SWEEP = [0.50, 0.55, 0.60, 0.634, 0.70, 0.85, 1.00]
for p in P_ADV_SWEEP:
    r = run(n_good=2, n_bad=3, p_adversarial=p, resolution='consensus')
    gap = r['honest_mean'][-1] - r['mal_mean'][-1]
    acc = r['acc_mean'][-1]
    below_be = "LLM regime" if p < BE_2V3 else "Above break-even"
    print(f"  p_adv={p:.3f}: wealth_gap={gap:+.1f}  acc={acc:.3f}  {below_be}")
print()

print("=" * 70)
print("PLOT 2 — Wealth Trajectories in LLM Regime (p_adv=0.60)")
print("=" * 70)
for res in ['ground_truth', 'consensus']:
    r = run(n_good=2, n_bad=3, p_adversarial=P_ADV_LLAMA_8B, resolution=res)
    summarize(r, "2v3", res, P_ADV_LLAMA_8B)

print("=" * 70)
print("PLOT 3 — Adversarial Spectrum (consensus only)")
print("=" * 70)
for p in [0.55, 0.60, BE_2V3, 0.70, 0.85, 1.00]:
    r = run(n_good=2, n_bad=3, p_adversarial=p, resolution='consensus')
    gap = r['honest_mean'][-1] - r['mal_mean'][-1]
    acc = r['acc_mean'][-1]
    print(f"  p_adv={p:.3f}: honest_end={r['honest_mean'][-1]:.1f}  "
          f"mal_end={r['mal_mean'][-1]:.1f}  gap={gap:+.1f}  acc={acc:.3f}")
print()

print("=" * 70)
print("PLOT 4 — Compositions in LLM Regime (p_adv=0.60)")
print("=" * 70)
CONFIGS = [
    ("2v3 bad-majority", 2, 3, BE_2V3),
    ("3v2 good-majority", 3, 2, BE_3V2),
    ("3v7 bad-majority",  3, 7, BE_3V7),
    ("1v4 bad-majority",  1, 4, BE_1V4),
]
for res in ['ground_truth', 'consensus']:
    for label, ng, nb, be in CONFIGS:
        r = run(n_good=ng, n_bad=nb, p_adversarial=P_ADV_LLAMA_8B, resolution=res)
        gap = r['honest_mean'][-1] - r['mal_mean'][-1]
        wins = "YES" if gap > 0 else "NO"
        print(f"  [{res}] {label} (break-even={be:.3f}): gap={gap:+.1f}  "
              f"acc={r['acc_mean'][-1]:.3f}  honest_wins={wins}")
print()

print("=" * 70)
print("PLOT 5 — Empirical Calibration")
print(f"What accuracy does each p_adversarial produce? (2v3 consensus, 100 questions)")
print("=" * 70)
for p in [0.50, 0.55, 0.60, 0.634, 0.70, 1.00]:
    r = run(n_good=2, n_bad=3, p_adversarial=p, resolution='consensus')
    print(f"  p_adv={p:.3f}: final accuracy = {r['acc_mean'][-1]:.3f}  "
          f"(paper refs: Llama8B=0.540, GPT20B=0.670, GPT120B=0.823)")

print()
print("Feedback loop — accuracy change over episodes at p_adv=0.60 (LLM regime):")
r = run(n_good=2, n_bad=3, p_adversarial=0.60, resolution='consensus')
for ep in [10, 25, 50, 100]:
    print(f"  After episode {ep:3d}: accuracy = {r['acc_mean'][ep-1]:.3f}")
