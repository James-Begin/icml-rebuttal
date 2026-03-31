# Rebuttal — Reviewer xd5v

We thank the reviewer. Data, CIs, and traces are in the [supplementary repository]([ANONYMISED_REPO_URL]).

---

## Key Questions

**Q1: Log traces.**
The two mechanisms have different interaction structures. The **LMSR market** (Section 6) has no dialogue: each agent independently observes prices and submits a trade. The **Semantic market** (Section 5) has a sequential MM↔Trader exchange. Full JSONL traces for both are in `traces/` of the supplementary repository. In a representative LMSR episode, blind malicious agents inadvertently support the correct outcome — illustrating the imperfect adversarial effectiveness discussed below. We will revise Figures 1–2 to clarify this distinction.

**Q2: Market population / mode collapse.**
Each experiment uses N=5 agents of the same model family with independent calls and distinct role prompts. From our episode logs: 10.0% of TruthfulQA 2v3 Blind episodes had both honest agents vote incorrectly with 100% pairwise agreement (n=3). Correlated errors are real, but adversarial agents exhibit the same pattern — noise is bidirectional. Honest claim: the market provides adversarial resistance, not Condorcet-style diversity.

**Q3: Vote Acc >0% in 2v3 Informed.**
The theoretical prediction is 0%. The observed 0.3% is sampling noise: LLMs occasionally fail their adversarial role prompt — the same p_adv < 1.0 property discussed below. 0.3% ≈ 0; voting is essentially entirely compromised. A note will be added to Section 6.2.1 to clarify.

---

## Absolute Baselines

Verified TruthfulQA baselines match Table 1 net gains exactly (full table: supplementary, Section 3): Qwen 0.6B 45.7%, Qwen 8B 68.1%, Qwen 235B 89.5%. To our knowledge these are the first published Qwen3 TruthfulQA evaluations. Llama 3.1 8B standalone (supplementary, Section 1): TruthfulQA 70.2% [68.5%, 71.9%], consistent with MC2 68.3%. Baseline columns will be added to all tables.

---

## LMSR Choice and Economic Basis

LMSR is the only *strictly proper* scoring rule among the alternatives: agents maximise expected utility by reporting true beliefs when resolution is against ground truth. CLOB lacks liquidity guarantees for thin markets (N=5); RFQ is bilateral and non-public; LS-LMSR trades properness for bounded loss. A comparative paragraph will be added to Section 3.2; Section 2.2 expanded for economics-unfamiliar readers.

---

## Section 6.3: Pump-and-Dump and the Epistemic Arbitrage Argument

Section 6.3's argument assumes fixed-directional adversaries, not wealth-maximisers. We concede this and will state it explicitly.

**Analytical intuition on bluffing.** A bluffing adversary bets truthfully for k rounds, then defects. LMSR payoffs are proportional to shares: malicious hold 3/5, honest 2/5. After settlement both scale by (1+r) — the 3:2 wealth ratio is preserved. Bluffing provides no relative capital advantage; the genuine vulnerability requires persistent cross-episode state and inter-agent coordination, both of which our setup structurally prevents — each agent receives an independent stateless prompt with no memory of prior rounds.

**Empirical confirmation.** Ablation augmenting prompts with capital balances (supplementary, Section 5): accuracy dropped 12.2pp (73.3% → 61.1%), malicious wealth doubled (33.3% → 67.2%). Contribution #3 will be scoped to fixed-directional adversaries; wealth-maximising adversaries flagged as future work.

---

## Confidence Intervals and Capability Threshold

Key CIs (supplementary, Section 2): TruthfulQA 2v3 Informed 54.0% [49.8%, 58.2%]; GPQA Binary 2v3 Informed 39.1% [32.2%, 46.1%] — below chance, because Llama 3 8B's GPQA Binary standalone is 49.8% (near chance), leaving almost no calibration signal to amplify. The mechanism's adversarial advantage is model-capability-dependent: GPT-OSS (p_h ≈ 0.80, break-even at p_adv = 0.70; empirical 0.17–0.42) — market dominates strongly; Llama 3 8B (p_h ≈ 0.70, break-even 0.63; empirical ≈ 0.63) — at threshold, no market advantage over debate. ---

## Debate Scalability and Attack Vectors

**Debate scalability.** We have this data. Fresh 3-trial experiments (supplementary, Section 4): market outperforms debate at all nine adversarial ratios (77%→58% vs 74%→44%, Blind Deception). The debate curve will be added to Figure 5. Blind and Informed Deception mirror Amayuelas et al. (2024) (EMNLP 2024 Findings), cited in Section 2.3; we will cross-reference in Section 6.1.

---

## Revision Commitments

- Baseline columns (all tables); CI columns; GPT/Llama appendix pointer
- LMSR vs alternatives (Section 3.2); expand economic basis (Section 2.2)
- Scope Contribution #3; bluffing analysis and wealth-aware limitation in Section 7
- Debate scalability curve in Figure 5; cross-reference Amayuelas et al. in Section 6.1
- **Release full experimental code and traces** as supplementary material
- Clarifications: Figures 1–2; Sections 6.2.1, 6.2.3; Table 3 caption; properness (Section 2.2); Brier score (Figure 3)
