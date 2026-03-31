# Rebuttal — Reviewer 4

We address each concern honestly.

---

## Formal Incentive Compatibility

The reviewer is correct. LMSR is strictly proper when resolution is against external ground truth; under consensus resolution that guarantee does not apply. We concede there is no formal truthtelling guarantee in our setup. What we can claim rigorously: the LMSR cost function creates asymmetric costs between truth-seeking and deceptive behaviour when agents are imperfectly adversarial, and these compound over episodes into systematic wealth transfer. This is an empirical property of the mechanism interacting with observed LLM behaviour. The paper's current framing — that truthfulness *emerges* from market incentives — overstates this. The revision will carry the reframing through consistently across title, abstract, introduction, and Section 6.3, not as word substitution but as a rewritten narrative arc.

---

## "Is this just debate with a scoring formula?" — Wealth and Prompts

Agents are not told their capital. This is intentional: wealth redistribution operates at the system level via the LMSR cost function regardless of agent awareness. To test whether this design choice is consequential, we ran an ablation augmenting each agent's prompt with its capital balance and the statement that wealth determines future influence (supplementary, Section 5; Llama 3.1 8B, TruthfulQA, 2v3 Informed, 3 trials):

| Condition | Market Acc | Mal. Wealth Share |
|-----------|-----------|------------------|
| No wealth in prompt (paper setup) | 73.3% ± 7.2% | 33.3% |
| Wealth in prompt | 61.1% ± 3.1% | 67.2% |

We proactively acknowledge two interpretations: (1) the directional effect is consistent with economic reasoning — wealth-aware adversaries become more effective in exactly the way a capital-maximising agent would; (2) LLMs trained on market data may produce strategic-looking behaviour through role-framing alone, without genuine economic optimization within the mechanism. We cannot rule out (2). What the ablation shows is that *economic framing specifically* produces systematically different behaviour in the predicted direction. The revision will address this distinction explicitly. The ablation is underpowered (1 model, 1 dataset, 3 trials) and we will expand it.

The LMSR creates asymmetric outcomes: a lie requires capital forfeited at resolution, unlike debate where persuasion costs nothing. This operates regardless of agent awareness.

---

## Missing Iterative-Refinement Baseline

The semantic market (Section 5) is structurally similar to iterative refinement: propose → critique → update. Key structural differences: (a) the Trader is a *separate agent* with independent sampling; (b) probability quantification forces explicit calibration; (c) in the adversarial setting, the Trader has an opposing objective. We have not run a direct comparison against self-refine (Madaan et al., 2023). This is a legitimate gap and the most important missing piece for the cooperative setting. We will add this ablation and report honestly even if it partially explains the gains.

---

## Experimental Assumptions and Scope

**Strategic adversaries.** This is a fundamental limitation, not a minor caveat. The mechanism is demonstrated only against non-strategic, fixed-directional adversaries. On mildly adaptive strategies in binary markets: an adversary stopping when prices are unfavorable limits its own price impact; distributing bets across false outcomes is impossible (only 2 outcomes). The most threatening adaptive strategies apply to MCQ, which already fails. We will provide an analytical treatment in the revision. The honest scope: LMSR dynamics amplify existing LLM calibration in binary settings against non-strategic adversaries — stated in abstract and contributions, not only Section 7.

**Dataset memorisation.** GPQA Diamond is substantially newer than most pre-training data; accuracy (39–65% vs. 25% random baseline) is consistent with TruthfulQA, suggesting genuine signal. We acknowledge this is not definitive and will note it in Section 7.

**MCQ degradation.** GPQA MCQ (2v3 Informed): 21.3% for Llama 3 8B, below 25% random baseline. The mechanism fails here; we do not dispute it. Abstract and contributions will scope robustness claims to binary settings.

---

## Revision Commitments

- Rewrite narrative arc (title, abstract, introduction, Section 6.3) to carry empirical reframing consistently — not word substitution
- State in Appendix B.2 that agents do not observe their capital, and why
- Expand wealth ablation (multiple models/datasets); address role-framing vs. economic reasoning explicitly
- Add self-refine comparison (Section 5); report honestly
- Analytical treatment of mildly adaptive adversaries; strategic adversary limitation in abstract
- Scope robustness to binary settings; memorisation caveat in Section 7
