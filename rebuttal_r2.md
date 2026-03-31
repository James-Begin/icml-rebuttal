# Rebuttal — Reviewer 2

We address each point directly; data, CIs, and simulation plots are in the [supplementary repository]([ANONYMISED_REPO_URL]).

---

## Q1: The Resolution Mechanism

**The reviewer is correct.** With perfectly adversarial agents, consensus resolution is symmetric and cannot distinguish honest from malicious majorities. We concede this fully. The phrase "incentives are not guaranteed to be perfectly truth-aligned" understates the issue; it should read: *with perfectly adversarial agents, consensus resolution fails entirely.*

The reviewer correctly identifies why the results are positive: dishonest traders do not execute their adversarial role perfectly. The contribution is not that consensus resolution provides formal guarantees — it does not — but that the LMSR structure amplifies residual truth-signal into systematic wealth transfer. We will reframe Section 6.3 accordingly.

Defining p_adv = P(malicious agent votes wrong): at p_adv = 1.0 the symmetry holds; break-even for 2v3 (p_h = 0.80) is p_adv < 0.700. Estimated from Table 2: Llama 3 8B ≈ 0.63, GPT-OSS 20B ≈ 0.42, GPT-OSS 120B ≈ 0.17 — all below break-even because these LLMs are insufficiently adversarial for the symmetry to activate, not because the mechanism is theoretically sound.

The episode logs (supplementary, `traces/`) confirm this: a Blind Malicious agent acknowledges its adversarial role, identifies the correct answer, then votes for it. Prompting a model to identify truth in order to oppose it causes it to vote for truth — the property we should have foregrounded.

**On the p_adv trend.** Llama 3 8B ≈ 0.63, GPT-OSS 20B ≈ 0.42, GPT-OSS 120B ≈ 0.17 — larger models resist adversarial prompting more strongly (stronger RLHF alignment). The market therefore partly measures alignment strength. We will frame this explicitly: the contribution is that LMSR dynamics *amplify* existing alignment into systematic influence transfer.

Debate agents also exhibit p_adv < 1.0; the market vs. debate comparison tests relative exploitation of the same regularity. We will acknowledge this explicitly.

**On the scaling trajectory.** As models improve at adversarial instruction-following, p_adv will increase toward 1.0, at which point the mechanism collapses by our own analysis. We will state this in Section 7 as a boundary condition: the contribution is robust for current LLMs but not guaranteed to persist. We note that RLHF alignment and adversarial instruction-following may be in tension — models harder to make adversarial are also better honest agents, so both desiderata may not degrade simultaneously.

---

## Q2: Confidence Intervals

We apologise for this omission. Key results (3 runs; supplementary, Section 2): TruthfulQA 2v3 Informed 54.00% [49.8%, 58.2%] — above chance and far above voting (0.3%). We must correct a misstatement: GPQA Binary 2v3 Informed 39.13% [32.2%, 46.1%] has a lower CI bound below 50% chance — this is *not* statistically above chance at 95% confidence. Correct framing: GPQA Binary reflects partial adversarial resistance, not truth recovery; the market still substantially outperforms voting (2.3%). Full CI columns will be added to all appendix tables; GPT-OSS CIs via 3-seed replication.

---

## Q3: Debate Baseline Consistency

The semantic market (Section 5) is a two-agent MM↔Trader dialogue; a multi-agent debate baseline requires N>2 agents with opposing roles — a categorically different framework. The standalone Market Maker is the appropriate baseline, which Tables 1 and 6 use. The LMSR market (Section 6) has N=5 agents with adversarial roles, making debate and voting natural comparisons; both are in Appendix Tables 3–5. Clarifying paragraphs will be added to both sections.

---

## MCQ Degradation

GPQA MCQ (2v3 Informed): 21.3% for Llama 3 8B, below 25% random baseline. The mechanism fails here; we do not dispute it. Binary markets allow honest capital to automatically short the false outcome; with k=4 options, honest capital fragments while an adversary concentrates. The abstract and contributions will scope robustness claims to binary settings and frame MCQ as an open problem.

---

## Revision Commitments

- Section 6.3: reframe as empirical; add break-even derivation, p_adv estimates, and simulation figure
- Appendix B.2: explicitly state that agents are not told their capital balance; they observe prices and role prompt only; wealth redistribution occurs at settlement after trades are committed
- Section 7: flag reliance on adversarial imperfection; rational adversaries could defeat the mechanism
- Abstract: scope robustness claims to binary settings; frame MCQ as open problem
- CIs: add to all appendix tables; 3-seed protocol for GPT-OSS models
- Debate baseline: clarifying paragraph in Sections 5 and 6.1
- **Reframe**: foreground the finding that LLMs struggle to execute adversarial roles faithfully as the primary contribution, with LMSR dynamics amplifying this property into influence transfer
