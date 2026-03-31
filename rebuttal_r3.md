# Rebuttal — Reviewer 3

We are grateful for the thoughtful and generous review. The suggestions are all actionable and we address each in turn.

---

## Central Figure

We agree this would significantly help readers orient early. We will add a summary figure (end of Section 1 or beginning of Section 3) covering: (a) the semantic MM↔Trader dialogue with a Brier-score comparison; (b) the LMSR mechanism with agent roles and price dynamics; and (c) a headline comparison of market vs. debate vs. voting across adversarial conditions. This gives readers a visual anchor before the technical details.

---

## Conditions Where the Mechanism Fails — Moving to the Introduction

This is good pedagogical advice. The limitations are currently confined to Section 7, which means readers encounter them only after investing in the full technical presentation. We will move a brief, honest summary of the key failure modes into Section 1 or the abstract: (1) the mechanism assumes adversarial agents are imperfect (i.e. not fully rational wealth-maximisers); (2) performance degrades significantly in multi-choice settings due to liquidity fragmentation; (3) very weak market makers may be unable to distinguish valid from deceptive reasoning in the semantic variant. Flagging these upfront sets appropriate expectations and, as the reviewer notes, frees up working memory to engage with what the mechanism does well.

---

## Related Literature

We thank the reviewer for these pointers and will conduct a broader sweep before the revision.

**Todasco (2025):** This work uses stake size as a confidence signal from individual LLMs, showing that larger bets correlate with higher accuracy (~99% at maximum stakes vs. 74% at minimum stakes). The conceptual connection to our work is real — both treat financial commitment as a proxy for epistemic certainty — but the mechanisms are distinct. Todasco extracts confidence from a *single* agent through stake calibration; our paper uses *multi-agent* LMSR markets for collective truth elicitation and adversarial robustness. The single-agent result is complementary: it suggests that the financial penalty structure in our LMSR setting may have additional confidence-calibrating effects beyond what we currently measure. We will position this as related but distinct in Section 2.2.

**OpenForecaster:** This project trains a specialised 8B forecasting model optimised for Brier score and calibration using GRPO reinforcement learning. It addresses the complementary question of *training* models to be well-calibrated, whereas our work addresses *coordinating* models that were not specifically trained for forecasting. We will cite this in the related work as an orthogonal approach to the same calibration goal.

---

## Moderators of Mechanism Efficacy

These are valuable questions we will address in an expanded discussion. On **time horizon**: our simulations suggest wealth divergence is visible within 20–30 episodes for typical adversarial effectiveness values; we will add a figure showing this. On **external reward structure**: our results already show ground-truth resolution outperforms consensus, consistent with the reviewer's intuition; we will discuss this as a direction for future work. On **capability differential**: Section 5's weak-to-strong experiments begin to address this (GPT-4.1-nano successfully extracts signal from GPT-4.1-standard under adversarial conditions); we will expand the discussion of where this breaks down when the capability gap becomes extreme.

---

## Title

We appreciate the suggestion and agree the current title under-sells the comparative result. We will consider a revised title that more directly signals the head-to-head finding, while remaining consistent with the venue's conventions.

---

## Revision Commitments

- Add a central summary figure (mechanisms + headline comparison) in Section 1 or 3
- Move key failure modes to the introduction: adversarial imperfection assumption, MCQ degradation, semantic market maker capability threshold
- Add Todasco (2025) and OpenForecaster to Section 2.2 with appropriate positioning
- Conduct a broader related-work sweep for prediction-market-as-inference literature
- Expand Section 7 discussion of moderators: episode count, external reward structure, capability differential limits
- Consider a more descriptive title that reflects the comparative result
