# PatchSteg ICML Workshop Readiness Review

Date: 2026-04-29

Scope: current `paper/main.tex`, local `icml2026/` template, current experiment log, and the ICML 2026 Mechanistic Interpretability Workshop CFP.

## Pre-ICML Fix List To Maximize Acceptance Odds

No set of edits can guarantee acceptance. Workshop decisions depend on reviewer assignment, submission pool, fit, and taste. The closest practical target is to remove obvious rejection reasons, make the core claim easy to review positively, and ensure the evidence is disciplined. The current ICML-styled manuscript is plausible for the ICML 2026 Mechanistic Interpretability Workshop, but it should still be tightened before submission.

Use this section as the submission checklist. Items marked P0 are acceptance-critical; P1 items materially improve the odds; P2 items are useful if time remains.

### P0: Remove Likely Rejection Triggers

- **Add a "Hypotheses and Evidence" table to the main text.**
  - Why: the MechInterp CFP explicitly values falsifiable hypotheses and careful evidence boundaries.
  - What to add: a compact table with rows such as:
    - H1: local VAE latent perturbations survive decode-reencode for some positions.
    - H2: stability is spatially non-uniform.
    - H3: stability is content-dependent.
    - H4: reconstruction error does not explain carrier quality.
    - H5: local geometry partially predicts carrier quality.
  - Columns: evidence, dataset/protocol, support level, limitation.
  - Done when: a reviewer can identify the scientific claim and evidence without reconstructing it from scattered paragraphs.

- **Add a "Protocol Summary" table for all headline numbers.**
  - Why: the paper currently mixes results from different sample sizes, resolutions, and detectors. That is manageable only if transparent.
  - Include: dataset, number of images, resolution, VAE, carriers, epsilon, carrier-selection method, detector/features if applicable, metric, confidence interval if available.
  - Done when: no headline result appears in the paper without an associated protocol.

- **Strengthen the limitation language around security claims.**
  - Replace any remaining broad phrasing like "monitoring failure", "covert channel", or "survives deployment" with bounded phrasing.
  - Suggested phrasing: "under this shared-cover VAE setting", "under evaluated detector families", "in our current distortion suite", "pilot evidence".
  - Done when: a skeptical reviewer cannot reasonably say the paper claims broad undetectability or deployable steganography.

- **Make the shared-cover assumption impossible to miss.**
  - Current baseline PatchSteg requires a clean cover latent or clean round-trip baseline.
  - Put this in the method, limitations, and maybe a small threat-model paragraph.
  - Explain that this is acceptable because the paper studies latent stability, not a complete real-world covert messaging system.
  - Done when: the assumption is presented as part of the probe design, not as a buried caveat.

- **Remove or further demote small-n extension claims from the main narrative.**
  - CDF, adaptive encoding, sanitizer, and defense results should not drive acceptance unless re-run at scale.
  - Keep them in one short "Exploratory Extensions" paragraph or appendix.
  - Avoid "first", "provable", "undetectable", "SOTA", "universal", and "defeats" in the main text.
  - Done when: the main paper can stand entirely on latent-stability results without relying on small-n extensions.

- **Verify double-blind safety.**
  - Search the manuscript and appendix for names, GitHub usernames, HuggingFace usernames, hackathon names, file paths that reveal identity, acknowledgements, and teammate references.
  - Do not link directly to the named GitHub repo in the blind PDF. Use an anonymized code archive if submitting code.
  - Done when: `pdftotext paper/main.pdf - | rg "Joy|J0YY|PatchSteg repo|GitHub|Hackathon|Wang|Team"` returns nothing identifying, except generic non-identifying terms if deliberately included.

- **Create an anonymized reproducibility package or code note.**
  - The CFP says reviewers will consider reproducibility and code/data access.
  - Use an anonymized GitHub mirror or anonymous.4open.science archive.
  - Include scripts for the headline figures/tables only, not every experimental prototype.
  - Done when: a reviewer can reproduce at least the stability map, random-vs-stable carrier comparison, 300-image evaluation summary, and multi-VAE figure.

### P1: Make The Paper Scientifically Stronger

- **Run one clean random-vs-stability carrier ablation on the same natural-image protocol.**
  - Current evidence says stability selection helps, but some examples are synthetic or old-protocol.
  - Target: 100-300 CIFAR-10 images, same VAE, same resolution, same K, same epsilon values.
  - Report random carriers vs stability-selected carriers with bootstrap CIs.
  - Acceptance impact: high. This directly validates the core method as an interpretability probe.

- **Re-run or expand the Jacobian analysis so it is not the weakest link.**
  - Current result is suggestive: carrier Jacobian norms 2050 vs 1684 with `p=0.058`.
  - Better target: compute local directional gain for the actual perturbation direction, or approximate the Jacobian of `Enc o Dec` rather than decoder-only sensitivity.
  - Report effect size, confidence interval, and correlation with stability.
  - Acceptance impact: high. This turns the paper from "interesting empirical phenomenon" into stronger mechanistic evidence.

- **Add a small predictive model for carrier stability.**
  - Features: entropy, high-frequency energy, local variance, decoder sensitivity, baseline drift, channel statistics.
  - Predict: per-position or per-image bit accuracy/stability.
  - Report held-out correlation or ranking accuracy.
  - Acceptance impact: high. Interpretability reviewers like when observations become predictive.

- **Normalize the detector story.**
  - Current manuscript wisely narrows detectability, but reviewers may still ask why numbers differ across experiments.
  - Add one table or appendix section: detector family, dataset, n, attack epsilon, AUC.
  - Explicitly state that stronger pixel/spectral detectors can detect low-epsilon variants in some protocols.
  - Acceptance impact: medium-high. Prevents security reviewers from objecting.

- **Add native high-resolution images if feasible.**
  - CIFAR-10 upscaling is a known weakness.
  - Even 50-100 native 256 or 512 images from COCO/ImageNet/LAION-style sources would help.
  - Report whether the content-dependence trends hold.
  - Acceptance impact: medium-high. This reduces "artifact of upscaled CIFAR" skepticism.

- **Add SSIM and LPIPS alongside PSNR for visual quality.**
  - PSNR alone is not enough for image quality claims.
  - Keep the language conservative: "low distortion by PSNR/SSIM/LPIPS", not "imperceptible" unless human-tested.
  - Acceptance impact: medium. Helps with vision/steganography reviewer expectations.

- **Improve references and related work.**
  - Add more direct citations on mechanistic interpretability of generative models, autoencoder latent geometry, representation interventions, and image steganalysis.
  - Make sure every "closest prior" statement is supported.
  - Acceptance impact: medium. Prevents reviewer complaints that the paper is isolated from the field.

### P2: Polish And Presentation

- **Use the two remaining ICML pages strategically.**
  - The current PDF is 6 pages. The long-paper limit is 8 ICML pages excluding references/appendix.
  - Add at most 1-1.5 pages of high-signal material:
    - Hypotheses/evidence table.
    - Protocol summary table.
    - One stronger ablation or Jacobian result.
  - Leave a little white space rather than filling every inch with lower-quality extensions.

- **Tighten figure captions.**
  - Captions should say what the result means, not just what is plotted.
  - Include sample size and protocol in captions where possible.
  - Remove any "hero" phrasing from the scientific manuscript.

- **Add a short "Why this is mechanistic interpretability" paragraph.**
  - State: the intervention is on an internal latent state; recovery measures whether that intervention is preserved by the model computation; spatial/content variation reveals structure of the representation.
  - Place it near the end of the introduction or start of method.

- **Add an ethics/safety note.**
  - Because the paper studies covert channels, include responsible disclosure style language:
    - the evaluated channel has limiting assumptions;
    - code release should avoid turnkey misuse where possible;
    - the purpose is monitoring and interpretability.
  - Keep it concise and factual.

- **Clean the appendix.**
  - Put all small-n tables and extra attack/defense variants in appendix.
  - Label them "pilot" or "exploratory".
  - Do not expect reviewers to read the appendix; the main paper must be self-contained.

- **Run final PDF compliance checks.**
  - `./build.sh`
  - `pdfinfo paper/main.pdf`
  - `pdffonts paper/main.pdf`
  - `pdftotext paper/main.pdf - | rg "AUTHORERR|Title Suppressed|Missing|J0YY|Hackathon|Wang|Team"`
  - Confirm main body is within page limit and one PDF includes references/appendix if submitting appendix.

### Suggested Minimum Submission Patch

If time is very limited, do only these before submitting:

1. Add a main-text "Hypotheses and Evidence" table.
2. Add a main-text or appendix "Protocol Summary" table.
3. Add an anonymized reproducibility/code note.
4. Strengthen shared-cover and detector-scope caveats.
5. Add one clean random-vs-stability ablation on a consistent natural-image protocol.
6. Rebuild and re-check anonymity.

This minimum patch is the highest-return path. It directly addresses the workshop's evaluation criteria without expanding the paper into another unfocused systems draft.

### Reviewer Objections To Preempt

- **"This is steganography, not interpretability."**
  - Preempt by defining PatchSteg as an intervention probe of `Enc o Dec`, with bit recovery as the observable.

- **"The method assumes the receiver has the clean cover."**
  - Preempt by stating this is a shared-cover probe assumption, not a claim of complete deployment realism.

- **"The security claims are overstated."**
  - Preempt by bounding every detector and robustness claim to the evaluated protocol.

- **"The Jacobian mechanism is not established."**
  - Preempt by presenting current Jacobian evidence as partial and adding stronger composite-Jacobian analysis if possible.

- **"CIFAR upscaling may create artifacts."**
  - Preempt by acknowledging it and adding native-resolution images if possible.

- **"Small-n results should not support big claims."**
  - Preempt by moving small-n extension results to appendix and labeling them pilot evidence.

### Acceptance Probability After Fixes

Estimated probabilities are subjective, but useful for prioritization:

- Current ICML-styled version: approximately 45-60%.
- After the minimum submission patch above: approximately 60-70%.
- After P0 plus the most important P1 experiments: approximately 70-80%.

These are not guarantees. They assume the final submission is anonymous, formatted correctly, reproducible enough for reviewers, and clearly framed as mechanistic interpretability rather than a broad security claim.

## Change Log: ICML-Styled Regeneration

Implemented after the initial review:

- Replaced the old article-style `paper/main.tex` with an ICML 2026 workshop-style manuscript using `../icml2026/icml2026`.
- Removed the manually printed hackathon author block and switched to anonymized ICML author metadata.
- Reframed the paper from an attack/defense systems paper into an interpretability-oriented study of spatially localized information preservation in diffusion VAE latents.
- Rewrote the title, abstract, introduction, contribution list, and conclusion around PatchSteg as a latent-stability intervention probe.
- Made the stability score the central mathematical object and explicitly defined the shared-cover assumption for baseline decoding.
- Consolidated the main results around carrier stability, content dependence, 300-image CIFAR-10 evaluation, multi-VAE generality, reconstruction-error mismatch, local geometry, and bounded monitoring implications.
- Moved small-n CDF, adaptive payload, and sanitizer material out of the main claim path and into a short exploratory/extensions discussion plus appendix note.
- Calibrated overstrong language: the new version avoids broad "undetectable", "SOTA", and unconditional defense claims in the main text.
- Added `paper/main.bib` and switched the manuscript to BibTeX-style references using the local ICML bibliography style.
- Rebuilt `paper/main.pdf` from the ICML source. The regenerated PDF is 6 pages, US letter size, about 1.23 MiB, and uses the ICML review footer/header behavior.

## Executive Assessment

The current draft has enough raw material for a credible ICML workshop submission, but not in its current form. The main blocker is not the idea; it is framing and evidence discipline. Right now the manuscript reads like a full security/steganography systems paper with interpretability observations mixed in. For an interpretability workshop, the paper needs to become a focused study of VAE internals: which latent directions and spatial positions survive decode-reencode, why this creates a communication channel, and what this teaches us about representation geometry and monitoring.

Best submission target: long paper, not short paper, if the page budget is 8 ICML pages excluding references and appendices. A 4-page short paper would need to drop most defenses and variants and present one clean phenomenon.

Recommended title direction:

> PatchSteg: Probing Spatially Localized Information Preservation in Diffusion VAE Latents

or:

> Covert Channels as Probes of Latent Stability in Diffusion Autoencoders

The current title is too attack-forward for a mechanistic interpretability venue. Keep the security motivation, but lead with what the channel reveals about model internals.

## Fit To ICML 2026 Style

The current paper is not in ICML style:

- `paper/main.tex:1` uses `\documentclass[11pt,a4paper]{article}` instead of the ICML template's `\documentclass{article}` plus `\usepackage{icml2026}`.
- `paper/main.tex:7` uses manual `geometry`, which ICML explicitly says not to alter.
- `paper/main.tex:14-16` prints title, author, and date manually. The ICML template uses `\twocolumn[...]`, `\icmltitle`, `\icmlauthor`, `\icmlaffiliation`, and `\printAffiliationsAndNotice{}`.
- The current rendered `paper/v1.pdf` is 27 A4 pages, uses 11pt single-column article style, and is not close to the workshop page budget.
- The abstract is one paragraph, but it is far too long for ICML style. The template says abstracts should be roughly 4-6 sentences.
- Captions and tables broadly follow the right convention, but many figures are too large and too numerous for an 8-page two-column manuscript.
- References are handwritten in `thebibliography`; use BibTeX with `icml2026.bst` or at least a `.bib` file for complete references.
- The draft is not anonymized. `paper/main.tex:15` identifies "Team PatchSteg / Varsity Hackathon 2026"; MechInterp requires double-blind submissions.

Required LaTeX conversion:

```tex
\documentclass{article}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{hyperref}
\newcommand{\theHalgorithm}{\arabic{algorithm}}
\usepackage{../icml2026/icml2026}
\usepackage{amsmath,amssymb,mathtools}
\usepackage[capitalize,noabbrev]{cleveref}

\icmltitlerunning{Latent Stability in Diffusion VAEs}

\begin{document}
\twocolumn[
\icmltitle{...}
\begin{icmlauthorlist}
  \icmlauthor{Anonymous Authors}{anon}
\end{icmlauthorlist}
\icmlaffiliation{anon}{Anonymous Institution}
\icmlkeywords{mechanistic interpretability, autoencoders, latent representations}
\vskip 0.3in
]
\printAffiliationsAndNotice{}
```

For a blind submission, the ICML style hides author information, but the manuscript body must also remove self-identifying text, repo names that reveal identities, acknowledgements, teammate names, and hackathon references.

## Fit To Mechanistic Interpretability Workshop

The MechInterp CFP says submissions should further the ability to use internal states of neural networks to understand them. It explicitly values falsifiable hypotheses, careful evidence boundaries, strong baselines, reproducibility, code/data access, and clear limitations. It also says work that downplays limitations will not be accepted.

PatchSteg can fit if framed as:

1. A behavioral probe of VAE latent stability under the composite map `Enc(Dec(z + delta))`.
2. A measurement of spatially local and content-dependent information preservation in diffusion autoencoders.
3. A case study showing that interpretability of latent geometry matters for monitoring and safety, because small internal-state edits can survive through image space.

PatchSteg fits poorly if framed primarily as:

- "We created a covert channel."
- "This is empirically undetectable."
- "This defeats SOTA defenses."
- "This is the first/provable/undetectable method" without much stronger evidence.

The workshop is likely receptive to the phenomenon, but reviewers will punish overclaiming and small-n evidence.

## Current Scientific Strengths

The strongest parts of the current work are:

- The core channel is simple and explainable: 4D VAE latent vector, signed directional perturbations, decode, re-encode, projection-based recovery.
- The spatial carrier-stability result is genuinely interesting: not every latent position has equal survival under round-trip.
- Content dependence is valuable: gradients and smooth images are poor carriers, while textured/natural images have high reliability.
- The result connects naturally to local decoder/encoder geometry through the composite Jacobian of `Enc o Dec`.
- The multi-model VAE tests help show this is not a one-checkpoint accident.
- The codebase appears to contain runnable experiments and generated figures, which is a major plus for workshop review.

The best interpretability contribution is not "steganography exists"; it is:

> Diffusion VAEs have spatially non-uniform, content-dependent latent directions that are preserved through image-space reconstruction, and covert-channel recovery gives a quantitative probe of this preservation.

## Major Acceptance Risks

### 1. The paper has no stable thesis

The manuscript currently contains at least four papers:

- Baseline PatchSteg attack.
- Distribution-preserving CDF-PatchSteg.
- Latent steganalysis and sanitization defenses.
- Mechanistic/interpretability analysis of VAE latent stability.

For an interpretability workshop, these should not have equal weight. The paper needs one thesis, one main table/figure flow, and one claim hierarchy.

Recommended thesis:

> We use steganographic bit recovery as an intervention-based probe of information-preserving latent directions in pretrained diffusion VAEs. The probe reveals spatially non-uniform and content-dependent stability not explained by reconstruction error alone.

Everything else becomes either motivation, stress test, or appendix.

### 2. Claims outrun evidence

Examples:

- `paper/main.tex:57` says CDF-PatchSteg is the first distribution-preserving method for post-hoc VAE latent editing. This needs either careful literature support or softer wording.
- `paper/main.tex:321` says marginal moment preservation is the "minimal sufficient condition"; the evidence does not establish minimality or sufficiency against broad detectors.
- `paper/main.tex:588-590` calls CDF empirically undetectable using `n=4`. That is too strong.
- `paper/main.tex:697` says "all four SOTA attacks" with `n=4`; this reads inflated.
- `paper/main.tex:837` says the defense has an unconditional theoretical guarantee, but the empirical setup and actual decode pipeline include VAE round-trip effects and estimated means. The guarantee is for an idealized CDF half-membership channel, not automatically the full image pipeline.

Use language like "under this detector family," "in this small-scale experiment," "for the idealized CDF carrier model," and "consistent with" instead of "proves," "undetectable," or "SOTA."

### 3. Detectability story is internally inconsistent

The draft says:

- Simple latent logistic detector: eps=1 AUC 0.44, eps=2 AUC 0.68, eps=5 AUC 0.93.
- All-attacks naive detector: PatchSteg eps=2 AUC 0.556, eps=5 AUC 1.000.
- Stronger detection: "Even strongest detector struggles at eps=1.0" in the caption, but the experiment log says pixel-residual and spectral detectors achieve AUC=1.0 even at eps=1.0.
- LSB comparison: PatchSteg eps=2 AUC 0.349, eps=5 AUC 0.807.

These may all be true under different data sizes, features, image sets, and resolutions, but the manuscript currently reads like contradiction. Reviewers will see this as unreliability unless every detectability number is reported in a single normalized evaluation table with columns for dataset, n, resolution, feature set, detector, CV protocol, and attack configuration.

### 4. The interpretability evidence is underdeveloped

The mechanistic section currently has useful observations but not enough causal/mechanistic rigor:

- Per-channel importance is too coarse for a strong interpretability claim.
- Direction robustness on one color-patch image is weak.
- Reconstruction error not predicting stability is interesting, but the next step should identify what does.
- Jacobian analysis is promising, but the text does not report the actual statistical result near the figure.

The paper needs to make the stability metric mathematically central:

\[
S_{r,c,d,\epsilon}(x)=\langle Enc(Dec(z+\epsilon d_{r,c}))_{r,c} - Enc(Dec(z))_{r,c}, d\rangle
\]

Then test hypotheses:

- H1: `S` varies significantly across spatial positions.
- H2: `S` is content-dependent and predictable from image statistics.
- H3: `S` is better explained by local Jacobian/alignment properties than by reconstruction error.
- H4: high-`S` positions causally support bit recovery.

### 5. Too many weak small-n sections dilute the strong result

Several sections use `n=3`, `n=4`, or `n=6`. In a workshop paper this is acceptable only if clearly positioned as pilot analysis, but currently these sections drive large claims.

Move these to appendix or compress:

- CDF experiments on 4 images.
- CertifiedPurifier on 4 images.
- Teammate defense evaluation on 3 images.
- All-attacks naive detector on 6 images.
- QuantileShuffleSanitizer on 4 images.

Keep in main text:

- Main PatchSteg stability and recovery.
- 300-image natural evaluation.
- Multi-model evaluation.
- Content/Jacobian analysis.
- One consolidated detection caveat table.

### 6. The threat model needs tightening

The method assumes the receiver has the clean cover latent or clean image baseline for baseline PatchSteg. That is a major operational assumption. It is stated, but it needs to be elevated:

- Is this a same-cover communication channel where both agents see the cover image?
- Can the receiver reconstruct carriers without the exact cover?
- Does the sender need to transmit/stage a cover image first?
- Which variants are blind-decodable?

For an interpretability paper, this matters less as a security claim and more as experimental setup. For a security/steganography claim, it is central.

## Recommended 8-Page Long Paper Structure

Target structure:

1. Introduction: one page. Lead with latent-state interpretability and safety motivation. End with three contributions.
2. Related Work: half page. Mechanistic interpretability of generative models, latent autoencoder geometry, neural steganography/watermarking.
3. PatchSteg as a Latent-Stability Probe: one page. Define the intervention, recovery metric, stability score, and carrier selection.
4. Experimental Setup: half page. Datasets, VAE backbones, metrics, detectors, bootstrap/CV details.
5. Results: two to three pages.
   - Carrier stability is spatially non-uniform.
   - Stability is content-dependent and predicts bit recovery.
   - Stability generalizes across VAE backbones.
   - Robustness/detectability tradeoff, carefully bounded.
6. Mechanistic Analysis: one to two pages.
   - Reconstruction error is not enough.
   - Local decoder or encode-decode Jacobian predicts carriers.
   - PCA/direction findings if they support geometry.
7. Implications, Limitations, and Defenses: one page. Security implications and bounded defense observations.
8. Conclusion: short.

Appendix:

- Full attack variants.
- CDF-PatchSteg.
- Quantile sanitizer.
- Teammate defense tables.
- Full figures and implementation details.

## Recommended Main Figures

Use at most 5 main figures/tables:

1. Pipeline figure: compact, two-column top figure.
2. Stability map plus carrier selection and bit recovery.
3. Natural-image evaluation: per-class accuracy and histogram.
4. Content/Jacobian figure: predictors of capacity and Jacobian comparison.
5. Consolidated robustness/detectability table: all protocols in one place.

Move all visual examples and most defense heatmaps to appendix. The current manuscript has too many figures for an ICML-style workshop paper.

## Concrete Experiments To Add Or Re-run

Highest priority:

1. Normalize the detection suite.
   - Same dataset, same resolution, same train/test split, same K, same epsilon values.
   - Report AUC with confidence intervals for each detector.
   - Include pixel residual and spectral detectors in the same table as latent-statistics LR.

2. Strengthen Jacobian evidence.
   - Estimate the Jacobian of `Enc(Dec(z))` with respect to local latent perturbations, not only decoder pixel sensitivity if feasible.
   - Correlate local Jacobian singular values or directional gain with stability.
   - Report effect sizes and p-values in text.

3. Scale CDF-PatchSteg beyond `n=4` if it remains a main claim.
   - At minimum 100-300 natural images, same as baseline.
   - Otherwise move it to appendix and call it a pilot distribution-matching variant.

4. Run native high-resolution images.
   - CIFAR-10 upscaled to 256x256 is acceptable for a first pass but weak for vision/steganography claims.
   - Add MS-COCO, ImageNet validation samples, LAION aesthetic subset, or any native 256/512 image set.

5. Add a clean ablation table.
   - Random vs stability-selected carriers.
   - Global random direction vs PCA direction vs local/Jacobian direction.
   - With and without repetition.
   - Same dataset and protocol.

Medium priority:

6. Report SSIM/LPIPS alongside PSNR.
   - PSNR alone is not enough for "imperceptible".

7. Add a blinded human visual check only if easy.
   - Even a small 2AFC perceptual check would reduce reviewer skepticism about imperceptibility.

8. Add a baseline from a known steganography package or paper if feasible.
   - LSB is useful as a sanity baseline but not enough for steganography reviewers.

## Claim Calibration

Replace:

> undetectable

with:

> not detected by the evaluated detector family

Replace:

> SOTA attacks

with:

> evaluated latent steganography variants

Replace:

> provable defense against CDF-PatchSteg

with:

> a defense with a simple guarantee for the idealized Gaussian half-space carrier model

Replace:

> first distribution-preserving method

with:

> to our knowledge, a post-hoc distribution-matching variant; unlike generation-time watermarking methods, it modifies existing images

Replace:

> practical operating point

with:

> representative operating point under the evaluated detector and dataset

## Abstract Rewrite Direction

Current abstract is too long and attack-heavy. A stronger workshop abstract:

> We study whether information inserted into the latent representation of a pretrained diffusion VAE survives a decode-reencode round trip, and use this as an intervention-based probe of latent stability. We introduce PatchSteg, a simple training-free probe that encodes bits by adding signed perturbations to selected spatial latent positions and decodes them from the re-encoded image. Across Stable Diffusion VAE backbones, recovery is highly non-uniform across spatial positions and strongly content-dependent: textured images support reliable recovery while smooth images do not. Reconstruction error does not explain carrier quality; instead, local encode-decode geometry and image statistics better predict which positions preserve perturbation direction. These findings reveal a concrete representation-level failure mode for monitoring image-mediated model communication, while also providing a quantitative probe for studying information preservation in generative autoencoders.

This version is closer to MechInterp: it states a falsifiable object of study and avoids overclaiming detectability.

## Suggested Contribution List

Use three contributions, not seven:

1. We introduce a training-free intervention probe for measuring local information preservation in diffusion VAE latents via decode-reencode bit recovery.
2. We show that latent stability is spatially non-uniform, content-dependent, and not explained by reconstruction error alone, with evidence across natural images and multiple VAE backbones.
3. We characterize the safety implication: these stable latent directions instantiate a bounded covert channel, and simple detectors/sanitizers reveal a reliability-stealth-fidelity tradeoff.

## Specific Line-Level Issues

- `paper/main.tex:14`: Title should be shorter and less security-only.
- `paper/main.tex:15`: Remove author/team/date for double blind.
- `paper/main.tex:22`: Abstract is overloaded; compress to 4-6 sentences and avoid small-n CDF/defense claims.
- `paper/main.tex:33-41`: Seven contributions is too many. Reduce to three.
- `paper/main.tex:53-59`: Related work is too short and missing interpretability of generative models/autoencoders.
- `paper/main.tex:98-112`: CapacityPatchSteg and AdaptivePatchSteg should move to appendix or code-release note, not main method.
- `paper/main.tex:117-126`: CDF method is interesting, but it should not be central unless scaled.
- `paper/main.tex:265-333`: Detection protocol needs consolidation and contradiction cleanup.
- `paper/main.tex:350`: Direction-vector robustness on one synthetic image is not enough to claim isotropy.
- `paper/main.tex:548`: The JND claim is risky and should be cited or removed.
- `paper/main.tex:559-779`: "Promising Extensions" and defense material should mostly become appendix for MechInterp.
- `paper/main.tex:728`: Remove teammate name for double blind and because it reads informal.
- `paper/main.tex:831`: Limitations section is good, but should appear earlier or be integrated more visibly.
- `paper/main.tex:839-876`: References need complete metadata and BibTeX formatting.

## Acceptance-Oriented Revision Plan

Phase 1: Reframe and format.

- Convert to ICML 2026 template.
- Anonymize.
- Cut to 8 pages main text.
- Rewrite title, abstract, introduction, and contributions around latent stability/intervention probing.

Phase 2: Stabilize claims.

- Create a single "Claims and Evidence" table.
- Mark every result by dataset size and protocol.
- Move small-n defenses/CDF to appendix unless re-run at scale.

Phase 3: Strengthen interpretability.

- Make stability score the central object.
- Add formal hypotheses and corresponding tests.
- Expand Jacobian analysis and report statistics.

Phase 4: Polish.

- Replace handwritten bibliography.
- Reduce figures.
- Add appendix with reproducibility details, code pointer, hyperparameters, seeds, and full tables.
- Verify page count, fonts, anonymity, and PDF size.

## Bottom Line

This can become a solid workshop paper if it becomes a focused interpretability study with a security-motivated probe. The current version is too broad and too assertive. The highest-leverage change is to stop treating every repo feature as a paper contribution. Put the strongest phenomenon in the center: pretrained diffusion VAEs preserve some local latent perturbations much more than others, and that stability is measurable, content-dependent, and mechanistically tied to encode-decode geometry.
