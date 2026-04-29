# PatchSteg → 3 ICML 2026 Workshops: Codex Task Spec

Three workshop submissions sharing ~70% of the body. Build order below is
priority order, not chronological. Tasks marked [SHARED] feed all three
papers; tasks marked [WORKSHOP] are workshop-specific.

## Repo setup

- Three branches off current draft: `aiwild`, `ai4good`, `fmai`
- Shared `core/` for figures, tables, body sections reused across all three
- Single `experiments/` directory; results organized by experiment name
- Each branch owns its own intro, related work, threat model wrapper, discussion

---

## Experimental tasks (in priority order)

### [SHARED] P1: Add LPIPS and SSIM to all image-fidelity reporting

Currently the paper reports PSNR only. Limitations section flags this.

- Dependencies: `lpips` (Alex backbone), `pytorch_msssim`
- Re-run 300-CIFAR eval at ε=2.0 and ε=5.0 reporting PSNR, LPIPS, SSIM
- Re-run cross-VAE eval with same three metrics
- Update Table 1, Figure 4, all reconstruction-quality claims
- Pure metric addition, no new data collection

### [SHARED] P2: Detector reporting beyond AUC

Reviewers will want operating-point info, not just AUC.

- Compute precision and recall at fixed FPR thresholds (0.1%, 1%, 5%) for each ε
- Report confusion matrices at the 1% FPR operating point
- Same data as existing detector sweep, just different summary stats

### [SHARED] P3: Robustness table refresh with realistic distortions

Current robustness tests use JPEG Q=10 and aggressive resize. Replace with realistic deployment distortions.

- Add: PNG→JPEG Q=70→PNG roundtrip
- Add: social-media-style downscale (resize to 1024 max edge) + JPEG Q=80
- Keep: VAE re-encode, screenshot simulation, additive noise σ=0.10
- Drop or move-to-appendix: JPEG Q=10 (too unrealistic)
- Optional if fast: one neural-compression baseline (e.g., `compressai` bmshj2018)

### [FMAI] P4: Agent-in-loop demonstration

Highest-leverage new result. Without this, the FMAI paper is "theoretically a covert channel exists." With this, it's "here is the channel running through a realistic agent pipeline."

- Two VLM agents (use Claude Sonnet 4.5 or GPT-4V via API) with PatchSteg embed/extract as a tool
- Shared workspace contains an image and a caption (mimics a multi-agent platform)
- Third agent acts as monitor: sees pixels + caption, outputs flagged/not-flagged + natural-language description of image
- Task: sender embeds 32-bit secret payload into a cover image, posts to workspace with innocuous caption; receiver retrieves and extracts; monitor inspects
- Metrics:
  - End-to-end payload recovery rate (target ≥95% at ε=2.0)
  - Monitor flag rate on stego vs clean (target: indistinguishable)
  - Monitor's natural-language descriptions: do they mention anything visually wrong? (qualitative table)
  - Latency and cost per message (operational realism)
- Scale: 50 trials, 32-bit payloads, 5 cover images, varied captions and payloads
- Output: one figure + one table for FMAI paper, one figure for AI4GOOD paper if time permits

### [FMAI] P5: Sanitizer Pareto curves (fidelity vs disruption)

Current sanitization is described as "preliminary safety probes." FMAI's "verified fixes" output requires a real Pareto.

- Sanitizers to evaluate:
  - Latent smoothing (Gaussian blur on latent), σ ∈ {0.1, 0.3, 1.0}
  - Quantile reflection (existing implementation)
  - Additive Gaussian noise on latent, σ ∈ {0.05, 0.1, 0.3}
  - JPEG re-encode at Q ∈ {95, 85, 70, 50}
  - SDEdit low-strength regeneration (strength ∈ {0.1, 0.2, 0.3}) — most expensive, do last
- Metrics axes:
  - X-axis: bit accuracy after sanitization (lower = better defense)
  - Y-axis: LPIPS distance from original clean image (lower = better fidelity preservation)
  - Each sanitizer is a curve across its parameter sweep
- Scale: 100 stego images at ε=2.0 across natural-image set
- Output: single Pareto plot. Identify which sanitizers Pareto-dominate. Expected finding: no sanitizer is free; all hurt benign quality measurably. State this as a negative result for defenses.

### [FMAI] P6: Adaptive sender vs detector

Current detector AUC tests one fixed sender against one fixed detector. FMAI cares about robustness under adaptation.

- Train latent-statistics logistic regression detector on existing protocol
- Implement gradient-aware sender:
  - At each carrier position, compute detector gradient w.r.t. latent
  - Project ε·d onto null space of detector gradient direction
  - Renormalize to maintain ‖perturbation‖ = ε
- Metrics: detector AUC at ε=2.0 with adaptive vs non-adaptive sender; bit accuracy under both
- Scale: 100 images, single ε
- Output: one row added to detector table; one paragraph in defense section about arms race

### [FMAI] P7: Native high-resolution validation

Removes the upscaled-CIFAR limitation.

- Dataset: 100 native 512×512 images from LAION-Aesthetics or COCO
- Run full PatchSteg pipeline at ε ∈ {2.0, 5.0}
- Report bit accuracy + PSNR + LPIPS + SSIM
- Output: one row in main results table; mention in limitations that 512×512 is now covered

### [FMAI] P8 (optional, only if P1-P7 are done): Composite Jacobian analysis

Strongest mechanistic claim in the paper if it works. Cuttable.

- For 30 carrier positions across 10 images:
  - Compute local Jacobian J = ∂(Enc∘Dec)/∂z at z̄ via finite differences (4 channels, small δ)
  - SVD on J
  - Alignment score = ⟨d, top singular vector⟩ × top singular value
- Correlate alignment score with empirical bit-recovery margin
- Output: scatter plot, Pearson correlation, brief mechanistic claim if r > 0.5

---

## What we're NOT building

- Full adaptive attack against a learning monitor (out of scope, mention as future work)
- More VAE backbones beyond SD-VAE-MSE, SD-VAE-EMA, SDXL-VAE
- More datasets beyond CIFAR-10, native 512×512 (LAION/COCO)
- CDF-PatchSteg extension experiments (move fully to appendix)
- More structural-image synthetics (move to appendix)

---

## Reframing tasks by workshop

### AIWILD (Agents in the Wild) — agent security framing

**Body reuse:** ~95% from current draft after P1-P3 are done.

**Intro rewrite (1 page):**
- Lead with agentic threat scenario, not interpretability
- Open: multimodal agents are increasingly deployed in environments where they share images with each other and with monitoring infrastructure (cite OpenAI Operator, OpenClaw, AIWILD about-page language)
- If the same VAE backbone is used by sender, receiver, and intermediate processing, an information channel exists in the latent that pixel-domain monitoring cannot see
- Introduce PatchSteg as the probe demonstrating this channel exists, is reliable, and is hard to detect with naive monitors
- Treat interpretability framing as one consequence, not the spine

**New threat model section (~3/4 page):**
- Sender capabilities: can perturb latent before decode
- Receiver capabilities: can re-encode received image
- Monitor capabilities: sees pixels, may have access to a detector model
- Shared knowledge: sender and receiver share VAE checkpoint and seed
- In scope: covert info passing through a deployed multimodal pipeline
- Out of scope: full adaptive attacks against a learning monitor

**Move Section 7 (Safety and Monitoring) earlier and rename:**
- New section title: "Implications for Multi-Agent Image-Mediated Communication"
- Every claim should reference the threat model

**Expand discussion (~1/2 page):**
- Detection-evasion arms race
- Platform-level defenses (mandatory latent sanitization on shared image infrastructure)
- Implications for monitoring agent interactions in deployment

**Trim:**
- CDF-PatchSteg → appendix
- Synthetic-image content-dependence figure → appendix or compress to one figure

**Cite heavily:** Motwani et al. (secret collusion) as load-bearing, not side citation.

**Page limit:** 9 pages regular track. Use AIWILD's modified ICML template.

---

### AI4GOOD (Cooperative AI track) — covert collusion as cooperative-AI failure

**Body reuse:** ~85% from AIWILD branch after AIWILD is shipped.

**Intro rewrite (1 page):**
- Cooperative AI is about trust, commitment, and verifiability between AI systems
- PatchSteg is the failure case: two AI systems can coordinate covertly through shared media in a way that breaks the verifiability assumption
- Open with vignette: multi-agent system where verifiability is supposed to be enforced through transparent communication, but participants can coordinate hidden side payments or strategy through pixel content that looks innocuous to a monitor
- Cite Motwani et al. heavily; CAIF cares about exactly this kind of coordination failure

**New "scenarios where this matters" section (~1/2 page):**
- AI-mediated content moderation systems passing images through shared VAE-based preprocessing
- Cooperative multi-agent platforms where agents share generated visualizations as part of their coordination
- AI-assisted civic discourse tools where covert content embedded in shared images undermines legitimacy of public deliberation
- Frame each as a "potentially affected setting," not as an attack we executed

**Discussion swap:**
- AIWILD's discussion is about agent monitoring
- AI4GOOD's is about cooperative-AI verifiability and what defenses look like at platform/governance level
- One paragraph on policy-relevant implications (don't go full TAIGR, just signal the concern)

**Optional inclusion:** simplest version of agent-in-loop demo (P4) as one figure + one table if available by submission time. Workshop is interdisciplinary, so concrete scenarios carry more weight here than dense experimental tables.

**Submission detail:** mark "Cooperative AI" track on OpenReview submission. This is what makes the paper eligible for the CAIF prize and travel funding.

**Page limit:** 2-8 pages.

---

### FMAI (Failure Modes in Agentic AI) — strongest version, full restructure

**Body reuse:** ~70% from AIWILD branch, plus all new experiments P4-P7.

**Restructure body around FMAI's four research outputs:**

- §3 Operational Definition
  - What is the failure: covert latent channel surviving decode-reencode round trip
  - Boundaries: requires shared VAE checkpoint; baseline protocol requires shared seed
  - Falsifiable mechanistic hypothesis: stable directions correspond to high-singular-value directions of composite Enc∘Dec Jacobian aligned with d

- §4 Reproducible Trigger
  - PatchSteg pipeline (current Section 3 content)
  - Code release commitment with anonymized URL

- §5 Diagnostics
  - Stability maps (existing)
  - Content correlation (existing)
  - Detector AUC + operating-point precision/recall (P2)
  - Agent-in-loop measurements (P4)
  - Native-res validation (P7)

- §6 Verified Fixes
  - Sanitizer Pareto curves (P5)
  - Adaptive sender results (P6)
  - Honest framing of what works and what doesn't

**Add explicit negative results framing:**
- Sanitizer Pareto: no defense gives free disruption — frame as negative result for defenses
- Adaptive sender: if it breaks the detector, frame as negative result for detection
- FMAI explicitly welcomes well-documented negative results

**Limitations update:**
- Composite Jacobian analysis: partial (if P8 done) or future work (if not)
- Agent demo uses two specific VLMs; may not generalize to all multimodal architectures
- Sanitizers tested on 100 images per setting (smallest evaluation in paper)

**Page limit:** 8 pages excluding references and appendix.

---

## Build order summary

1. P1, P2, P3 (shared metric/detector/robustness fixes) → unblocks AIWILD
2. AIWILD intro + threat model + section restructure → AIWILD ships
3. AI4GOOD intro + scenarios section → AI4GOOD ships
4. P4 (agent-in-loop demo) → highest-leverage FMAI addition
5. P5 (sanitizer Pareto) → FMAI "verified fixes"
6. P6 (adaptive sender) → FMAI rigor
7. P7 (native high-res) → removes stated limitation
8. FMAI restructure around four-output framework → FMAI ships
9. P8 (Jacobian) only if everything else is done

## Cuts if running behind

- First cut: P8 (Jacobian)
- Second cut: P7 (native high-res) — keep limitation as-is
- Third cut: P6 (adaptive sender) — substitute with stronger P5 sweep
- Do not cut: P1, P4, P5