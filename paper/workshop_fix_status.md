# Workshop Fix Status

Date: 2026-04-29

Source task spec: `fixes.md`

This tracks what is implemented in the three workshop folders and what remains experimental work. The current state is not "all fixes complete"; it is a venue-specific paper split with appendices and explicit placeholders for the highest-priority new experiments.

## Folder Setup

| Item | Status | Notes |
|---|---|---|
| `paper/fmai/` | Done | FMAI failure-mode framing with operational definition, reproducible trigger, diagnostics, verified-fixes framing, and appendix. |
| `paper/aiwild/` | Done | Agent-security framing with threat model, monitoring assumptions, and appendix. |
| `paper/ai4good/` | Done | Cooperative-AI/verifiability framing with scenarios and appendix. |
| Versioned PDFs | Done | Build scripts now copy `main.pdf` to `fmai_v1.pdf`, `aiwild_v1.pdf`, and `ai4good_v1.pdf`. |
| Separate git branches `aiwild`, `ai4good`, `fmai` | Not done | User requested folders; current implementation keeps all three versions on branch `icml`. |
| Shared `core/` paper body sections | Partially done | Reused manually across folders, not factored into TeX includes yet. |

## Shared Experimental Tasks

| Task | Status | Current handling |
|---|---|---|
| P1: Add LPIPS and SSIM | Not done | All drafts explicitly retain PSNR-only limitation and call LPIPS/SSIM a required next step. |
| P2: Detector precision/recall at fixed FPR | Not done | Drafts still report AUC and note operating-point metrics as future work. |
| P3: Robustness refresh with realistic distortions | Not done | Existing robustness figure remains in appendix; main text notes current distortion suite limitations. |

## FMAI-Specific Experimental Tasks

| Task | Status | Current handling |
|---|---|---|
| P4: Agent-in-loop demo | Not done | FMAI includes an "Agent-In-The-Loop Evaluation Plan" section. AIWILD/AI4GOOD mention this as the most important missing experiment. |
| P5: Sanitizer Pareto curves | Not done | FMAI has a verified-fixes section using existing sanitizer pilots and explicitly says a full Pareto is required before final submission. |
| P6: Adaptive sender vs detector | Not done | Mentioned as future/needed work in limitations and discussion. |
| P7: Native high-resolution validation | Not done | All drafts identify upscaled CIFAR-10 as a limitation and native 512x512 validation as needed. |
| P8: Composite Jacobian analysis | Not done | FMAI states the falsifiable hypothesis and marks composite-Jacobian analysis as future work. |

## Reframing Tasks

| Workshop | Status | Notes |
|---|---|---|
| AIWILD intro rewrite | Done | Leads with multimodal agent threat scenario. |
| AIWILD threat model | Done | Sender, receiver, monitor, and out-of-scope assumptions included. |
| AIWILD safety/monitoring section moved earlier | Done | "Implications For Multi-Agent Systems" appears before limitations. |
| AIWILD CDF trimmed to appendix | Done | CDF is appendix-only exploratory material. |
| AI4GOOD intro rewrite | Done | Leads with Cooperative AI trust, commitment, and verifiability. |
| AI4GOOD scenarios section | Done | Includes cooperative platforms, moderation/deliberation, and information-integrity workflows. |
| AI4GOOD discussion swap | Done | Discussion focuses on verifiability and platform/governance implications. |
| FMAI four-output restructure | Done | Main body uses operational definition, reproducible trigger, diagnostics, and verified fixes. |
| FMAI negative-results framing | Done | Sanitization is framed as a non-free fidelity-disruption tradeoff. |

## Next Work Before Submission

Highest-priority remaining work:

1. Run P1 LPIPS/SSIM metric refresh.
2. Run P4 agent-in-loop demo.
3. Run P5 sanitizer Pareto curves.
4. Add P2 detector operating-point metrics.
5. Add P7 native high-resolution validation if time permits.

