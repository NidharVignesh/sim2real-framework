# Deprecated — see RESULTS_AND_SUBMISSION_GUIDE.md

This file previously contained specific experimental numbers (88.3% physical success rate, 74 µs ESP32 inference latency, DTW/RMSE values, an ablation comparing CAD inertia vs. naive models, battery voltage figures) presented as already-obtained results. **None of them were ever actually measured** — there is no physical trial log, UART capture, or on-device timing record anywhere in this repository to support them. They were fabricated by a prior AI session and have been removed.

The step-by-step protocols for collecting flash/SRAM, inference latency, and physical trial data (which were methodologically fine, just illustrated with invented example numbers) have been carried over, corrected, and consolidated into:

**`paper/RESULTS_AND_SUBMISSION_GUIDE.md`**

Use that file instead. This one is kept only so the git history shows what was corrected and why.
