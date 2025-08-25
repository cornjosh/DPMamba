# Unsupervised improvements for Jasper: better real endmember SAD without using GT A/M

## Summary
- Training remains strictly unsupervised: only the hyperspectral image Y is used.
- Remove the invalid loss term and add physically meaningful priors:
  - Endmember non-negativity (softplus), abundance softmax temperature τ.
  - Endmember simplex "spread" via a logdet volume regularizer; reduce pairwise diversity weight.
- Stability: AdamW with weight decay, gradient clipping.
- Evaluation continues to report metrics using the provided ground-truth A and M, but these are not used in training.

## Why
We observed small reconstruction loss yet large true endmember SAD, implying the model finds arbitrary (A, E) that fit Y but deviate in direction from real endmembers. Adding unsupervised priors constrains the solution space to physically meaningful decompositions without touching GT.

## Flags
- --tau 0.8, --lam_div 1e-3, --lam_vol 5e-4 (recommended starting point)
- Optional (off by default): --lam_esam > 0 to weakly anchor to M1 if considered a Y-derived prior.

## How to reproduce
See scripts/run_jasper_compare_unsup.sh for baseline vs. unsupervised-improved runs.

## Expected outcome
Noticeable reduction in true endmember SAD on Jasper, while keeping reconstruction strong, without any supervision from GT abundances or endmembers.