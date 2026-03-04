# Experiment: GCN Temporal Optimized (2026-02-28)

## Goal
Repeat the previous temporal GCN experiment with several optimizations and corrections.

## Changes
1.  **Labels:** Signal is now defined as `labels != 0` (previously `labels > 0`).
2.  **Dropout:** Reduced from 0.3 to 0.1.
3.  **Temporal Edges:** Optimized `_get_temporal_edges` using broadcasting to avoid Python loops.
4.  **Activation:** Switched from `ReLU` to `GELU`.

## Parameters
- `k` (neighbors): 4
- `hidden_channels`: 128
- `lr`: 5e-4
- `epochs`: 80
- `batch_size`: 512
