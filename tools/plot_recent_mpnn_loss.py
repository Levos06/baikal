#!/usr/bin/env python3
"""
Plot train/val loss and primary P@R columns from virtual-epoch logs:
faint raw + bright EMA-smoothed (α=0.06).

PR metric column indices match project logs: T_* at 7, V_* at 11.
English comments only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tools" / "plots"

# (label, path to log) — recent MPNN experiments, chronological
RUNS: list[tuple[str, Path]] = [
    ("General MPNN k=4", ROOT / "2026-05-05_general_mpnn_k4" / "train_mpnn_k4.log"),
    ("Master vertex k=4", ROOT / "2026-05-09_mpnn_master_vertex_k4" / "train_mpnn_master_k4.log"),
    ("Edge geom k=4", ROOT / "2026-05-10_mpnn_edge_geom_k4" / "train_mpnn_edge_geom_k4.log"),
    ("Master + 12D edges, FC+master", ROOT / "2026-05-11_mpnn_master_vertex_edge12_fc_master_k4" / "train_mpnn_master_edge12_k4.log"),
]


def _header_metric_names(header_line: str) -> tuple[str, str]:
    """Return short names for train/val PR columns, e.g. P@R0.95."""
    parts = [p.strip() for p in header_line.split("|")]
    train_h = parts[7] if len(parts) > 7 else "T_metric"
    val_h = parts[11] if len(parts) > 11 else "V_metric"
    t_short = train_h[2:] if train_h.startswith("T_") else train_h
    v_short = val_h[2:] if val_h.startswith("V_") else val_h
    return t_short, v_short


def parse_log(log_path: Path) -> dict | None:
    if not log_path.is_file():
        return None
    lines = log_path.read_text().splitlines()
    if len(lines) < 2:
        return None
    t_short, v_short = _header_metric_names(lines[0])
    if t_short != v_short:
        # Normalized logs should match; use train name for y-label
        tag = f"{t_short} (train) / {v_short} (val)"
    else:
        tag = t_short

    epochs, t_loss, v_loss, t_pr, v_pr = [], [], [], [], []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 12:
            continue
        try:
            epochs.append(int(parts[0]))
            t_loss.append(float(parts[4]))
            v_loss.append(float(parts[8]))
            t_pr.append(float(parts[7]))
            v_pr.append(float(parts[11]))
        except ValueError:
            continue
    if not epochs:
        return None
    return {
        "epoch": np.asarray(epochs, dtype=np.float64),
        "t_loss": np.asarray(t_loss, dtype=np.float64),
        "v_loss": np.asarray(v_loss, dtype=np.float64),
        "t_pr": np.asarray(t_pr, dtype=np.float64),
        "v_pr": np.asarray(v_pr, dtype=np.float64),
        "pr_tag": t_short,
        "pr_label": tag,
    }


def ema(y: np.ndarray, alpha: float = 0.06) -> np.ndarray:
    if y.size == 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parsed: list[tuple[str, dict]] = []
    for name, lp in RUNS:
        data = parse_log(lp)
        if data is None:
            print(f"skip missing or empty: {lp}")
            continue
        parsed.append((name, data))

    if not parsed:
        raise SystemExit("No logs parsed")

    train_color = "#2563eb"
    val_color = "#ea580c"
    n = len(parsed)
    fig_h = max(2.8 * n, 4.0)

    # --- Loss panels
    fig, axes = plt.subplots(n, 1, figsize=(10.0, fig_h), sharex=True, sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (name, d) in zip(axes, parsed):
        ep, tl, vl = d["epoch"], d["t_loss"], d["v_loss"]
        tl_s, vl_s = ema(tl), ema(vl)
        ax.plot(ep, tl, color=train_color, alpha=0.28, linewidth=1.0, label="Train (raw)")
        ax.plot(ep, tl_s, color=train_color, alpha=1.0, linewidth=2.0, label="Train (smooth)")
        ax.plot(ep, vl, color=val_color, alpha=0.28, linewidth=1.0, label="Val (raw)")
        ax.plot(ep, vl_s, color=val_color, alpha=1.0, linewidth=2.0, label="Val (smooth)")
        ax.set_ylabel("Loss")
        ax.set_title(name, fontsize=11, loc="left")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("Virtual epoch")
    fig.suptitle("Cross-entropy loss (raw vs EMA α=0.06)", fontsize=12, y=1.002)
    fig.tight_layout()
    out_loss = OUT_DIR / "recent_mpnn_loss_panels.png"
    fig.savefig(out_loss, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_loss)

    # --- P@R panels (key metric)
    figp, axesp = plt.subplots(n, 1, figsize=(10.0, fig_h), sharex=True, sharey=False)
    if n == 1:
        axesp = [axesp]
    for ax, (name, d) in zip(axesp, parsed):
        ep, tm, vm = d["epoch"], d["t_pr"], d["v_pr"]
        tm_s, vm_s = ema(tm), ema(vm)
        ax.plot(ep, tm, color=train_color, alpha=0.28, linewidth=1.0, label="Train (raw)")
        ax.plot(ep, tm_s, color=train_color, alpha=1.0, linewidth=2.0, label="Train (smooth)")
        ax.plot(ep, vm, color=val_color, alpha=0.28, linewidth=1.0, label="Val (raw)")
        ax.plot(ep, vm_s, color=val_color, alpha=1.0, linewidth=2.0, label="Val (smooth)")
        ax.set_ylabel(d["pr_label"])
        ax.set_title(f"{name} — {d['pr_tag']}", fontsize=11, loc="left")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    axesp[-1].set_xlabel("Virtual epoch")
    figp.suptitle("Key metric from logs (raw vs EMA α=0.06)", fontsize=12, y=1.002)
    figp.tight_layout()
    out_pr = OUT_DIR / "recent_mpnn_pr_metric_panels.png"
    figp.savefig(out_pr, dpi=160, bbox_inches="tight")
    plt.close(figp)
    print("Wrote", out_pr)

    # --- Smoothed val loss compare
    fig2, ax2 = plt.subplots(figsize=(10.0, 5.0))
    cmap = plt.colormaps["tab10"]
    for i, (name, d) in enumerate(parsed):
        ep, vl = d["epoch"], d["v_loss"]
        ax2.plot(ep, ema(vl), color=cmap(i % 10), linewidth=2.2, label=name)
    ax2.set_xlabel("Virtual epoch")
    ax2.set_ylabel("Val loss (smoothed)")
    ax2.set_title("Validation loss (EMA-smoothed): comparison")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9)
    fig2.tight_layout()
    out_vl = OUT_DIR / "recent_mpnn_val_loss_smoothed_compare.png"
    fig2.savefig(out_vl, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print("Wrote", out_vl)

    # --- Smoothed val P@R: only same threshold (fair compare)
    p95 = [(name, d) for name, d in parsed if d["pr_tag"] == "P@R0.95"]
    if len(p95) >= 2:
        fig3, ax3 = plt.subplots(figsize=(10.0, 5.0))
        for i, (name, d) in enumerate(p95):
            ep, vm = d["epoch"], d["v_pr"]
            ax3.plot(ep, ema(vm), color=cmap(i % 10), linewidth=2.2, label=name)
        ax3.set_xlabel("Virtual epoch")
        ax3.set_ylabel("Val P@R0.95 (smoothed)")
        ax3.set_title("Validation P@R0.95 (EMA-smoothed): experiments with the same metric")
        ax3.grid(True, alpha=0.25)
        ax3.legend(loc="lower right", fontsize=9)
        fig3.tight_layout()
        out_vp95 = OUT_DIR / "recent_mpnn_val_p_at_r095_smoothed_compare.png"
        fig3.savefig(out_vp95, dpi=160, bbox_inches="tight")
        plt.close(fig3)
        print("Wrote", out_vp95)

    p09 = [(name, d) for name, d in parsed if d["pr_tag"] == "P@R0.9"]
    if len(p09) == 1:
        name, d = p09[0]
        fig4, ax4 = plt.subplots(figsize=(10.0, 4.0))
        ep, vm = d["epoch"], d["v_pr"]
        ax4.plot(ep, ema(vm), color=cmap(0), linewidth=2.2, label=f"{name} (Val P@R0.9)")
        ax4.set_xlabel("Virtual epoch")
        ax4.set_ylabel("Val P@R0.9 (smoothed)")
        ax4.set_title("General MPNN k=4 — validation P@R0.9 (other runs use P@R0.95 in logs)")
        ax4.grid(True, alpha=0.25)
        ax4.legend(loc="lower right", fontsize=9)
        fig4.tight_layout()
        out_v09 = OUT_DIR / "recent_mpnn_val_p_at_r09_smoothed_general.png"
        fig4.savefig(out_v09, dpi=160, bbox_inches="tight")
        plt.close(fig4)
        print("Wrote", out_v09)


if __name__ == "__main__":
    main()
