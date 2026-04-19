#!/usr/bin/env python3
"""Compare last log columns (val metrics) vs cumulative train events for res vs 18M marathon."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RES_LOG = Path("/home/levos/experiments/2026-04-01_gat_baselines_res_study/train_res.log")
M18_LOG = Path("/home/levos/experiments/2026-04-05_final_marathon_18M/train_18M.log")
OUT_PNG = Path("/home/levos/experiments/2026-04-05_final_marathon_18M") / "val_metrics_vs_seen_events.png"

# Must match train_res.py / train_18M.py
RES_BATCH, RES_VE = 512, 200
M18_BATCH, M18_VE = 256, 200

RES_EVENTS_PER_VE = RES_BATCH * RES_VE
M18_EVENTS_PER_VE = M18_BATCH * M18_VE

VAL_NAMES = ("V_Loss", "V_Prec", "V_Rec", "V_P@R0.9")


def parse_rows(path: Path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Step"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 12:
                continue
            try:
                ve = int(parts[0])
                v_loss = float(parts[8])
                v_prec = float(parts[9])
                v_rec = float(parts[10])
                v_p9 = float(parts[11])
            except ValueError:
                continue
            rows.append((ve, v_loss, v_prec, v_rec, v_p9))
    return rows


def cumulative_events(rows, events_per_ve):
    # Virtual epoch index in log = number of completed VE windows; cumulative events after that point
    return [r[0] * events_per_ve for r in rows]


def main():
    res_rows = parse_rows(RES_LOG)
    m18_rows = parse_rows(M18_LOG)
    if not res_rows or not m18_rows:
        raise SystemExit("Need non-empty parsed rows from both logs.")

    x_res = cumulative_events(res_rows, RES_EVENTS_PER_VE)
    x_m18 = cumulative_events(m18_rows, M18_EVENTS_PER_VE)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    fig.suptitle(
        "Валидация (последние 4 колонки лога) vs накопленные train-события\n"
        f"res: {RES_EVENTS_PER_VE:,} событий/вирт. эп.; 18M: {M18_EVENTS_PER_VE:,} событий/вирт. эп.",
        fontsize=11,
    )

    for ax, j, name in zip(axes.flat, range(4), VAL_NAMES):
        y_res = [r[j + 1] for r in res_rows]
        y_m18 = [r[j + 1] for r in m18_rows]
        ax.plot(
            [x / 1e6 for x in x_res],
            y_res,
            label=f"train_res ({len(res_rows)} VE)",
            color="C0",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.plot(
            [x / 1e6 for x in x_m18],
            y_m18,
            label=f"train_18M ({len(m18_rows)} VE)",
            color="C1",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    for ax in axes[1, :]:
        ax.set_xlabel("Накопленные train-события (млн)")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
