"""
Unified re-evaluation of all 13 General-MPNN experiments.
For each experiment: load every checkpoint (100..1000), run the module's own
evaluate() on a FIXED validation sample (same seed -> identical events for all
experiments), capture (labels, probs) and compute BOTH P@R0.9 and P@R0.95.

Output: CSV lines  key,epoch,n_params,p_at_r090,p_at_r095
"""
import importlib.util
import os
import sys
import random
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

BASE = "/home/levos/experiments"
SEED = 1234
TARGET_EVENTS = 12000

# (key, dir, train_file, model_class, ckpt_prefix)
CONFIGS = [
    ("base",            "2026-05-05_general_mpnn_k4",                       "train_general_mpnn.py",                    "MPNN_FCLast",                      "model_mpnn_k4"),
    ("cossim_b07",      "2026-05-07_general_mpnn_cos_sim_beta07_k4",        "train_mpnn_cos_sim.py",                    "MPNN_CosSim_FCLast",               "model_mpnn_cossim_k4"),
    ("cossim_top8",     "2026-05-08_general_mpnn_cos_sim_top8_k4",          "train_mpnn_cos_sim_top8.py",               "MPNN_CosSimTopK_FCLast",           "model_mpnn_cossim_top8_k4"),
    ("master_vertex",   "2026-05-09_mpnn_master_vertex_k4",                 "train_mpnn_master.py",                     "MPNN_MasterVertex_FCLast",         "model_mpnn_master_k4"),
    ("edge_geom",       "2026-05-10_mpnn_edge_geom_k4",                     "train_mpnn_edge_geom.py",                  "MPNN_FCLastEdge",                  "model_mpnn_edge_geom_k4"),
    ("cossim_b05",      "2026-05-11_general_mpnn_cos_sim_beta05_k4",        "train_mpnn_cos_sim_beta05.py",             "MPNN_CosSim_FCLast",               "model_mpnn_cossim_beta05_k4"),
    ("edge3_learn32",   "2026-05-11_mpnn_edge3_learnable32_k4",             "train_mpnn_edge3_learn32.py",              "MPNN_FCLastEdgeProductEmb",        "model_mpnn_edge3_learn32_k4"),
    ("edge3_learn8",    "2026-05-11_mpnn_edge3_learnable8_k4",              "train_mpnn_edge3_learn8.py",               "MPNN_FCLastEdgeProductEmb",        "model_mpnn_edge3_learn8_k4"),
    ("master_edge12",   "2026-05-11_mpnn_master_vertex_edge12_fc_master_k4","train_mpnn_master_edge12.py",              "MPNN_MasterVertexEdge_AllLayers",  "model_mpnn_master_edge12_k4"),
    ("master_attn",     "2026-05-13_mpnn_master_attn_token_k4",            "train_mpnn_master_attn_token.py",          "MPNN_MasterAttnToken_FCLast",      "model_mpnn_master_attn_token_k4"),
    ("attn_fc_master",  "2026-05-14_mpnn_master_attn_token_fc_master_k4",  "train_mpnn_master_attn_token_fc_master.py","MPNN_MasterAttnToken_FCPlusMaster","model_mpnn_master_attn_token_fc_master_k4"),
    ("edge_learn8_master","2026-06-23_mpnn_edge_learn8_master_k4",          "train_mpnn_edge_learn8_master_k4.py",      "MPNN_EdgeLearn8_Master",           "model_mpnn_edge_learn8_master_k4"),
    ("fc2_master",      "2026-06-23_mpnn_fc2_master_k4",                    "train_mpnn_fc2_master_k4.py",              "MPNN_FC2_Master",                  "model_mpnn_fc2_master_k4"),
]


def load_module(key, path):
    spec = importlib.util.spec_from_file_location("mod_" + key, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m  # torch_geometric MessagePassing inspects sys.modules[__module__]
    spec.loader.exec_module(m)
    return m


def build_model(m, cls, key):
    C = getattr(m, cls)
    if key == "cossim_b05":
        return C(21, 2, cos_threshold=m.COS_SIM_BETA).to(m.DEVICE)
    return C(21, 2).to(m.DEVICE)


for key, d, f, cls, prefix in CONFIGS:
    path = os.path.join(BASE, d, f)
    try:
        m = load_module(key, path)
    except Exception as e:
        print(f"# IMPORT FAIL {key}: {e}", flush=True)
        continue

    # capture (labels, probs) by monkeypatching calculate_metrics
    captured = {}
    def cap(labels, probs, _c=captured):
        _c["labels"] = np.asarray(labels)
        _c["probs"] = np.asarray(probs)
        return 0.0, 0.0, 0.0
    m.calculate_metrics = cap

    bs = m.BATCH_SIZE
    nb = max(1, (TARGET_EVENTS + bs - 1) // bs)

    nparams = None
    for ep in range(100, 1001, 100):
        ckpt = os.path.join(BASE, d, "checkpoints", f"{prefix}_{ep}.pt")
        if not os.path.exists(ckpt):
            print(f"# MISSING {key} ep{ep}", flush=True)
            continue
        try:
            model = build_model(m, cls, key)
            if nparams is None:
                nparams = sum(p.numel() for p in model.parameters())
            sd = torch.load(ckpt, map_location=m.DEVICE, weights_only=True)
            model.load_state_dict(sd)
            model.eval()

            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
            val_ds = m.MediumDataset("val", k=4)
            loader = m.DataLoader(val_ds, batch_size=bs, num_workers=0)
            m.evaluate(model, loader, num_batches=nb)

            lab, prob = captured["labels"], captured["probs"]
            p, r, _ = precision_recall_curve(lab, prob)
            p90 = float(np.interp(0.90, r[::-1], p[::-1]))
            p95 = float(np.interp(0.95, r[::-1], p[::-1]))
            print(f"{key},{ep},{nparams},{p90:.4f},{p95:.4f}", flush=True)

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"# EVAL FAIL {key} ep{ep}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
