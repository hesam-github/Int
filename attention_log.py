"""attention_log.py -- per-epoch mean branch-attention weights on val. Reader: python attention_log.py <run_dir>"""
import json
import os
import sys

import numpy as np
import tensorflow as tf


class AttentionLogger(tf.keras.callbacks.Callback):
    def __init__(self, intent, val_dataset, save_path):
        super().__init__()
        self.intent, self.val_dataset, self.save_path = intent, val_dataset, save_path
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        att = getattr(self.intent, "attention_model", None)
        if att is None:
            if epoch == 0:
                print("[attn] this architecture has no attention_model; logger idle")
                print("[attn] top1 is printed only for softmax (static_pool, core_plus_extras).")
                print("[attn] gated_add prints mean/open only when gate_stack exists.")
            return
        w = np.asarray(att.predict(self.val_dataset, verbose=0), dtype=np.float64)
        if w.ndim == 1:
            w = w[:, None]
        k = int(w.shape[1])
        gate_names = list(getattr(self.intent, "gate_names", None) or [])
        branch_names = list(getattr(self.intent, "branch_names", None) or [])
        if len(gate_names) == k:
            names = gate_names
            kind = "gate"
        elif len(branch_names) == k:
            names = branch_names
            kind = "softmax"
        else:
            names = [f"b{i}" for i in range(k)]
            kind = "softmax"
        mean = w.mean(0)
        row_sum = float(w.sum(1).mean())
        if kind == "gate" or abs(row_sum - 1.0) > 0.15:
            open_share = (w > 0.5).mean(0)
            rec = {"epoch": epoch + 1,
                   "names": ["core_bbox_visual"] + list(names),
                   "mean": [1.0] + mean.tolist(),
                   "open_share": [1.0] + open_share.tolist(),
                   "kind": "gate"}
            self.records.append(rec)
            print(f"epoch {epoch + 1}  residual gates")
            parts = ["core_bbox_visual: weight=1.000"]
            parts += [f"{n}: mean={mean[i]:.3f} open={open_share[i]:.3f}"
                      for i, n in enumerate(names)]
            print("  " + " | ".join(parts))
            if epoch == 0:
                print("  mean = average learned sigmoid. That is the residual coefficient.")
                print("  open = fraction of val windows with gate > 0.5. open is not top1.")
                print("  core weight is fixed at 1. It is not a softmax weight and not a gate.")
        else:
            entropy = float((-(w * np.log(w + 1e-12)).sum(1) / np.log(max(k, 2))).mean())
            top = np.bincount(w.argmax(1), minlength=k) / max(len(w), 1)
            rec = {"epoch": epoch + 1, "names": names, "mean": mean.tolist(),
                   "std": w.std(0, ddof=1).tolist() if len(w) > 1 else [0.0] * k,
                   "argmax_share": top.tolist(), "entropy": entropy, "kind": "softmax"}
            self.records.append(rec)
            print(f"epoch {epoch + 1}  attention")
            parts = [f"{n}: mean={mean[i]:.3f} top1={top[i]:.3f}" for i, n in enumerate(names)]
            print("  " + " | ".join(parts) + f" | entropy={entropy:.3f} (1=uniform, 0=one branch)")
        tmp = self.save_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fid:
            json.dump(self.records, fid, indent=2)
        os.replace(tmp, self.save_path)


def _f(x):
    return f"{x:>9.4f}" if isinstance(x, (int, float)) else f"{'n/a':>9}"


if __name__ == "__main__":
    run = sys.argv[1]
    with open(os.path.join(run, "attention_log.json"), encoding="utf-8") as fid:
        att = json.load(fid)
    with open(os.path.join(run, "epoch_metrics.json"), encoding="utf-8") as fid:
        em = {r["epoch"]: r for r in json.load(fid)}
    names = att[0]["names"]
    print(f"{'epoch':>5} " + " ".join(f"{n[:12]:>12}" for n in names)
          + f" {'kind':>8} {'train_loss':>9} {'val_loss':>9} {'val_auc':>9}")
    for r in att:
        m = em.get(r["epoch"], {})
        kind = r.get("kind", "softmax")
        mark = f"{r['entropy']:8.3f}" if "entropy" in r else f"{'gate':>8}"
        print(f"{r['epoch']:>5} " + " ".join(f"{w:12.3f}" for w in r["mean"]) + f" {mark}"
              + _f(m.get("train_loss")) + _f(m.get("val_loss")) + _f(m.get("val_roc_auc"))
              + f"  {kind}")