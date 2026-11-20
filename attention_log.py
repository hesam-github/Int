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
            return
        w = np.asarray(att.predict(self.val_dataset, verbose=0), dtype=np.float64)  # (N, K)
        k = w.shape[1]
        names = list(getattr(self.intent, "branch_names", [f"b{i}" for i in range(k)]))
        mean = w.mean(0)
        entropy = float((-(w * np.log(w + 1e-12)).sum(1) / np.log(k)).mean())  # 1 = uniform, 0 = one-hot
        top = np.bincount(w.argmax(1), minlength=k) / len(w)
        rec = {"epoch": epoch + 1, "names": names, "mean": mean.tolist(),
               "std": w.std(0, ddof=1).tolist(), "argmax_share": top.tolist(), "entropy": entropy}
        self.records.append(rec)
        print(f"epoch {epoch + 1:>3} [attn] " + " ".join(f"{n}={m:.3f}" for n, m in zip(names, mean))
              + f" | entropy={entropy:.3f} | top1 share: "
              + " ".join(f"{t:.2f}" for t in top))
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
    print(f"{'epoch':>5} " + " ".join(f"{n[:9]:>9}" for n in names)
          + f" {'entropy':>7} {'train_loss':>9} {'val_loss':>9} {'val_auc':>9}")
    for r in att:
        m = em.get(r["epoch"], {})
        print(f"{r['epoch']:>5} " + " ".join(f"{w:>9.3f}" for w in r["mean"]) + f" {r['entropy']:>7.3f}"
              + _f(m.get("train_loss")) + _f(m.get("val_loss")) + _f(m.get("val_roc_auc")))