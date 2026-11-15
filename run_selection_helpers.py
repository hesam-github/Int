"""
run_selection_helpers.py

Purpose: shared interactive-selection helpers for the standalone
evaluation/diagnostic scripts that browse ALREADY-TRAINED checkpoints
under experiments/<framework>/<architecture>/<run_name>/ -- fine-tuning,
metric plotting, gradient-extremes checking, and model comparison. All of
these pick from folders that must already EXIST.

NOT the same as train_helpers.interactive_pick_architecture(), which
train_pie_intent_tensorflow_gpu.py uses to choose (and optionally CREATE)
the architecture folder for a brand-new training run -- that one accepts
a freely-typed name that doesn't have to exist yet. The two were
previously copy-pasted, byte-for-byte identical except for their
docstring, into finetune_from_baseline_gpu_lr1e4.py, plot_run_metrics.py,
and compare_model_architectures.py; consolidated here so all three stay
in sync instead of silently drifting apart.

parse_indices() was likewise duplicated, byte-for-byte identical, in
check_gradient_extremes.py and evaluate_and_compare.py.
"""

import os
import glob
import time


def parse_indices(raw, max_index):
    """Parses '0 1 2' or '0,1,2' into a validated, de-duplicated list of
    ints in [0, max_index], preserving the order the user typed them in."""
    raw = raw.replace(',', ' ')
    parts = [p for p in raw.split() if p.strip() != '']
    if not parts:
        raise ValueError("No indices entered.")

    indices = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"'{p}' is not a valid number.")
        idx = int(p)
        if not (0 <= idx <= max_index):
            raise ValueError(f"{idx} is out of range (0-{max_index}).")
        indices.append(idx)

    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    return unique_indices


def interactive_pick_architecture(experiments_root, framework, default_architecture):
    """
    Lists existing architecture folders under experiments_root/framework
    with a numeric index next to each one. Enter keeps the default (must
    already exist in the list -- unlike
    train_helpers.interactive_pick_architecture(), this variant always
    operates on an EXISTING architecture and never creates a new one).
    """
    framework_dir = os.path.join(experiments_root, framework)
    existing = []
    if os.path.isdir(framework_dir):
        existing = sorted(
            d for d in os.listdir(framework_dir)
            if os.path.isdir(os.path.join(framework_dir, d))
        )

    print(f"\n[architecture] existing architecture folders under {framework_dir}:")
    if existing:
        for i, a in enumerate(existing):
            marker = "  <-- default" if a == default_architecture else ""
            print(f"  [{i}] {a}{marker}")
    else:
        print("  (none found)")

    while True:
        raw = input(
            f"[architecture] enter a number from the list above "
            f"[default: '{default_architecture}']: "
        ).strip()

        if raw == "":
            if default_architecture in existing:
                return default_architecture
            print(f"  ! default '{default_architecture}' not found under {framework_dir}, "
                  f"pick a number instead.")
            continue

        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(existing):
                return existing[idx]
            print(f"  ! out of range, pick 0-{len(existing) - 1}.")
            continue

        print("  ! please enter a number from the list.")


def interactive_pick_checkpoint(architecture_dir, name_hint=None, purpose='evaluate'):
    """
    Lists every run folder under architecture_dir that contains BOTH
    model.h5 and configs.pkl (i.e. a completed, loadable checkpoint), and
    lets the user pick one by index. Newest-created first.

    :param name_hint: optional substring (e.g. a previous default
        RUN_NAME prefix); matching folders are bubbled to the top and
        flagged, but EVERY other checkpoint is still selectable -- this
        avoids "auto-pick the newest matching prefix" silently choosing
        the wrong run when training was run more than once.
    :param purpose: verb phrase describing what the checkpoint will be
        used for, substituted into the prompt/error text (e.g.
        'fine-tune from', 'evaluate'). Does not affect selection logic --
        this is what previously made finetune_from_baseline_gpu_lr1e4.py's
        and compare_model_architectures.py's copies of this function look
        like two different functions when they were actually the same one.
    """
    if not os.path.isdir(architecture_dir):
        raise FileNotFoundError(
            f"Architecture directory does not exist: {architecture_dir}\n"
            f"Nothing to {purpose} -- train a baseline first."
        )

    all_dirs = [d for d in glob.glob(os.path.join(architecture_dir, '*'))
               if os.path.isdir(d)]
    loadable = [
        d for d in all_dirs
        if os.path.isfile(os.path.join(d, 'model.h5'))
        and os.path.isfile(os.path.join(d, 'configs.pkl'))
    ]

    if not loadable:
        raise FileNotFoundError(
            f"No completed checkpoints (model.h5 + configs.pkl) found under:\n"
            f"  {architecture_dir}\n"
            f"Run the training script first."
        )

    loadable.sort(key=os.path.getctime, reverse=True)
    if name_hint:
        loadable.sort(key=lambda d: name_hint not in os.path.basename(d))

    print(f"\n[checkpoint] available checkpoints under:\n  {architecture_dir}\n")
    for i, d in enumerate(loadable):
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getctime(d)))
        marker = "  <-- matches hint" if name_hint and name_hint in os.path.basename(d) else ""
        print(f"  [{i}] {os.path.basename(d)}   (created {ts}){marker}")

    while True:
        raw = input(
            f"\n[checkpoint] select a checkpoint to {purpose} "
            f"[0-{len(loadable) - 1}, default: 0]: "
        ).strip()
        if raw == "":
            choice = 0
        else:
            try:
                choice = int(raw)
            except ValueError:
                print("  ! please enter a number.")
                continue
        if 0 <= choice < len(loadable):
            return loadable[choice]
        print(f"  ! out of range, pick 0-{len(loadable) - 1}.")
