"""
log_mode.py

Two verbosity modes, chosen once per run via interactive_pick_log_mode()
in train_helpers.py:

  'full'  -- unchanged: every print statement exactly as it exists today.
             This mode is a strict superset of 'clean' and preserves the
             current behavior exactly, so nothing breaks if you pick it.

  'clean' -- only the phase() explanations (below) plus the genuinely
             meaningful numbers (per-epoch metrics, split class counts,
             final summaries, warnings). Raw dict dumps of dead/unused
             config keys, duplicate config prints, bare function-object
             reprs, and Keras' own model.summary()/predict progress bars
             are suppressed.

phase() ALWAYS prints, in BOTH modes -- it is not verbosity, it is a
fixed short narration of what stage the pipeline is entering and why.
"""

_STATE = {'mode': 'full'}

def configure_tf_logging():
    """
    Call once, right after `import tensorflow as tf`. In 'clean' mode,
    silences TensorFlow's own WARNING-level logs (e.g. "Layer X will not
    use cuDNN kernels...") -- these are internal kernel-selection notes,
    not signals about correctness of your run. In 'full' mode, TF's
    default logging is left untouched.
    """
    import logging
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.setLevel(logging.WARNING if is_full() else logging.ERROR)
    

def set_log_mode(mode):
    assert mode in ('clean', 'full'), f"unknown log mode: {mode}"
    _STATE['mode'] = mode


def get_log_mode():
    return _STATE['mode']


def is_full():
    return _STATE['mode'] == 'full'


def phase(title, explanation):
    """
    Prints a short section header + a 1-2 line plain-language explanation
    of the PURPOSE of the step about to run (the "why", not "which
    function is being called"). Always visible, in both log modes.
    """
    print(f"\n>>> {title}")
    print(f"    {explanation}")


def vprint(*args, **kwargs):
    """
    Prints only when log mode is 'full'. Use this for anything that is
    NOT a phase() header and NOT a genuinely useful per-epoch/summary
    number -- raw config dict dumps, duplicate prints, bare object
    reprs, and other diagnostic noise meaningful only when debugging the
    pipeline itself.
    """
    if is_full():
        print(*args, **kwargs)