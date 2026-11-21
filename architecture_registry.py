# architecture_registry.py
"""Interactive architecture registry. PIE is one option among others."""

ARCHITECTURES = {
    'pie_convlstm': {
        'desc': 'PIE paper: ConvLSTM on VGG + single LSTM on bbox/decoder',
        'fusion': 'pie',
        'feature_target': 'decoder',
    },
    'bbox_only': {
        'desc': 'BBox-only baseline (zero visual features, LSTM on bbox)',
        'fusion': 'pie',
        'feature_target': 'decoder',
        'zero_visual': True,
    },
    'pcpa': {
        'desc': 'PCPA-style: visual branch + bbox LSTM + shared extras + attention',
        'fusion': 'pcpa',
        'feature_target': 'decoder',
    },
    'branchattn': {
        'desc': 'Branch attention. Sub-menu: static_pool, temporal_concat, or core_plus_extras',
        'fusion': 'branchattn',
        'feature_target': 'decoder',
    },
    'branchattn_temporal': {
        'desc': 'Same sub-menu as branchattn. static_pool: temporal attention on every span including bbox. core_plus_extras: temporal attention only on extra spans, not on the core. temporal_concat: no temporal attention.',
        'fusion': 'branchattn_temporal',
        'feature_target': 'decoder',
    },
    'encoder_extras': {
        'desc': 'Extras tiled into ConvLSTM channels (feature_target=encoder)',
        'fusion': 'pie',
        'feature_target': 'encoder',
    },
    'both': {
        'desc': 'Extras in both ConvLSTM channels and decoder LSTM',
        'fusion': 'pie',
        'feature_target': 'both',
    },
    'temporal_encoders': {
        'desc': 'PIE decoder + selectable temporal visual encoder (ConvLSTM2D / Transformer / S4 / S4D / Mamba)',
        'fusion': 'pie',
        'feature_target': 'decoder',
        'use_temporal_menu': True,
    },
    'decoder_only': {
        'desc': 'use_visual=False. One LSTM on the decoder matrix. No ConvLSTM, no visual_norm, no capacity.',
        'fusion': 'pie',
        'feature_target': 'decoder',
    },
}


def interactive_pick_architecture(default='pie_convlstm'):
    """Print menu and return chosen architecture name."""
    names = list(ARCHITECTURES.keys())
    print('\n--- Architecture ---')
    for i, name in enumerate(names, 1):
        meta = ARCHITECTURES[name]
        mark = ' *' if name == default else ''
        print(f'  {i}. {name}{mark} — {meta["desc"]}')
    raw = input(f'Select [1-{len(names)}] (default {default}): ').strip()
    if not raw:
        return default
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(names):
            return names[idx]
    except ValueError:
        pass
    print('Invalid choice, using default')
    return default


def interactive_pick_branch_mode(default='static_pool'):
    """Sub-menu for branchattn and branchattn_temporal. Does not change other architectures."""
    modes = ['static_pool', 'temporal_concat', 'core_plus_extras']
    print('\n--- Branch fusion ---')
    print('  1. static_pool *     current: Dense(visual) + one LSTM per column + BranchAttnPool')
    print('  2. temporal_concat   PIE graph: RepeatVector(Flatten) + whole decoder + one LSTM')
    print('  3. core_plus_extras  PIE core on bbox only; each extra column gets its own LSTM')
    raw = input(f'Select [1-3] (default {default}): ').strip()
    if not raw:
        return default
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(modes):
            return modes[idx]
    except ValueError:
        pass
    print('Invalid choice, using static_pool')
    return 'static_pool'
    
def interactive_pick_aux_loss(default=0.0):
    """Total weight of the per-branch auxiliary losses. 0 = off (single loss, as before)."""
    print('\n--- Auxiliary per-branch loss (any multi-branch architecture) ---')
    print('  0   off (only the loss on the fused output)  <-- default')
    print('  >0  every branch gets its own sigmoid head; the TOTAL aux weight is split equally')
    print('      between the heads (main loss weight stays 1.0). Try 0.25 / 0.5 / 1.0.')
    raw = input(f'aux_loss_weight (default {default}): ').strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
        if value >= 0:
            return value
    except ValueError:
        pass
    print('Invalid value, aux loss off')
    return 0.0

def interactive_pick_fusion_mode(has_offline_features, default='pie_concat'):
    """Dead menu. Do not call.
    branchattn uses branch_mode. Option 8 uses temporal_fusion.
    This question used to be stored as opts['fusion_mode'] and never reached the graph.
    """
    raise RuntimeError(
        "interactive_pick_fusion_mode is not wired. "
        "Use branch_mode or temporal_fusion.")
    if not has_offline_features:
        return 'pie_concat'
    print('\n--- Fusion mode (when offline features are present) ---')
    print('  1. pie_concat *     RepeatVector + Concatenate all decoder features + one LSTM')
    print('  2. core_plus_extras Visual+bbox core LSTM + separate LSTM per extra feature + BranchAttnPool')
    raw = input(f'Select [1-2] (default {default}): ').strip()
    if not raw:
        return default
    if raw == '1':
        return 'pie_concat'
    if raw == '2':
        return 'core_plus_extras'
    print('Invalid choice, using pie_concat')
    return 'pie_concat'

def get_architecture_config(name):
    """Return fusion / feature_target / flags for the chosen architecture."""
    if name not in ARCHITECTURES:
        raise ValueError(f'Unknown architecture: {name}')
    cfg = ARCHITECTURES[name].copy()
    cfg['name'] = name
    return cfg