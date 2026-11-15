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
        'desc': 'Same sub-menu as branchattn. Temporal attention is only inside extra branches of core_plus_extras',
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
    
def get_architecture_config(name):
    """Return fusion / feature_target / flags for the chosen architecture."""
    if name not in ARCHITECTURES:
        raise ValueError(f'Unknown architecture: {name}')
    cfg = ARCHITECTURES[name].copy()
    cfg['name'] = name
    return cfg