# sweep_runner.py (v6) -- ablation on paper-style inputs only: decoder bbox + VGG16 context (ConvLSTM encoder).
# No pose / semantic segmentation / optical flow is selected, loaded or processed.
# Usage (run from the project folder, next to train_pipeline.py):
#   python sweep_runner.py --stage isolated --seeds 1,2,3 --dry-run
#   python sweep_runner.py --stage isolated --seeds 1,2,3
#   python sweep_runner.py --stage visual --seeds 1,2,3 --max-hours 72
#   python sweep_runner.py --aggregate
import argparse, builtins, hashlib, itertools, json, os, random, re, statistics, subprocess, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIE_PATH = os.environ.get('PIE_PATH', r'D:\PIE codes\PIE_dataset')
OUT_ROOT = HERE / 'sweep_runs'
DEFAULT_MIN_PER_RUN = 145.0          # measured on GTX 1650 Ti, baseline config, 160 epochs

HP = dict(
    framework='tensorflow', lr=1e-5, clipvalue=None, batch_size=64,
    hidden_units=128, reg=0.001, lstm_dropout=0.4, lstm_recurrent_dropout=0.2,
    convlstm_filters=64, convlstm_kernel=2,
    es_min_delta=1e-4, es_patience=10,
    plateau_patience=3, plateau_factor=0.5, plateau_min_lr=1e-7, plateau_min_delta=1e-3,
    n_epochs=400, seed=1, data_seed=42, split_seed=42, deterministic_ops=True,
    attach_pose=False, log_mode='clean', threshold_space='raw', decision_threshold=0.5,
)
DATA_OPTS = {
    'fstride': 1, 'sample_type': 'all', 'height_rng': [0, float('inf')], 'squarify_ratio': 0,
    'data_split_type': 'default', 'seq_type': 'intention',
    'min_track_size': 0, 'max_size_observe': 15, 'max_size_predict': 5, 'seq_overlap_rate': 0.5,
    'balance': True, 'crop_type': 'context', 'crop_mode': 'pad_resize',
    'encoder_input_type': [], 'decoder_input_type': ['bbox'], 'output_type': ['intention_binary'],
}

# ---- 1. Pinned menu answers, keyed by the name of the function that calls input() ----------------
PINNED = {
    'interactive_pick_log_mode': ['0'],                    # clean
    'interactive_pick_balance_mode': ['1'],                # undersample_base_article
    'interactive_pick_monitor_metric': ['2'],              # val_loss (min)
    'interactive_pick_loss_choice': ['0'],                 # binary_crossentropy
    'interactive_pick_weight_decay': [''],                 # -> 0.0
    'interactive_pick_lstm_recurrent_activation': ['2'],   # hard_sigmoid
    'interactive_pick_normalize_bbox': ['0'],              # raw pixels
    'plateau': ['y'],
    'tte': ['n'],
    'interactive_pick_eval_every': ['4'],                  # never
    'confirm_visual_pipeline': [''],                       # only asked for vnorm + log1p -> proceed
    'interactive_pick_architecture': ['1'],                # pie_convlstm
    'interactive_pick_offline_features': ['n'],            # no pose / scene / flow
    'interactive_pick_decoder_features': ['0'],            # bbox only
    'interactive_pick_extra_norm': ['0'],
}
ALWAYS = {'interactive_pick_feature_norms': ''}            # asked once per extra feature; none selected

def build_queues(spec):
    q = {k: list(v) for k, v in PINNED.items()}
    if spec['split'] == 'default':                         # menu [0]: fixed video-set split, no sub-prompts
        q['interactive_pick_data_split'] = ['0']
        q['interactive_pick_seeds'] = [str(spec['seed']), '', '42']        # train, shuffle, data
    else:                                                  # menu [1]: ratios train,test,val; regen=N
        q['interactive_pick_data_split'] = ['1', '', '']
        q['interactive_pick_seeds'] = [str(spec['seed']), '', '42', '42']  # train, shuffle, data, split
    q['interactive_pick_use_visual'] = [spec['use_visual']]
    q['interactive_pick_zero_decoder_bbox'] = [spec['bbox_mode']]
    q['interactive_pick_visual_norm'] = [spec['visual_norm']]
    q['visual_percentile'] = [spec['percentile']]
    e, i, h = cap_values(spec['capacity'])
    head = [spec['capacity']] if spec['capacity'] in PRESET_CAP else ['5', str(e), str(i), str(h)]
    q['interactive_pick_capacity'] = head + [spec['log1p'], spec['rs']]
    q['interactive_pick_new_run_name'] = [f"{spec['run_tag']}_s{spec['seed']}"]   # short: Windows path limit
    return q

# ---- 2. Grid ---------------------------------------------------------------------------------
# Capacity code: '0'..'4' = presets; anything else = custom menu entry 'e<E>i<I>h<H>'.
PRESET_CAP = {'0': (0, 0, 128), '1': (8, 0, 128), '2': (0, 64, 128), '3': (8, 64, 128), '4': (8, 64, 64)}
CUSTOM_E, CUSTOM_I, CUSTOM_H = (0, 4, 8, 16, 32), (0, 32, 64, 128), (32, 64, 128)

def cap_values(code):
    if code in PRESET_CAP:
        return PRESET_CAP[code]
    m = re.fullmatch(r'e(\d+)i(\d+)h(\d+)', code)
    return tuple(int(x) for x in m.groups())

def cap_code(e, i, h):
    """A custom triple equal to a preset is the preset (no duplicate runs)."""
    for k, v in PRESET_CAP.items():
        if v == (e, i, h):
            return k
    return f'e{e}i{i}h{h}'

ALL_CAPS = [cap_code(e, i, h) for e in CUSTOM_E for i in CUSTOM_I for h in CUSTOM_H]
PIPELINE = {'A': ('n', 'n'), 'B': ('y', 'n'), 'C': ('y', 'y'), 'D': ('n', 'y')}        # (visual_norm, log1p)

# Reference point of every single-factor ablation = the paper baseline (all menu defaults).
BASE = dict(use_visual='y', bbox_mode='0', visual_norm='n', percentile='n',
            capacity='0', log1p='n', rs='n')
NOVIS = dict(bbox_mode='0', visual_norm='n', percentile='n', capacity='0', log1p='n', rs='n')

def mk(**kw):
    return {**BASE, **kw}

def is_valid(c):
    if c['use_visual'] == 'n':                             # visual menus are never asked without VGG16
        return all(c[k] == v for k, v in NOVIS.items())
    return True

def is_coherent(c):
    """NOT verified against pie_intent.py: percentile scales are fitted on RAW train features
    but applied AFTER standardize/log1p in the graph, so percentile + (standardize or log1p) is flagged."""
    if c['use_visual'] == 'n':
        return True
    return not (c['percentile'] == 'y' and (c['visual_norm'] == 'y' or c['log1p'] == 'y'))

def full_grid(with_custom=False):
    caps = ALL_CAPS if with_custom else list(PRESET_CAP)
    for uv, bm, vn, pc, lg, rs, cap in itertools.product('yn', '01', 'yn', 'yn', 'yn', 'yn', caps):
        c = dict(use_visual=uv, bbox_mode=bm, visual_norm=vn, percentile=pc,
                 capacity=cap, log1p=lg, rs=rs)
        if is_valid(c):
            yield c

def logical_grid(with_custom=False):
    return (c for c in full_grid(with_custom) if is_coherent(c))

def isolated_grid():
    """One factor at a time from BASE; every other menu keeps its BASE value."""
    yield mk()
    yield mk(use_visual='n', **NOVIS)                      # bbox-only (no VGG16 context)
    yield mk(bbox_mode='1')                                # bbox zeroed in decoder; crop still encodes position/size
    for vn, pc in itertools.product('ny', 'ny'):
        yield mk(visual_norm=vn, percentile=pc)
    yield mk(log1p='y')
    yield mk(rs='y')
    e0, i0, h0 = cap_values(BASE['capacity'])
    for e in CUSTOM_E: yield mk(capacity=cap_code(e, i0, h0))      # encoder 1x1 alone
    for i in CUSTOM_I: yield mk(capacity=cap_code(e0, i, h0))      # input 1x1 alone
    for h in CUSTOM_H: yield mk(capacity=cap_code(e0, i0, h))      # decoder units alone
    for cap in PRESET_CAP: yield mk(capacity=cap)

def visual_grid():
    """{standardize} x {log1p} x {percentile} x five capacity presets."""
    for vn, lg, pc, cap in itertools.product('ny', 'ny', 'ny', PRESET_CAP):
        yield mk(visual_norm=vn, log1p=lg, percentile=pc, capacity=cap)

def capacity_grid(pipeline='A'):
    """All 60 custom-menu capacities x return_sequences under ONE visual pipeline."""
    vn, lg = PIPELINE[pipeline]
    for cap, rs in itertools.product(ALL_CAPS, 'ny'):
        yield mk(visual_norm=vn, log1p=lg, percentile='n', capacity=cap, rs=rs)

def cfg_id(c):
    # The constant 'ox' prefix is kept so run folders from earlier sweeps are still recognised (resume).
    return (f"ox_{c['use_visual']}{c['bbox_mode']}{c['visual_norm']}{c['percentile']}"
            f"{c['log1p']}{c['rs']}_c{c['capacity']}")

def run_tag(c):
    return 'h' + hashlib.md5(cfg_id(c).encode()).hexdigest()[:8]

# ---- 3. Prompt feeder ------------------------------------------------------------------------
class _Tee:
    def __init__(self, stream, logfile):
        self.stream, self.logfile, self.since_input, self.track = stream, logfile, [], True
    def write(self, s):
        self.stream.write(s)
        if not self.logfile.closed:                         # colorama writes here again at interpreter exit
            self.logfile.write(s)
        if self.track:
            self.since_input.append(s)
        return len(s)
    def flush(self):
        self.stream.flush()
        if not self.logfile.closed:
            self.logfile.flush()
    def __getattr__(self, name):
        return getattr(self.stream, name)

class Feeder:
    def __init__(self, queues, tee):
        self.queues, self.tee, self.audit = queues, tee, []

    def _key(self, prompt, frame):
        for tag, key in (('[visual_percentile]', 'visual_percentile'), ('[plateau]', 'plateau'),
                         ('[tte]', 'tte')):
            if tag in prompt:
                return key
        f = frame
        while f is not None:
            name = f.f_code.co_name
            if name in self.queues or name in ALWAYS:
                return name
            if name.startswith(('interactive_pick_', 'confirm_')):
                return name                                 # a menu with no pinned answer
            f = f.f_back
        return None

    def __call__(self, prompt=''):
        text = ''.join(self.tee.since_input) + str(prompt)
        self.tee.since_input.clear()
        key = self._key(str(prompt), sys._getframe(1))
        q = self.queues.get(key)
        if q:
            ans = q.pop(0)
        elif key in ALWAYS:
            ans = ALWAYS[key]
        else:
            raise RuntimeError(f'[feeder] no pinned answer for menu: {key}\n'
                               f'--- menu text ---\n{text[-1200:]}')
        self.audit.append({'menu': key, 'answer': ans})
        print(f'[feeder] {key} <- {ans!r}')
        self.tee.since_input.clear()
        if key == 'interactive_pick_new_run_name':           # last menu: stop buffering stdout
            self.tee.track = False
        return ans

# ---- 4. Verification + output saving -----------------------------------------------------------
def verify_opts(o, spec):
    """Fails the run if the menus did not produce what the spec says (guards against wrong pinned indices)."""
    uv = spec['use_visual'] == 'y'
    assert bool(o['use_visual']) == uv, ('use_visual', o['use_visual'])
    assert not o['offline'], ('offline features must be empty', o['offline'])
    assert o['balance_mode'] == 'undersample_base_article', o['balance_mode']
    assert o['monitor_metric'] == 'val_loss' and o['monitor_mode'] == 'min'
    assert o['split_type'] == spec['split'] and o['batch_size'] == 64 and o['eval_every'] is None
    assert o['normalize_bbox'] is False and o['rec_act'] == 'hard_sigmoid' and o['use_plateau'] is True
    assert (o['train_seed'], o['data_seed'], o['split_seed']) == (spec['seed'], 42, 42)
    if uv:
        assert bool(o['zero_decoder_bbox']) == (spec['bbox_mode'] == '1'), o['zero_decoder_bbox']
        assert bool(o['visual_norm']) == (spec['visual_norm'] == 'y'), o['visual_norm']
        assert bool(o['use_percentile']) == (spec['percentile'] == 'y'), o['use_percentile']
        c = o['capacity']
        assert (c['encoder_reduce'], c['input_reduce'], c['hidden_units']) == cap_values(spec['capacity']), c
        assert bool(c['log1p']) == (spec['log1p'] == 'y'), c
        assert bool(c['return_sequences']) == (spec['rs'] == 'y'), c

def _jsonable(o):
    import numpy as np
    if o is None or isinstance(o, (str, int, float, bool)): return o
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist() if o.size <= 10000 else f'<ndarray {o.shape}>'
    if isinstance(o, dict): return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [_jsonable(v) for v in o]
    return repr(o)[:200]

def append_row(row):
    with open(OUT_ROOT / 'summary.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(_jsonable(row)) + '\n')

def n_rows():
    p = OUT_ROOT / 'summary.jsonl'
    return sum(1 for l in p.read_text(encoding='utf-8').splitlines() if l.strip()) if p.exists() else 0

METRIC_KEYS = ('roc_auc', 'pr_auc', 'accuracy', 'balanced_acc', 'f1', 'nll', 'ece', 'n_test_samples')

def save_run_outputs(run_dir, spec, feeder, result, status, err, minutes):
    """Per run: config.json, history.json, console.log in its own folder; one row in summary.jsonl and one line
    in all_histories.jsonl. Heavy files (model.h5, ped_intents*.pkl, plots) stay in the pipeline's own run_dir
    (see the 'run_dir' column)."""
    import numpy as np
    run_dir = Path(run_dir)
    (run_dir / 'config.json').write_text(json.dumps(_jsonable({
        'spec': spec, 'feeder_audit': feeder.audit, 'HP': HP, 'DATA_OPTS': DATA_OPTS,
        'python': sys.version}), indent=2), encoding='utf-8')
    row = {'run_id': spec['run_id'], 'run_tag': spec['run_tag'], 'cfg_id': spec['cfg_id'], 'seed': spec['seed'],
           'split': spec['split'], 'status': status, 'minutes': round(minutes, 2), 'coherent': is_coherent(spec),
           'use_visual': spec['use_visual'], 'bbox_mode': spec['bbox_mode'], 'visual_norm': spec['visual_norm'],
           'percentile': spec['percentile'], 'capacity': spec['capacity'], 'log1p': spec['log1p'],
           'rs': spec['rs'], 'cap_E_I_H': '-'.join(map(str, cap_values(spec['capacity']))),
           'error': (err or '')[-300:]}
    if result:
        o, hist = result['opts'], (result.get('history') or {})
        row['run_dir'] = result.get('saved_files_path')
        vals = list(hist.get(o['monitor_metric'], []))      # best epoch by the MONITORED metric
        row['n_epochs_run'] = len(vals) or None
        if vals:
            row['best_epoch'] = int((np.argmin if o['monitor_mode'] == 'min' else np.argmax)(vals)) + 1
        (run_dir / 'history.json').write_text(json.dumps(_jsonable(hist)), encoding='utf-8')
        with open(OUT_ROOT / 'all_histories.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps({'run_id': spec['run_id'], 'cfg_id': spec['cfg_id'], 'seed': spec['seed'],
                                'history': _jsonable(hist)}) + '\n')
        for s in ('train', 'val', 'test'):
            m, mc = result['metrics'][s], result['metrics_calibrated'][s]
            for k in METRIC_KEYS:
                row[f'{s}_{k}'] = m.get(k)
            row[f'{s}_f1_cal'], row[f'{s}_acc_cal'] = mc.get('f1'), mc.get('accuracy')
        row['cal_threshold'] = result.get('threshold')
        try:                                                # pedestrian-clustered bootstrap CI
            from train_helpers import load_ped_intents_full, bootstrap_auc_by_group
            for s in ('val', 'test'):
                p, y, pid = load_ped_intents_full(row['run_dir'], s)
                _, lo, hi = bootstrap_auc_by_group(y, p, pid, n_boot=1000)
                row[f'{s}_auc_lo'], row[f'{s}_auc_hi'], row[f'{s}_n_peds'] = lo, hi, len(set(pid))
        except Exception as e:
            print(f'[save] bootstrap CI skipped: {type(e).__name__}: {e}')
    append_row(row)
    if status == 'ok':
        (run_dir / 'done.json').write_text(json.dumps({'minutes': minutes}))

def print_result_paths():
    print('\n' + '=' * 90)
    print('[sweep] RESULTS ARE SAVED IN:')
    print(f'  main table (Excel / pandas)        : {OUT_ROOT / "summary.csv"}')
    print(f'  same data, append-only, 1 row/run  : {OUT_ROOT / "summary.jsonl"}')
    print(f'  per-epoch history of every run     : {OUT_ROOT / "all_histories.jsonl"}')
    print(f'  list of planned runs               : {OUT_ROOT / "planned_runs.txt"}')
    print(f'  per-run folders (console.log, config.json, history.json): {OUT_ROOT}{os.sep}<run_id>{os.sep}')
    print('  heavy files (model.h5, ped_intents*.pkl, metrics_*.json, plots): see the run_dir column of summary.csv')
    print('=' * 90, flush=True)

def aggregate(quiet=False):
    import pandas as pd
    p = OUT_ROOT / 'summary.jsonl'
    if not p.exists():
        print('[sweep] no summary.jsonl yet'); return None
    df = pd.DataFrame([json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()])
    df = df.drop_duplicates('run_id', keep='last')
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    if quiet:
        return df
    ok = df[df.status == 'ok']
    print(f'{len(ok)} ok / {len(df)} rows; not ok: {sorted(df[df.status != "ok"].run_id)[:10]}')
    if not ok.empty:
        g = ok.groupby('cfg_id').agg(n=('seed', 'count'), coherent=('coherent', 'first'),
                                     val_auc=('val_roc_auc', 'mean'), val_sd=('val_roc_auc', 'std'),
                                     test_auc=('test_roc_auc', 'mean'), test_sd=('test_roc_auc', 'std'),
                                     min=('minutes', 'mean'))
        print(g.sort_values('val_auc', ascending=False).round(4).to_string())   # rank by VAL, never by test
    return df

# ---- 5. Child (one run per process) and driver ------------------------------------------------------
def child(spec_path):
    global OUT_ROOT
    spec = json.loads(Path(spec_path).read_text())
    run_dir = Path(spec['run_dir'])
    OUT_ROOT = run_dir.parent                              # same results folder as the driver
    logf = open(run_dir / 'console.log', 'w', encoding='utf-8')
    tee = _Tee(sys.stdout, logf); sys.stdout = tee
    feeder = Feeder(build_queues(spec), tee); builtins.input = feeder
    os.environ['PIE_PATH'] = PIE_PATH
    hp = {**HP, 'seed': spec['seed']}
    t0, status, result, err = time.time(), 'ok', None, None
    try:
        from train_pipeline import run_training_pipeline
        result = run_training_pipeline(hp, DATA_OPTS, PIE_PATH)
        verify_opts(result['opts'], spec)
    except BaseException:
        status, err = 'failed', traceback.format_exc()
        print(err)
    save_run_outputs(run_dir, spec, feeder, result, status, err, (time.time() - t0) / 60)
    sys.stdout = tee.stream; logf.close()
    sys.exit(0 if status == 'ok' else 1)

STAGE_ORDER = ['isolated', 'visual', 'capacity', 'logical']     # what '--stage all' runs, most informative first

def grid_for(stage, pipeline, with_custom):
    return {'isolated': isolated_grid, 'visual': visual_grid,
            'capacity': lambda: capacity_grid(pipeline),
            'logical': lambda: logical_grid(with_custom),
            'full': lambda: full_grid(with_custom)}[stage]()

def build_jobs(stage, seeds, with_custom, pipeline, sample, sample_seed, limit, include_incoherent):
    """Blocks run in priority order; inside a block, seed-major. A config already planned in an earlier
    block is not repeated. Configs flagged incoherent are skipped unless include_incoherent (or stage 'full')."""
    stages = STAGE_ORDER if stage == 'all' else [stage]
    jobs, info, seen = [], [], set()
    for st in stages:
        cfgs = []
        for c in grid_for(st, pipeline, with_custom and stage != 'all'):   # 'all' never adds the custom caps
            if cfg_id(c) in seen:
                continue
            if st != 'full' and not include_incoherent and not is_coherent(c):
                continue
            seen.add(cfg_id(c)); cfgs.append(c)
        if sample and sample < len(cfgs) and (stage != 'all' or st == 'logical'):
            cfgs = random.Random(sample_seed).sample(cfgs, sample)
        info.append((st, len(cfgs)))
        jobs += [(c, s) for s in seeds for c in cfgs]
    return jobs[: limit or None], info

def measured_minutes():
    p = OUT_ROOT / 'summary.jsonl'
    vals = []
    if p.exists():
        for l in p.read_text(encoding='utf-8').splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get('status') == 'ok' and r.get('minutes'):
                    vals.append(r['minutes'])
    return (statistics.median(vals), len(vals)) if vals else (DEFAULT_MIN_PER_RUN, 0)

def driver(a):
    OUT_ROOT.mkdir(exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(',')]
    jobs, info = build_jobs(a.stage, seeds, a.with_custom, a.pipeline, a.sample, a.sample_seed, a.limit,
                            a.include_incoherent)
    (OUT_ROOT / 'planned_runs.txt').write_text(
        '\n'.join(f'{cfg_id(c)}_s{s}' for c, s in jobs), encoding='utf-8')
    print(f'[sweep] stage={a.stage}: ' + ', '.join(f'{st}={n} configs' for st, n in info)
          + f' | seeds={seeds} | {len(jobs)} runs ({sum(not is_coherent(c) for c, _ in jobs)} incoherent)')
    done = lambda c, s: (OUT_ROOT / f'{cfg_id(c)}_s{s}' / 'done.json').exists()
    pending = [(c, s) for c, s in jobs if not done(c, s)]
    med, n_ok = measured_minutes()
    est_h = len(pending) * med / 60
    print(f'[sweep] pending {len(pending)} runs x {med:.0f} min '
          f'({"median of " + str(n_ok) + " finished runs" if n_ok else "default estimate"}) '
          f'= ~{est_h:.0f} h (~{est_h / 24:.1f} days). Runs without VGG16 are faster.')
    print(f'[sweep] data split: {a.split} | results folder: {OUT_ROOT}')
    if a.dry_run:
        print(f'[sweep] dry run: nothing executed. Planned runs listed in {OUT_ROOT / "planned_runs.txt"}')
        return
    t_start, current = time.time(), None
    try:
        for i, (c, s) in enumerate(jobs, 1):
            rid = f'{cfg_id(c)}_s{s}'; run_dir = OUT_ROOT / rid
            if done(c, s):
                print(f'[sweep] {i}/{len(jobs)} skip {rid}'); continue
            if a.max_hours and (time.time() - t_start) / 3600 >= a.max_hours:
                print(f'[sweep] --max-hours {a.max_hours} reached: stopping before {rid}. '
                      f'Run the same command again to resume.')
                break
            current = (rid, c, s)
            run_dir.mkdir(parents=True, exist_ok=True)
            spec_path = run_dir / 'spec.json'
            spec_path.write_text(json.dumps({**c, 'seed': s, 'run_id': rid, 'cfg_id': cfg_id(c),
                                             'run_tag': run_tag(c), 'run_dir': str(run_dir),
                                             'split': a.split}))
            print(f'[sweep] {i}/{len(jobs)} start {rid}', flush=True)
            before = n_rows()
            rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--one', str(spec_path)],
                                cwd=HERE, stdin=subprocess.DEVNULL).returncode
            print(f'[sweep] {i}/{len(jobs)} end {rid} rc={rc}', flush=True)
            if rc != 0 and n_rows() == before:      # child died before writing its row (OOM, CUDA, kill)
                append_row({'run_id': rid, 'cfg_id': cfg_id(c), 'seed': s, 'status': 'crashed', 'rc': rc})
            current = None
            try:
                aggregate(quiet=True)
            except Exception as e:
                print(f'[sweep] aggregate skipped: {type(e).__name__}: {e}')
    except KeyboardInterrupt:
        print('\n[sweep] interrupted. Run the same command again to resume (finished runs are skipped).')
        if current:
            append_row({'run_id': current[0], 'cfg_id': cfg_id(current[1]), 'seed': current[2],
                        'status': 'interrupted'})
    finally:
        try:
            aggregate()
        except Exception as e:
            print(f'[sweep] final aggregate failed: {type(e).__name__}: {e}')
        print_result_paths()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--one'); ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--stage', choices=['all', 'isolated', 'visual', 'capacity', 'logical', 'full'], default='isolated')
    ap.add_argument('--seeds', default='1,2,3'); ap.add_argument('--limit', type=int)
    ap.add_argument('--with-custom', action='store_true', help='logical/full: add the 55 custom capacities')
    ap.add_argument('--pipeline', choices=list(PIPELINE), default='A', help='capacity stage: visual pipeline A/B/C/D')
    ap.add_argument('--sample', type=int, help='random subset of N configs (logical/full block only with --stage all)')
    ap.add_argument('--sample-seed', type=int, default=0)
    ap.add_argument('--max-hours', type=float, help='do not START a new run after this many hours')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--include-incoherent', action='store_true',
                    help='also run configs flagged by is_coherent() (percentile combined with standardize/log1p)')
    ap.add_argument('--split', choices=['default', 'random'], default='default',
                    help="'default' = fixed video sets, 'random' = pedestrian-id 0.5/0.4/0.1")
    a = ap.parse_args()
    if a.one is None:                                      # driver / --aggregate (the child derives it from its run folder)
        OUT_ROOT = HERE / ('sweep_runs' if a.split == 'random' else f'sweep_runs_{a.split}')
    if a.one: child(a.one)
    elif a.aggregate:
        aggregate(); print_result_paths()
    else: driver(a)