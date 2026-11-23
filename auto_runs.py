"""auto_runs.py -- unattended menu answers for the audit runs A, B, C, D.

  %run auto_runs.py --run A C --seeds 1     # sequential, one subprocess per run, resumable
  %run auto_runs.py --compare               # val-only comparison + paired bootstrap
  %run auto_runs.py --probe-light           # 300 frames, checks the light-colour rule, no TensorFlow
"""
import argparse, builtins, faulthandler, json, os, random, re, subprocess, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIE_PATH = os.environ.get('PIE_PATH', r'D:\PIE codes\PIE_dataset')
OUT = HERE / 'audit_runs'

HP = dict(
    framework='tensorflow', lr=1e-5, clipvalue=None, batch_size=64,
    hidden_units=128, reg=0.001, lstm_dropout=0.4, lstm_recurrent_dropout=0.2,
    convlstm_filters=64, convlstm_kernel=2,
    es_min_delta=1e-4, es_patience=10,
    plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-7, plateau_min_delta=1e-3,
    n_epochs=400, seed=1, data_seed=42, split_seed=42, deterministic_ops=True,
    attach_pose=True, log_mode='clean', threshold_space='raw', decision_threshold=0.5,
)
DATA_OPTS = {
    'fstride': 1, 'sample_type': 'all', 'height_rng': [0, float('inf')], 'squarify_ratio': 0,
    'data_split_type': 'default', 'seq_type': 'intention',
    'min_track_size': 15, 'max_size_observe': 15, 'max_size_predict': 5, 'seq_overlap_rate': 0.5,
    'balance': True, 'crop_type': 'context', 'crop_mode': 'pad_resize',
    'encoder_input_type': [], 'decoder_input_type': ['bbox'], 'output_type': ['intention_binary'],
}

SPECS = {
    'A': dict(name='scene_only', prefix='aA', offline='1', families={'scene'}, light=False,
              feats=['scene_critical'], leakage_prone=False),
    'B': dict(name='scene_pose', prefix='aB', offline='0,1', families={'pose', 'scene'}, light=False,
              feats=['scene_critical', 'pose_critical'], leakage_prone=False),
    'C': dict(name='control', prefix='aC', offline='a', families={'pose', 'scene', 'flow'}, light=False,
              feats=['pose_critical', 'scene_critical', 'flow_res', 'flow_rel_ped_crop',
                     'flow_rel_bbox_crop', 'flow_critical'], leakage_prone=True),   # flow_* carry ego-motion cues
    'D': dict(name='scene_light', prefix='aD', offline='1', families={'scene'}, light=True,
              feats=['scene_critical', 'light_color'], leakage_prone=True),         # light seen by the ego camera = ego-state proxy
}
FORBIDDEN = {'obd_speed', 'gps_speed', 'speed', 'velocity', 'look', 'action', 'gesture', 'cross', 'tte'}


def build_queues(spec, seed):
    return {
        'interactive_pick_offline_features': [spec['offline']],
        'interactive_pick_use_visual': ['y'],
        'interactive_pick_architecture': ['5'],               # branchattn_temporal
        'interactive_pick_branch_mode': ['3'],                # core_plus_extras
        'choose_branch_temporal_for_menu': ['n', 'n'],        # keep ConvLSTM2D and per-branch LSTM
        'interactive_pick_zero_decoder_bbox': ['0'],
        'interactive_pick_visual_norm': ['n'],
        'visual_percentile': ['n'],
        'interactive_pick_capacity': ['0', 'n', 'n'],         # baseline, no log1p, no return_sequences
        'interactive_pick_log_mode': ['0'],
        'interactive_pick_balance_mode': ['2'],               # none
        'interactive_pick_monitor_metric': ['1'],             # val_auc (max)
        'interactive_pick_loss_choice': ['4', '', '', '0'],   # wbce+focal, gamma 2.0, weight 1.0, pos_weight auto
        'interactive_pick_clip_mode': ['3'],                  # none
        'interactive_pick_weight_decay': [''],
        'interactive_pick_lstm_recurrent_activation': ['2'],  # hard_sigmoid
        'interactive_pick_normalize_bbox': ['0'],             # raw pixels
        'plateau': ['y'],
        'interactive_pick_eval_every': ['4'],                 # never
        'interactive_pick_pose_scale': ['0'],                 # bbox_relative
        'interactive_pick_pose_min_height': ['1'],            # 40 px
        'tte': ['n'],
        'interactive_pick_data_split': ['0'],                 # default split
        'interactive_pick_seeds': [str(seed), '', '42'],      # train, shuffle (= train), data
        'interactive_pick_decoder_features': [None],          # built from the printed menu
        'interactive_pick_extra_norm': ['1'],                 # zscore
        'interactive_pick_aux_loss': ['0.5'],
        'interactive_pick_new_run_name': [spec['prefix']],
    }


ALWAYS = {'interactive_pick_feature_norms': '1'}              # zscore for every selected extra


class Tee:
    def __init__(self, stream, logfile):
        self.stream, self.logfile, self.since_input, self.track = stream, logfile, [], True

    def write(self, s):
        self.stream.write(s)
        if not self.logfile.closed:
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
    def __init__(self, queues, tee, feats):
        self.queues, self.tee, self.feats, self.audit = queues, tee, feats, []

    def _key(self, prompt, frame):
        for tag, key in (('[visual_percentile]', 'visual_percentile'), ('[plateau]', 'plateau'), ('[tte]', 'tte')):
            if tag in prompt:
                return key
        f = frame
        while f is not None:
            name = f.f_code.co_name
            if name in self.queues or name in ALWAYS or name.startswith(('interactive_pick_', 'confirm_')):
                return name
            f = f.f_back
        return None

    def _decoder_answer(self, text):
        menu = {name: idx for idx, name in re.findall(r'\[(\d+)\] \+(\w+)', text)}
        missing = [n for n in self.feats if n not in menu]
        if missing:
            raise RuntimeError(f'[feeder] features {missing} are not in the decoder menu. '
                               f'Available: {sorted(menu)}')
        return ','.join(menu[n] for n in self.feats)

    def __call__(self, prompt=''):
        text = ''.join(self.tee.since_input) + str(prompt)
        self.tee.since_input.clear()
        key = self._key(str(prompt), sys._getframe(1))
        q = self.queues.get(key)
        if q:
            ans = q.pop(0)
            if key == 'interactive_pick_decoder_features':
                ans = self._decoder_answer(text)
        elif key in ALWAYS:
            ans = ALWAYS[key]
        else:
            raise RuntimeError(f'[feeder] no pinned answer for menu: {key}\n--- menu text ---\n{text[-1200:]}')
        self.audit.append({'menu': key, 'answer': ans})
        print(f'[feeder] {key} <- {ans!r}')
        self.tee.since_input.clear()
        if key == 'interactive_pick_new_run_name':            # last menu: stop buffering stdout
            self.tee.track = False
        return ans


def verify(result, spec, seed):
    o = result['opts']
    checks = {
        'offline': set(o['offline']) == spec['families'], 'use_visual': o['use_visual'] is True,
        'architecture': o['architecture'] == 'branchattn_temporal', 'branch_mode': o['branch_mode'] == 'core_plus_extras',
        'branch_encoder': o['branch_visual_encoder'] in (None, 'convlstm2d') and o['branch_temporal_kind'] == 'lstm',
        'balance': o['balance_mode'] == 'none', 'monitor': (o['monitor_metric'], o['monitor_mode']) == ('val_auc', 'max'),
        'loss': str(o['loss_name']).startswith('wbce_focal_g2.0_w1.0'), 'clip': o['clip_type'] == 'none',
        'batch': o['batch_size'] == 64, 'eval_every': o['eval_every'] is None, 'bbox_raw': o['normalize_bbox'] is False,
        'rec_act': o['rec_act'] == 'hard_sigmoid', 'split': o['split_type'] == 'default',
        'seeds': (o['train_seed'], o['shuffle_seed'], o['data_seed'], o['split_seed']) == (seed, seed, 42, 42),
        'aux': abs(float(o['aux_loss_weight']) - 0.5) < 1e-9, 'extra_norm': o['extra_norm'] == 'zscore',
        'capacity': (o['capacity'] or {}).get('hidden_units') == 128,
        'zero_bbox': not o['zero_decoder_bbox'], 'vnorm': not o['visual_norm'],
    }
    bad = [k for k, v in checks.items() if not v]
    choices = json.loads((Path(result['saved_files_path']) / 'choices.json').read_text(encoding='utf-8'))
    if choices['decoder_features'] != ['bbox'] + spec['feats']:
        bad.append(('decoder_features', choices['decoder_features']))
    if set(choices['decoder_features']) & FORBIDDEN:
        bad.append(('forbidden feature selected', choices['decoder_features']))
    if bad:
        raise AssertionError(f'menus did not produce the spec: {bad}')


METRIC_KEYS = ('roc_auc', 'pr_auc', 'accuracy', 'specificity', 'balanced_acc', 'f1', 'nll', 'ece', 'n_test_samples')


def summarize(result, spec, run_id, seed, minutes, status, err):
    row = dict(run_id=run_id, spec=run_id.split('_')[0], name=spec['name'], seed=seed, status=status,
               minutes=round(minutes, 1), feats=spec['feats'], leakage_prone=spec['leakage_prone'],
               min_track_size=DATA_OPTS['min_track_size'], error=(err or '')[-400:])
    if not result:
        return row
    run_dir = Path(result['saved_files_path'])
    row['run_dir'] = str(run_dir)
    for s in ('train', 'val', 'test'):
        row[s] = {k: result['metrics'][s].get(k) for k in METRIC_KEYS}
    try:
        em = json.loads((run_dir / 'epoch_metrics.json').read_text(encoding='utf-8'))
        auc_key = 'val_roc_auc' if 'val_roc_auc' in em[0] else 'val_auc'
        best = max(em, key=lambda r: r.get(auc_key) if r.get(auc_key) is not None else -1)
        row['n_epochs'], row['best_epoch'] = len(em), best['epoch']
        row['head_val_auc_at_best'] = {k[len('val_auc_'):]: v for k, v in best.items() if k.startswith('val_auc_')}
        att = json.loads((run_dir / 'attention_log.json').read_text(encoding='utf-8'))
        rec = next((r for r in att if r['epoch'] == best['epoch']), att[-1])
        row['attention_at_best'] = dict(zip(rec['names'], rec['mean']))
        row['attention_entropy'] = rec['entropy']
    except Exception as e:
        row['summary_note'] = f'{type(e).__name__}: {e}'
    return row


def child(spec_id, seed):
    spec, run_id = SPECS[spec_id], f'{spec_id}_s{seed}'
    out = OUT / run_id
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    os.environ['PIE_PATH'] = PIE_PATH
    faulthandler.enable(file=open(HERE / 'faulthandler.log', 'a'), all_threads=True)
    logf = open(out / 'console.log', 'w', encoding='utf-8')
    tee = Tee(sys.stdout, logf)
    sys.stdout = tee
    feeder = Feeder(build_queues(spec, seed), tee, spec['feats'])
    builtins.input = feeder
    assert DATA_OPTS['min_track_size'] == 15 and not (set(spec['feats']) & FORBIDDEN)
    t0, status, result, err = time.time(), 'ok', None, None
    try:
        from train_pipeline import run_training_pipeline
        result = run_training_pipeline({**HP, 'seed': seed}, DATA_OPTS, PIE_PATH)
    except BaseException:
        status, err = 'failed', traceback.format_exc()
        print(err)
    if result is not None:
        try:
            verify(result, spec, seed)
        except BaseException:
            status, err = 'mismatch', traceback.format_exc()
            print(err)
    row = summarize(result, spec, run_id, seed, (time.time() - t0) / 60, status, err)
    with open(OUT / 'index.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, default=str) + '\n')
    if result is not None:   # a trained run is never repeated automatically, even if verify() disagreed
        (out / 'done.json').write_text(json.dumps({'minutes': row['minutes'], 'status': status}))
    sys.stdout = tee.stream
    logf.close()
    sys.exit(0 if status == 'ok' else 1)


def driver(ids, seeds):
    OUT.mkdir(exist_ok=True)
    jobs = [(i, s) for i in ids for s in seeds]
    print(f'[auto] plan: {[f"{i}_s{s}" for i, s in jobs]}; results in {OUT}')
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}
    for n, (i, s) in enumerate(jobs, 1):
        rid = f'{i}_s{s}'
        if (OUT / rid / 'done.json').exists():
            print(f'[auto] {n}/{len(jobs)} skip {rid} (done.json exists)')
            continue
        print(f'[auto] {n}/{len(jobs)} start {rid} at {time.strftime("%H:%M:%S")}', flush=True)
        p = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--one', i, '--seed', str(s)],
                             cwd=HERE, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=env)
        for line in p.stdout:
            print(line, end='')
        print(f'[auto] {n}/{len(jobs)} end {rid} rc={p.wait()} at {time.strftime("%H:%M:%S")}', flush=True)


def _rows():
    p = OUT / 'index.jsonl'
    rows = {}
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r['run_id']] = r
    return rows


def paired_delta(dir_a, dir_b, part='val', n_boot=2000):
    """AUC(b) - AUC(a), pedestrian-clustered bootstrap. Both runs must score the same windows."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from train_helpers import load_ped_intents_full
    pa, ya, ida = load_ped_intents_full(dir_a, part)
    pb, yb, idb = load_ped_intents_full(dir_b, part)
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    ya, yb = np.asarray(ya).ravel().astype(int), np.asarray(yb).ravel().astype(int)
    ida, idb = np.asarray(ida).astype(str), np.asarray(idb).astype(str)
    if not (np.array_equal(ya, yb) and np.array_equal(ida, idb)):
        raise ValueError('the two runs do not score the same windows')
    groups = {g: np.where(ida == g)[0] for g in np.unique(ida)}
    keys, rng, deltas = list(groups), np.random.RandomState(0), []
    for _ in range(n_boot):
        idx = np.concatenate([groups[keys[i]] for i in rng.randint(0, len(keys), len(keys))])
        if len(np.unique(ya[idx])) < 2:
            continue
        deltas.append(roc_auc_score(ya[idx], pb[idx]) - roc_auc_score(ya[idx], pa[idx]))
    point = roc_auc_score(ya, pb) - roc_auc_score(ya, pa)
    return point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def compare():
    import numpy as np
    allrows = _rows()
    bad = sorted(k for k, r in allrows.items() if r['status'] != 'ok')
    if bad:
        print(f'[compare] excluded (failed or config mismatch, see console.log): {bad}')
    ok = [r for r in allrows.values() if r['status'] == 'ok']
    if not ok:
        print('[compare] no finished runs')
        return
    by_spec = {}
    for r in ok:
        by_spec.setdefault(r['spec'], {})[r['seed']] = r
    print(f'{"run":6s} {"features":58s} {"val_auc":>7s} {"val_spec":>8s} {"val_bAcc":>8s} {"best_ep":>7s}  '
          f'test_auc (observed, not used for choices)')
    for k in sorted(by_spec):
        for seed in sorted(by_spec[k]):
            r = by_spec[k][seed]
            v, t = r['val'], r['test']
            tag = ' [leakage-prone]' if r['leakage_prone'] else ''
            feats = ','.join(r['feats']) + tag
            print(f'{r["run_id"]:6s} {feats:58s} {v["roc_auc"]:7.4f} {(v["specificity"] or 0):8.4f} '
                  f'{(v["balanced_acc"] or 0):8.4f} {r.get("best_epoch", "-"):>7}  {t["roc_auc"]:.4f}')
            h = r.get('head_val_auc_at_best')
            if h:
                print('       branch val AUC: ' + ' '.join(f'{n}={a:.3f}' for n, a in h.items()))
        if len(by_spec[k]) > 1:
            a = np.array([by_spec[k][s]['val']['roc_auc'] for s in by_spec[k]])
            print(f'  {k}: val AUC mean {a.mean():.4f}, std {a.std(ddof=1):.4f} over seeds {sorted(by_spec[k])}')
    print('\n[compare] paired val AUC differences per seed (pedestrian bootstrap, 95% CI); '
          'CI covers data noise only, not seed noise')
    for a, b, what in (('A', 'C', 'C-A: do flows/pose add anything?'), ('A', 'B', 'B-A: does pose_critical add?'),
                       ('A', 'D', 'D-A: does light_color add?')):
        for seed in sorted(set(by_spec.get(a, {})) & set(by_spec.get(b, {}))):
            try:
                d, lo, hi = paired_delta(by_spec[a][seed]['run_dir'], by_spec[b][seed]['run_dir'])
                verdict = 'CI excludes 0' if lo > 0 or hi < 0 else 'CI includes 0 -> no evidence of a difference'
                print(f'  {what:34s} seed {seed}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}')
            except Exception as e:
                print(f'  {what} seed {seed}: skipped ({type(e).__name__}: {e})')


def probe_light(n=300):
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(HERE))
    from extra_features import traffic_light_color
    label_root = Path(PIE_PATH) / 'extra' / 'scene'
    files = sorted(label_root.glob('set*/video_*/*.png'))
    sample = random.Random(0).sample(files, min(n, len(files)))
    t0, n_light, n_valid, cls = time.time(), 0, 0, np.zeros(3)
    for f in sample:
        lm = np.asarray(Image.open(f))
        if int((lm == 6).sum()) < 20:
            continue
        n_light += 1
        img = Path(PIE_PATH) / 'images' / f.parent.parent.name / f.parent.name / f.name
        if not img.is_file():
            continue
        v = traffic_light_color(np.asarray(Image.open(img).convert('RGB')), lm, None, lm.shape[1] / 1920.0)
        if v[3] == 1.0:
            n_valid += 1
            cls += v[:3]
    dt = time.time() - t0
    print(f'[probe] {len(sample)} frames: {n_light} have >=20 light pixels, {n_valid} gave a usable lamp')
    if n_valid:
        print(f'[probe] mean red/green/yellow fraction over usable frames: {np.round(cls / n_valid, 3).tolist()}')
    print(f'[probe] {1000 * dt / max(len(sample), 1):.0f} ms/frame')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', nargs='+', choices=sorted(SPECS))
    ap.add_argument('--one', choices=sorted(SPECS))
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--seeds', default='1')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--probe-light', action='store_true')
    a = ap.parse_args()
    if a.one:
        child(a.one, a.seed)
    elif a.run:
        driver(a.run, [int(s) for s in a.seeds.split(',')])
    elif a.compare:
        compare()
    elif a.probe_light:
        probe_light()
    else:
        ap.print_help()