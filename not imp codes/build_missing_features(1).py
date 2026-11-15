"""
build_missing_features.py -- run LOCALLY (pie_tf env, raw images available).

Computes ONLY the VGG16 features that check_cache_coverage.py reports as missing, with the same
code path training uses (PIEIntent.load_images_and_process: same crop, flip, photometric factors,
preprocess, cache file names). No training, no .dat files, no menus. Re-running skips what exists,
so it can be stopped and restarted.

Typical use after check_cache_coverage.py reported missing frames for 'augmentation':
  conda activate pie_tf
  cd /d "D:\\PIE codes\\PIE"
  python build_missing_features.py --pie-path "D:\\PIE codes\\PIE_dataset" --modes augmentation --dry-run
  python build_missing_features.py --pie-path "D:\\PIE codes\\PIE_dataset" --modes augmentation
  python check_cache_coverage.py  --pie-path "D:\\PIE codes\\PIE_dataset" --modes none augmentation
The last command must end with [OK] before running compress_features.py pack.

Needs the raw images of every frame in the missing list (they are on this machine, not on Colab).
"""
import argparse
import json
import os
import sys
import time


def collect_missing(split_seqs, index, frame_key):
    """split_seqs: {subset: sequence dict}; index: {subset: set of present keys}.
    Returns {subset: {key: (image_path, bbox, [ped_id])}} for every frame of every track that is not
    cached. All frames of a track are needed: windows at overlap 1 start at every offset, so they
    cover every frame."""
    todo = {s: {} for s in split_seqs}
    for subset, seq in split_seqs.items():
        for imgs, boxes, pids in zip(seq['image'], seq['bbox'], seq['ped_id']):
            for imp, box, pid in zip(imgs, boxes, pids):
                key = frame_key(imp, pid[0])
                if key not in index[subset] and key not in todo[subset]:
                    todo[subset][key] = (imp, list(box), pid)
    return todo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pie-path', required=True)
    ap.add_argument('--code-dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--modes', nargs='+', default=['augmentation'],
                    choices=['none', 'undersample_base_article', 'augmentation'])
    ap.add_argument('--val-augment-modes', nargs='+', default=['undersample_val'],
                    choices=['undersample_val', 'augment_val', 'natural_val'])
    ap.add_argument('--split', default='default', choices=['default', 'random'])
    ap.add_argument('--ratios', default='0.5,0.4,0.1')
    ap.add_argument('--data-opts-json', default=None)
    ap.add_argument('--chunk', type=int, default=128, help='frames per call (VGG batches of 32 inside)')
    ap.add_argument('--dry-run', action='store_true', help='only count what would be computed')
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(args.code_dir, 'pie_data_Hesam.py')):
        raise SystemExit(f"pie_data_Hesam.py not found in --code-dir '{args.code_dir}'")
    if args.data_opts_json and not os.path.isfile(args.data_opts_json):
        raise SystemExit(f'--data-opts-json file not found: {os.path.abspath(args.data_opts_json)}')
    os.environ['PIE_PATH'] = args.pie_path
    sys.path.insert(0, args.code_dir)

    from check_cache_coverage import DEFAULT_DATA_OPTS, frame_key, build_cache_index
    from pie_data_Hesam import PIE
    from augmentation import augment_pie_tracks
    from train_helpers import build_split_sequences, apply_balancing_strategy
    from pie_intent import FEATURE_CACHE_NAME  # single source of truth for the cache folder name

    data_opts = dict(DEFAULT_DATA_OPTS)
    if args.data_opts_json:
        with open(args.data_opts_json, encoding='utf-8') as fh:
            data_opts.update(json.load(fh))
    data_opts['data_split_type'] = args.split
    if args.split == 'random':
        data_opts['random_params'] = {'ratios': [float(x) for x in args.ratios.split(',')],
                                      'val_data': True, 'regen_data': False}
    data_type_dir = 'features_' + data_opts['crop_type'] + '_' + data_opts['crop_mode']
    print(f'[build] cache folder: {data_type_dir}/{FEATURE_CACHE_NAME}')

    imdb = PIE(data_path=args.pie_path)
    train, val, test = build_split_sequences(imdb, dict(data_opts))
    index = {s: build_cache_index(args.pie_path, s, FEATURE_CACHE_NAME,
                                  {'features_context_pad_resize', data_type_dir})
             for s in ('train', 'val', 'test')}

    todo = {'train': {}, 'val': {}, 'test': {}}
    for mode in args.modes:
        for vmode in (args.val_augment_modes if mode == 'augmentation' else [None]):
            tr, va = apply_balancing_strategy(imdb, train, val, mode, augment_pie_tracks,
                                              val_augment_mode=vmode or 'undersample_val')
            part = collect_missing({'train': tr, 'val': va, 'test': test}, index, frame_key)
            for s, d in part.items():
                todo[s].update(d)

    total = sum(len(d) for d in todo.values())
    print('\n[build] missing frames per subset:', {s: len(d) for s, d in todo.items()})
    if total == 0:
        print('[build] nothing to compute.')
        return
    for s, d in todo.items():  # fail early if the raw images are not here
        for key, (imp, _b, _p) in list(d.items())[:20]:
            if not os.path.isfile(imp):
                raise SystemExit(f"raw image not found: {imp}. These frames cannot be computed on this machine.")
    if args.dry_run:
        print('[build] dry run, nothing computed.')
        return

    import tensorflow as tf
    from tensorflow.keras.applications import vgg16
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    from pie_intent_experiment import PIEIntentExperiment
    exp = PIEIntentExperiment(framework='tensorflow', architecture='build_missing', run_name='build_missing',
                              run_dir=os.path.join(args.pie_path, '_build_missing_tmp'))
    exp.context_model = vgg16.VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    t0, done = time.time(), 0
    for subset, items in todo.items():
        if not items:
            continue
        save_path = exp.get_path(type_save='data', data_type=data_type_dir,
                                 model_name=FEATURE_CACHE_NAME, data_subset=subset)
        vals = list(items.values())
        for start in range(0, len(vals), args.chunk):
            part = vals[start:start + args.chunk]
            # one single-frame "sequence" per missing frame; the production function crops, flips,
            # applies photometric factors, runs VGG and writes <image>_<ped_id>.pkl
            exp.load_images_and_process([[v[0]] for v in part], [[v[1]] for v in part],
                                        [[v[2]] for v in part], save_path, data_type=subset)
            getattr(exp, '_feat_buffers', {}).clear()
            done += len(part)
            rate = done / max(time.time() - t0, 1e-9)
            print(f'\n[build] {subset} {min(start + args.chunk, len(vals))}/{len(vals)}  '
                  f'total {done}/{total}  {rate:.1f} frames/s  ETA {(total - done) / rate / 60:.0f} min',
                  flush=True)
    print('\n[build] finished. Run check_cache_coverage.py again; it must report [OK].')


if __name__ == '__main__':
    main()
