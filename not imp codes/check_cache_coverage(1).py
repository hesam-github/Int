"""
check_cache_coverage.py -- run LOCALLY (pie_tf env, the machine that still has the images),
BEFORE compress_features.py pack.

Colab has no raw images (~1 TB), so every VGG feature a run needs must already be in the per-frame
cache: data/data/pie/<subset>/<data_type>/<cache_name>/<set>/<video>/<image>_<ped_id>.pkl
This script rebuilds, WITHOUT any image access, the exact (frame, ped_id) list of every window that
training and the final evaluation will request, for each balance mode, and counts what is missing.
A non-zero "missing" for a configuration means that configuration will fail on Colab.

Windows requested (as in train_pipeline.py):
  training/per-epoch metrics : train, val, test at seq_overlap_rate (default 0.5)
  final evaluation           : train, val, test at overlap 1 and at overlap 0.5
Augmentation modes need the synthetic ped_ids of augment_pie_tracks (seed 42, current
AUGMENTATION_CFG and schema version); features computed under another schema do not count.

Usage:
  python check_cache_coverage.py --pie-path "D:\\PIE codes\\PIE_dataset" ^
      --modes none augmentation --val-augment-modes undersample_val
  python check_cache_coverage.py ... --data-opts-json my_data_opts.json   # your real DATA_OPTS
Exit code 1 when anything is missing.
"""
import argparse
import json
import os
import sys

DEFAULT_DATA_OPTS = {
    'seq_type': 'intention', 'fstride': 1, 'height_rng': [0, float('inf')], 'squarify_ratio': 0,
    'min_track_size': 15, 'seq_overlap_rate': 0.5, 'max_size_observe': 15,
    'crop_type': 'context', 'crop_mode': 'pad_resize',
    'encoder_input_type': [], 'decoder_input_type': ['bbox'], 'output_type': ['intention_binary'],
}


def frame_key(image_path, ped_id):
    """'<set>/<video>/<image>_<ped_id>.pkl', the path layout of load_images_and_process
    (handles both Windows and POSIX separators)."""
    parts = image_path.replace('\\', '/').split('/')
    stem = os.path.splitext(parts[-1])[0]
    return f'{parts[-3]}/{parts[-2]}/{stem}_{ped_id}.pkl'


def build_cache_index(pie_path, subset, cache_name, data_types):
    """Set of '<set>/<video>/<file>.pkl' present for one subset, ONLY inside
    <data_type>/<cache_name>/ (other cache folders are other preprocessing variants and do not count)."""
    base = os.path.join(pie_path, 'data', 'data', 'pie', subset)
    present = set()
    for root, _dirs, files in os.walk(base):
        rel = os.path.relpath(root, base).replace(os.sep, '/').split('/')
        if len(rel) < 4:  # <data_type>/<cache_name>/<set>/<video>
            continue
        if rel[1] != cache_name or rel[0] not in data_types:
            continue
        sid_vid = '/'.join(rel[-2:])
        present.update(f'{sid_vid}/{f}' for f in files if f.endswith('.pkl'))
    return present


def required_keys(images, ped_ids):
    """Unique cache keys of all windows. images/ped_ids as returned by get_tracks."""
    keys = set()
    for seq, pid in zip(images, ped_ids):
        for imp, p in zip(seq, pid):
            keys.add(frame_key(imp, p[0]))
    return keys


def check_split(seq, data_type, overlap, present, seq_len):
    from pie_data_Hesam import get_tracks
    _t, images, _b, ped_ids = get_tracks(seq, data_type, seq_len, overlap)
    need = required_keys(images, ped_ids)
    missing = sorted(need - present)
    return len(images), len(need), missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pie-path', required=True)
    ap.add_argument('--code-dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--modes', nargs='+', default=['none'],
                    choices=['none', 'undersample_base_article', 'augmentation'])
    ap.add_argument('--val-augment-modes', nargs='+', default=['undersample_val'],
                    choices=['undersample_val', 'augment_val', 'natural_val'])
    ap.add_argument('--split', default='default', choices=['default', 'random'])
    ap.add_argument('--ratios', default='0.5,0.4,0.1', help="random split: 'train,test,val'")
    ap.add_argument('--data-opts-json', default=None, help='JSON file with your DATA_OPTS')
    ap.add_argument('--cache-name', default='vgg16_none_prep1',
                    help='must equal FEATURE_CACHE_NAME in pie_intent.py (vgg16_none_<PREP_TAG>)')
    ap.add_argument('--report', default='coverage_missing.txt')
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(args.code_dir, 'pie_data_Hesam.py')):
        raise SystemExit(f"pie_data_Hesam.py not found in --code-dir '{args.code_dir}'")
    if args.data_opts_json and not os.path.isfile(args.data_opts_json):
        raise SystemExit(f'--data-opts-json file not found: {os.path.abspath(args.data_opts_json)}')

    os.environ['PIE_PATH'] = args.pie_path
    sys.path.insert(0, args.code_dir)
    from pie_data_Hesam import PIE
    from augmentation import augment_pie_tracks
    from train_helpers import build_split_sequences, apply_balancing_strategy

    data_opts = dict(DEFAULT_DATA_OPTS)
    if args.data_opts_json:
        with open(args.data_opts_json, encoding='utf-8') as fh:
            data_opts.update(json.load(fh))
    data_opts['data_split_type'] = args.split
    if args.split == 'random':
        data_opts['random_params'] = {'ratios': [float(x) for x in args.ratios.split(',')],
                                      'val_data': True, 'regen_data': False}
    # Coverage does not depend on the decoder features (pose/velocity/scene), so use the minimal keys
    # that exist in a freshly built sequence dict.
    data_type = {'encoder_input_type': [], 'decoder_input_type': ['bbox'], 'output_type': ['intention_binary']}
    data_types = {'features_context_pad_resize',
                  'features_' + data_opts['crop_type'] + '_' + data_opts['crop_mode']}
    seq_len = data_opts['max_size_observe']
    train_overlap = data_opts['seq_overlap_rate']

    imdb = PIE(data_path=args.pie_path)
    train, val, test = build_split_sequences(imdb, dict(data_opts))
    present = {s: build_cache_index(args.pie_path, s, args.cache_name, data_types)
               for s in ('train', 'val', 'test')}
    print('cached frames on disk:', {s: len(p) for s, p in present.items()})

    combos = []
    for mode in args.modes:
        for vmode in (args.val_augment_modes if mode == 'augmentation' else [None]):
            combos.append((mode, vmode))

    total_missing, lines = 0, []
    print(f"\n{'balance / val mode':<42}{'split':<7}{'overlap':>8}{'windows':>10}{'frames':>9}{'missing':>9}")
    for mode, vmode in combos:
        tr, va = apply_balancing_strategy(imdb, train, val, mode, augment_pie_tracks,
                                          val_augment_mode=vmode or 'undersample_val')
        label = mode if vmode is None else f'{mode} / {vmode}'
        for split_name, seq in (('train', tr), ('val', va), ('test', test)):
            overlaps = sorted({train_overlap, 1, 0.5})
            for ov in overlaps:
                n_win, n_frames, missing = check_split(seq, data_type, ov, present[split_name], seq_len)
                total_missing += len(missing)
                print(f'{label:<42}{split_name:<7}{ov:>8}{n_win:>10}{n_frames:>9}{len(missing):>9}')
                lines += [f'{label}\t{split_name}\t{ov}\t{k}' for k in missing[:2000]]

    with open(args.report, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    if total_missing:
        print(f'\n[FAIL] {total_missing} missing frame feature(s); examples in {args.report}.')
        print('Run that configuration once locally (it will compute the missing features from the images), '
              'then pack again. A configuration with missing > 0 cannot run on Colab.')
        sys.exit(1)
    print('\n[OK] every requested feature exists in the cache for all listed configurations.')


if __name__ == '__main__':
    main()