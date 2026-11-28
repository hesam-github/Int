"""
pie_intent_experiment.py

Thin subclass of pie_intent.PIEIntent that changes:
  1. get_path() -- ONLY where files are written.
  2. test_chunk() -- adds adaptive/recursive sub-chunking to avoid host
     RAM MemoryError, WITHOUT changing any windowing/overlap/model logic.
  3. use_visual=False: VGG16 is never loaded, and the test input is the
     decoder matrix alone.
"""

import os
import gc
import pickle
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from keras.applications import vgg16

from pie_intent import PIEIntent, FEATURE_CACHE_NAME, decoder_only_dataset
from pie_intent_helpers import build_generator_dataset, save_visualizations
from pie_data_Hesam import get_tracks
from utils import concat_data
from log_mode import phase, vprint, is_full


class PIEIntentExperiment(PIEIntent):

    def __init__(self, *args, framework, architecture, run_name, run_dir,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.framework = framework
        self.architecture = architecture
        self.run_name = run_name
        self.run_dir = run_dir

    def get_path(self,
                 type_save='models',
                 models_save_folder='',
                 model_name='convlstm_encdec',
                 file_name='',
                 data_subset='',
                 data_type='',
                 save_root_folder=None):
        assert type_save in ['models', 'data']

        if type_save == 'models':
            os.makedirs(self.run_dir, exist_ok=True)
            return os.path.join(self.run_dir, file_name), self.run_dir

        if save_root_folder is None:
            save_root_folder = os.environ['PIE_PATH'] + '/data/'
        root = os.path.join(save_root_folder, type_save)
        save_path = os.path.join(root, 'pie', data_subset, data_type, model_name)
        os.makedirs(save_path, exist_ok=True)
        return save_path

    def __repr__(self):
        return (f"PIEIntentExperiment(framework={self.framework!r}, "
                f"architecture={self.architecture!r}, run_name={self.run_name!r}, "
                f"run_dir={self.run_dir!r})")

    # ------------------------------------------------------------------
    def _process_test_group(self, data_group, train_params, overlap, data_subset='test', verbose_prefix=''):
        tracks, images_chunk, bboxes_chunk, ped_ids_chunk = get_tracks(
            data_group, train_params['data_type'], self._sequence_length, overlap)

        decoder_input = concat_data(tracks, train_params['data_type']['decoder_input_type'])
        output = concat_data(tracks, train_params['data_type']['output_type'])

        if len(decoder_input) == 0:
            decoder_input = np.zeros(shape=np.array(bboxes_chunk).shape)
        else:
            decoder_input = self._normalize_decoder_bbox(
                decoder_input, train_params['data_type']['decoder_input_type'])
            from feature_routing import apply_feature_norms
            decoder_input = apply_feature_norms(self, decoder_input, self.feature_spans, fit=False)
        output = output[:, 0]

        flow = None
        if getattr(self, 'flow_representation', 'summary') == 'dense':
            import hashlib
            from flow_dense import ensure_flow_volume
            if not getattr(self, 'flow_root', None):
                raise RuntimeError(
                    'flow_representation is dense but flow_root was not restored '
                    'from hyperparams.json.')
            stamp = hashlib.md5(
                repr((images_chunk[0][0], images_chunk[-1][-1], len(images_chunk))).encode()
            ).hexdigest()[:12]
            cache = os.path.join(
                self.flow_root, 'summary', 'dense32',
                f'{data_subset}_{stamp}_{self._sequence_length}_{overlap}.npy')
            flow = ensure_flow_volume(
                images_chunk, ped_ids_chunk, self.flow_root, cache)
            print(f'[flow dense] eval windows={len(images_chunk)} '
                  f'overlap={overlap} subset={data_subset}')

        if getattr(self, 'use_visual', True):
            if not hasattr(self, 'context_model') or self.context_model is None:
                self.context_model = vgg16.VGG16(input_shape=(224, 224, 3),
                                                 include_top=False, weights='imagenet')
            feature_shape = self.context_model.output_shape[1:]
            save_path = self.get_path(type_save='data',
                                      data_type='features_context_pad_resize',
                                      model_name=FEATURE_CACHE_NAME,
                                      data_subset=data_subset)
            test_dataset = build_generator_dataset(
                self, images_chunk, bboxes_chunk, ped_ids_chunk,
                decoder_input, output, save_path,
                self._sequence_length, feature_shape,
                batch_size=train_params['batch_size'],
                data_type_label=data_subset, shuffle=False,
                flow=flow)
        else:
            if flow is not None:
                raise RuntimeError(
                    'dense flow is a third model input. decoder-only evaluation '
                    'cannot feed it.')
            test_dataset = decoder_only_dataset(
                decoder_input, output, train_params['batch_size'],
                shuffle=False, repeat=False)

        test_results_chunk = self.test_model.predict(
            test_dataset, verbose=(1 if is_full() else 0))

        return [(output, test_results_chunk, images_chunk, ped_ids_chunk, bboxes_chunk)]

    # ------------------------------------------------------------------
    def test_chunk(self, data_test, data_opts='', model_path='', visualize=False,
                   checkpoint_dir=None, results_suffix='', overlap_override=None,
                   vis_num_correct=10, vis_num_incorrect=10, vis_seed=42,
                   chunk_size=10, feature_subset=None):

        phase(
            f"Final evaluation on the '{results_suffix.lstrip('_') or 'test'}' split",
            "Reloads the BEST saved checkpoint (not necessarily the last "
            "training epoch) and runs it once over every window of this "
            "split. Default overlap is 1. Pass overlap_override=0.5 to match "
            "the training windows. Neither sentence is the paper's protocol "
            "by itself. "
            "-- independent of the lighter per-epoch metrics printed "
            "during training, which used a different overlap and the "
            "in-progress (not necessarily best) weights."
        )

        from keras.models import load_model

        with open(os.path.join(model_path, 'configs.pkl'), 'rb') as fid:
            try:
                configs = pickle.load(fid)
            except Exception:
                configs = pickle.load(fid, encoding='bytes')
        train_params = configs[1]
        hp_path = os.path.join(model_path, 'hyperparams.json')
        saved_hp = {}
        if os.path.isfile(hp_path):
            with open(hp_path, encoding='utf-8') as f:
                saved_hp = json.load(f)
        # Evaluate with the SAME bbox representation the checkpoint was trained with.
        self.normalize_decoder_bbox = bool(saved_hp.get('normalize_decoder_bbox', False))
        from feature_routing import restore_routing, CUSTOM_OBJECTS

        restore_routing(self, saved_hp)            # sets use_visual, fusion, ...
        self.temporal_encoder = saved_hp.get('temporal_encoder')
        self.temporal_hp = saved_hp.get('temporal_hp') or {}
        self.load_model_config(configs[0])

        if getattr(self, 'use_visual', True):
            self.context_model = vgg16.VGG16(input_shape=(224, 224, 3),
                                             include_top=False,
                                             weights='imagenet')
        else:
            print("[visual] use_visual=False: VGG16 not loaded")

        try:
            self.test_model = load_model(os.path.join(model_path, 'model.h5'),
                                         compile=False, custom_objects=CUSTOM_OBJECTS)
        except Exception:
            self.test_model = self.get_model(train_params['model'])
            self.test_model.load_weights(os.path.join(model_path, 'model.h5'))

        if is_full():
            self.test_model.summary()
        else:
            print(f"    Model loaded: {self.test_model.count_params():,} params")

        overlap = overlap_override if overlap_override is not None else 1
        vprint(f"[test_chunk] using overlap={overlap} "
               f"({'override' if overlap_override is not None else 'default 1, not the training overlap'})")

        # data_subset routes the VGG feature cache to the correct subfolder
        # (_process_test_group), derived from results_suffix e.g. '_train' -> 'train'.
        if feature_subset:
            data_subset = feature_subset
        else:
            data_subset = results_suffix.lstrip('_') if results_suffix else 'test'

        test_target_data = []
        test_results = []
        ped_ids = []
        images = []
        bboxes = []
        vis_results = []

        num_samples = len(data_test['image'])
        OUTER_GROUP_SIZE = 100

        for i in range(0, num_samples, OUTER_GROUP_SIZE):
            data_test_chunk = {
                key: value[i:min(i + OUTER_GROUP_SIZE, num_samples)]
                for key, value in data_test.items()
                if isinstance(value, list) and len(value) == num_samples
            }

            sub_results = self._process_test_group(
                data_test_chunk, train_params, overlap, data_subset=data_subset,
                verbose_prefix=f"[tracks {i}-{min(i + OUTER_GROUP_SIZE, num_samples)}] ")

            for (test_target_data_chunk, test_results_chunk,
                 images_chunk, ped_ids_chunk, bboxes_chunk) in sub_results:

                test_target_data.extend(test_target_data_chunk)
                test_results.extend(test_results_chunk)
                images.extend(images_chunk)
                ped_ids.extend(ped_ids_chunk)
                bboxes.extend(bboxes_chunk)

                for j, (imp, box, ped) in enumerate(zip(images_chunk, bboxes_chunk, ped_ids_chunk)):
                    vis_results.append({'imp': imp[-1],
                                        'bbox': box[-1],
                                        'ped_id': ped[-1][0],
                                        'res': test_results_chunk[j][0],
                                        'target': test_target_data_chunk[j]})

            del data_test_chunk, sub_results
            gc.collect()

        y_true = np.asarray(test_target_data)
        y_prob = np.asarray(test_results)
        if y_true.ndim == 2 and y_true.shape[1] > 1:
            true_id = y_true.argmax(axis=1)
            pred_id = y_prob.argmax(axis=1)
            acc = accuracy_score(true_id, pred_id)
            f1 = f1_score(true_id, pred_id, average='macro', zero_division=0)
        else:
            acc = accuracy_score(y_true, np.round(y_prob))
            f1 = f1_score(y_true, np.round(y_prob))

        save_results_path = os.path.join(model_path, f'ped_intents{results_suffix}.pkl')
        results = {'ped_id': ped_ids,
                   'images': images,
                   'results': test_results,
                   'gt': test_target_data}
        with open(save_results_path, 'wb') as fid:
            pickle.dump(results, fid, pickle.HIGHEST_PROTOCOL)
        print(f'Wrote predictions to {save_results_path}')

        if visualize:
            save_visualizations(vis_results,
                                model_path=model_path,
                                results_suffix=results_suffix,
                                num_correct=vis_num_correct,
                                num_incorrect=vis_num_incorrect,
                                seed=vis_seed)

        return acc, f1