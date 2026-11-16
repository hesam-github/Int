"""
pie_intent_experiment.py

Thin subclass of pie_intent.PIEIntent that changes:
  1. get_path() -- ONLY where files are written.
  2. test_chunk() -- adds adaptive/recursive sub-chunking to avoid host
     RAM MemoryError, WITHOUT changing any windowing/overlap/model logic.
     Everything else (architecture, loss, optimizer, accuracy/F1
     computation) is inherited unchanged from the reference PIEIntent.
"""

import os
import gc
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from pie_intent import PIEIntent
from pie_intent_helpers import build_generator_dataset,save_visualizations
from pie_data_Hesam import get_tracks
from utils import concat_data

from keras.applications import vgg16  
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
    # test_chunk override: identical logic/metrics to the reference
    # implementation, but the outer 100-track grouping is now adaptive.
    # ------------------------------------------------------------------
    def _process_test_group(self, data_group, train_params, overlap, data_subset='test', verbose_prefix=''):
        tracks, images_chunk, bboxes_chunk, ped_ids_chunk = get_tracks(
            data_group, train_params['data_type'], self._sequence_length, overlap)

        decoder_input = concat_data(tracks, train_params['data_type']['decoder_input_type'])
        output = concat_data(tracks, train_params['data_type']['output_type'])

        if len(decoder_input) == 0:
            decoder_input = np.zeros(shape=np.array(bboxes_chunk).shape)
        else:
            # The same normalization used by train -- inherited from the base class.
            # Goes through the SAME `self.normalize_decoder_bbox` flag as
            # PIEIntent.get_train_val_data/get_test_data (via self._normalize_decoder_bbox,
            # PIEIntent's thin wrapper around utils.normalize_decoder_bbox) so train
            # and eval can never silently drift out of sync on this setting -- see
            # the migration-risk note on _normalize_decoder_bbox in pie_intent.py.
            decoder_input = self._normalize_decoder_bbox(
                decoder_input, train_params['data_type']['decoder_input_type'])
        output = output[:, 0]

        if not hasattr(self, 'context_model') or self.context_model is None:
            self.context_model = vgg16.VGG16(input_shape=(224, 224, 3),
                                            include_top=False, weights='imagenet')
        feature_shape = self.context_model.output_shape[1:]

        # FIX: save_path now uses data_subset (routed from results_suffix in
        # test_chunk) instead of being hardcoded to 'test' -- previously,
        # evaluating the train/val splits silently read/wrote into the test
        # feature-cache folder.
        save_path = self.get_path(type_save='data',
                                data_type='features_context_pad_resize',
                                model_name='vgg16_none',
                                data_subset=data_subset)

        # The same generator used for training -- shuffle=False because the test order
        # must remain deterministic.
        test_dataset = build_generator_dataset(
            self, images_chunk, bboxes_chunk, ped_ids_chunk,
            decoder_input, output, save_path,
            self._sequence_length, feature_shape,
            batch_size=train_params['batch_size'],
            data_type_label=data_subset, shuffle=False)

        test_results_chunk = self.test_model.predict(
                                    test_dataset, verbose=(1 if is_full() else 0))

        return [(output, test_results_chunk, images_chunk, ped_ids_chunk, bboxes_chunk)]
    



    def test_chunk(self, data_test, data_opts='', model_path='', visualize=False,
                results_suffix='', overlap_override=None,
                vis_num_correct=10, vis_num_incorrect=10, vis_seed=42):

        phase(
            f"Final evaluation on the '{results_suffix.lstrip('_') or 'test'}' split",
            "Reloads the BEST saved checkpoint (not necessarily the last "
            "training epoch) and runs it once over every window of this "
            "split, at overlap=1 (the paper's fixed evaluation protocol) "
            "-- independent of the lighter per-epoch metrics printed "
            "during training, which used a different overlap and the "
            "in-progress (not necessarily best) weights."
        )

        from keras.applications import vgg16
        from keras.models import load_model

        with open(os.path.join(model_path, 'configs.pkl'), 'rb') as fid:
            try:
                configs = pickle.load(fid)
            except Exception:
                configs = pickle.load(fid, encoding='bytes')
        train_params = configs[1]
        self.load_model_config(configs[0])

        self.context_model = vgg16.VGG16(input_shape=(224, 224, 3),
                                        include_top=False,
                                        weights='imagenet')
        try:
            self.test_model = load_model(os.path.join(model_path, 'model.h5'), compile=False)
        except Exception:
            self.test_model = self.get_model(train_params['model'])
            self.test_model.load_weights(os.path.join(model_path, 'model.h5'))

        if is_full():
            self.test_model.summary()
        else:
            print(f"    Model loaded: {self.test_model.count_params():,} params")

        overlap = overlap_override if overlap_override is not None else 1
        vprint(f"[test_chunk] using overlap={overlap} "
            f"({'override' if overlap_override is not None else 'hardcoded to 1, matches reference paper'})")


        # data_subset routes the VGG feature cache to the correct subfolder
        # (_process_test_group), derived from results_suffix e.g. '_train' -> 'train'.
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
                'intention_binary': data_test['intention_binary'][i:min(i + OUTER_GROUP_SIZE, num_samples)],
                'image':            data_test['image'][i:min(i + OUTER_GROUP_SIZE, num_samples)],
                'ped_id':           data_test['ped_id'][i:min(i + OUTER_GROUP_SIZE, num_samples)],
                'intention_prob':   data_test['intention_prob'][i:min(i + OUTER_GROUP_SIZE, num_samples)],
                'bbox':             data_test['bbox'][i:min(i + OUTER_GROUP_SIZE, num_samples)],
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

        acc = accuracy_score(test_target_data, np.round(test_results))
        f1 = f1_score(test_target_data, np.round(test_results))

        save_results_path = os.path.join(model_path, f'ped_intents{results_suffix}.pkl')
        results = {'ped_id': ped_ids,
                'images': images,
                'results': test_results,
                'gt': test_target_data}
        with open(save_results_path, 'wb') as fid:
            pickle.dump(results, fid, pickle.HIGHEST_PROTOCOL)
        print(f'Wrote predictions to {save_results_path}')

        # VISUALIZATION: reuse the inherited PIEIntent._save_visualizations,
        # since this subclass overrides test_chunk entirely but the drawing
        # logic itself doesn't need to change.
        if visualize:
            save_visualizations(vis_results,
                                    model_path=model_path,
                                    results_suffix=results_suffix,
                                    num_correct=vis_num_correct,
                                    num_incorrect=vis_num_incorrect,
                                    seed=vis_seed)

        return acc, f1
