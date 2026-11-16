"""
pie_intent.py

PIEIntent: the ConvLSTM encoder/decoder model for pedestrian intention
estimation, and the train()/test_chunk() pipeline. Standalone
infrastructure (progress printing, metrics, trace/GPU logging, the
streaming dataset builder, visualization) lives in pie_intent_helpers.py.

Data-prep helpers that used to live here as no-op-on-self methods
(get_tracks, concat_data, _normalize_decoder_bbox) have been moved out
after the method-by-method audit: get_tracks -> pie_data_Hesam.py
(module-level function, next to the class that produces the sequence
dict it windows), concat_data and normalize_decoder_bbox -> utils.py
(generic array utilities). See pieintent_method_audit.md for the full
reasoning and the risks considered before moving each one.
"""
import tensorflow as tf
import numpy as np
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import gc

from keras import backend as K
from keras import regularizers
from keras.applications import vgg16
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Concatenate
from keras.layers import ConvLSTM2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import RepeatVector
from keras.layers import LSTM
from keras.models import Model
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.utils import img_to_array
from keras.utils import load_img
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from utils import *

import json

from PIL import Image, ImageEnhance
from augmentation import is_flip_augmented, get_photometric_factors

from log_mode import phase, vprint, is_full

from pie_intent_helpers import (
    _safe_update_progress,
    compute_full_metrics,
    CACHE_READ_WORKERS,
    trace_log,
    start_gpu_watchdog,
    stop_gpu_watchdog,
    build_generator_dataset,
    save_visualizations,
)

from pie_data_Hesam import get_tracks

def set_global_seed(seed):
    """Seeds python, numpy and TensorFlow (weight init + dropout masks).
    Must be called BEFORE the model is built."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)  # TF>=2.7: python+numpy+tf
    except AttributeError:
        tf.random.set_seed(seed)

class PIEIntent(object):
    """ConvLSTM encoder/decoder model for pedestrian intention prediction."""

    def __init__(self,
                 num_hidden_units=128,
                 regularizer_val=0.001,
                 activation='tanh',
                 lstm_dropout=0.4,
                 lstm_recurrent_dropout=0.2,
                 convlstm_num_filters=64,
                 convlstm_kernel_size=2,
                 crop_squarify_ratio=1):
        self._num_hidden_units = num_hidden_units
        self.reg_value = regularizer_val
        self._kernel_regularizer = regularizers.l2(regularizer_val)
        self._recurrent_regularizer = regularizers.l2(regularizer_val)
        self._bias_regularizer = regularizers.l2(regularizer_val)
        self._activation = activation

        self._lstm_dropout = lstm_dropout
        self._lstm_recurrent_dropout = lstm_recurrent_dropout

        self._convlstm_num_filters = convlstm_num_filters
        self._convlstm_kernel_size = convlstm_kernel_size

        self._encoder_input_size = 4
        self._decoder_dense_output_size = 1
        self._decoder_input_size = 4

        self.crop_squarify_ratio = crop_squarify_ratio
        self._model_name = 'convlstm_encdec'

    def get_path(self,
                 type_save='models',
                 models_save_folder='',
                 model_name='convlstm_encdec',
                 file_name='',
                 data_subset='',
                 data_type='',
                 save_root_folder=os.environ['PIE_PATH'] + '/data/'):
        """Builds and creates the save path for model or feature-cache data.

        NOTE (kept intentionally as an instance method, NOT moved to
        utils.py despite using no `self.*` attribute internally): this
        method is fully overridden in PIEIntentExperiment to redirect
        model-saving to a per-run directory (self.run_dir). train() and
        test_chunk() below call this polymorphically via `self.get_path`,
        so PIEIntentExperiment can change routing without touching those
        methods at all. Making this a plain module-level function would
        silently break that override -- see pieintent_method_audit.md.
        """
        assert(type_save in ['models', 'data'])
        if data_type != '':
            assert(any([d in data_type for d in ['images', 'features']]))
        root = os.path.join(save_root_folder, type_save)

        if type_save == 'models':
            save_path = os.path.join(save_root_folder, 'pie', 'intention', models_save_folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return os.path.join(save_path, file_name), save_path
        else:
            save_path = os.path.join(root, 'pie', data_subset, data_type, model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return save_path

    def get_model_config(self):
        """Returns the model's architecture hyperparameters as a dict."""
        config = dict()
        config['num_hidden'] = self._num_hidden_units
        config['reg_value'] = self.reg_value
        config['activation'] = self._activation
        config['encoder_input_size'] = self._encoder_input_size
        config['encoder_seq_length'] = self._encoder_seq_length
        config['sequence_length'] = self._sequence_length
        config['lstm_dropout'] = self._lstm_dropout
        config['lstm_recurrent_dropout'] = self._lstm_recurrent_dropout
        config['convlstm_num_filters'] = self._convlstm_num_filters
        config['convlstm_kernel_size'] = self._convlstm_kernel_size
        config['decoder_input_size'] = self._decoder_input_size
        config['decoder_dense_output_size'] = self._decoder_dense_output_size
        config['decoder_seq_length'] = self._decoder_seq_length

        if is_full():
            print(config)
        return config

    def load_model_config(self, config):
        """Restores architecture hyperparameters from a saved config dict."""
        self._num_hidden_units = config['num_hidden']
        self.reg_value = config['reg_value']
        self._activation = config['activation']
        self._encoder_seq_length = config['encoder_seq_length']
        self._sequence_length = config['sequence_length']
        self._lstm_dropout = config['lstm_dropout']
        self._lstm_recurrent_dropout = config['lstm_recurrent_dropout']
        self._convlstm_num_filters = config['convlstm_num_filters']
        self._convlstm_kernel_size = config['convlstm_kernel_size']
        self._encoder_input_size = config['encoder_input_size']
        self._decoder_input_size = config['decoder_input_size']
        self._decoder_dense_output_size = config['decoder_dense_output_size']
        self._decoder_seq_length = config['decoder_seq_length']

    def load_images_and_process(self,
                                    img_sequences,
                                    bbox_sequences,
                                    ped_ids,
                                    save_path,
                                    data_type='train',
                                    regen_pkl=False,
                                    debug_every=None):
            """
            Loads or computes the VGG16 feature for every frame in
            img_sequences: cache hits are read in parallel from disk,
            cache misses are cropped/augmented/resized on the calling
            thread and passed through VGG16 in batches of
            VGG_FORWARD_BATCH_SIZE, then cached.
            Returns an array of shape (n_sequences, seq_length, *feature_shape).
            """
            try:
                convnet = self.context_model
            except:
                raise Exception("No context model is defined")
            if getattr(self, '_cache_read_executor', None) is None:
                self._cache_read_executor = ThreadPoolExecutor(max_workers=CACHE_READ_WORKERS)

            if debug_every is not None:
                import tracemalloc, gc, psutil
                process = psutil.Process(os.environ.get('PID', os.getpid()))
                tracemalloc.start(10)
                debug_snapshots = []
                debug_img_count = [0]

                def _debug_snap(label):
                    gc.collect()
                    rss = process.memory_info().rss / 1024**2
                    snap = tracemalloc.take_snapshot()
                    debug_snapshots.append((label, snap))
                    print(f"\n[DEBUG] {label}: RSS = {rss:.1f} MB")
                    if len(debug_snapshots) >= 2:
                        prev_label, prev_snap = debug_snapshots[-2]
                        stats = snap.compare_to(prev_snap, 'lineno')
                        print(f"  Top allocation growth since '{prev_label}':")
                        for stat in stats[:5]:
                            print("   ", stat)

                _debug_snap("before loop")

            n_seqs = len(img_sequences)
            features_grid = [[None] * len(img_sequences[i]) for i in range(n_seqs)]
            hit_specs = []
            miss_specs = []

            for i, (seq, pid) in enumerate(zip(img_sequences, ped_ids)):
                for j, (imp, b, p) in enumerate(zip(seq, bbox_sequences[i], pid)):
                    parts = os.path.normpath(imp).split(os.sep)
                    set_id = parts[-3]
                    vid_id = parts[-2]
                    img_name = os.path.splitext(parts[-1])[0]
                    img_save_folder = os.path.join(save_path, set_id, vid_id)
                    img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')

                    if os.path.exists(img_save_path) and not regen_pkl:
                        hit_specs.append((i, j, img_save_path))
                    else:
                        miss_specs.append((i, j, imp, b, img_save_path, p[0]))

            total_frames = len(hit_specs) + len(miss_specs)
            trace_log(f"load_images_and_process[{data_type}]: hits={len(hit_specs)} "
                    f"misses={len(miss_specs)} (total={total_frames})")

            def _read_cached_feature(path):
                with open(path, 'rb') as fid:
                    try:
                        return pickle.load(fid)
                    except Exception:
                        fid.seek(0)
                        return pickle.load(fid, encoding='bytes')

            if hit_specs:
                progress_step = max(1, len(hit_specs) // 20)
                done_count = 0
                executor = self._cache_read_executor
                future_to_ij = {
                    executor.submit(_read_cached_feature, path): (i, j)
                    for (i, j, path) in hit_specs
                }
                for future in future_to_ij:
                    i, j = future_to_ij[future]
                    features_grid[i][j] = future.result()
                    done_count += 1
                    if done_count % progress_step == 0:
                        _safe_update_progress(done_count / len(hit_specs))
                _safe_update_progress(1.0)

            VGG_FORWARD_BATCH_SIZE = 32

            for batch_start in range(0, len(miss_specs), VGG_FORWARD_BATCH_SIZE):
                batch = miss_specs[batch_start: batch_start + VGG_FORWARD_BATCH_SIZE]

                preprocessed_batch = []
                batch_meta = []  # (i, j, img_save_path), same order as preprocessed_batch

                for (i, j, imp, b, img_save_path, ped_id_str) in batch:
                    img_data = load_img(imp)
                    bbox = jitter_bbox([list(b)], imp, 'enlarge', 2)[0]
                    bbox = squarify(bbox, self.crop_squarify_ratio, img_data.size[0])
                    bbox = list(map(int, bbox[0:4]))
                    cropped_image = img_data.crop(bbox)

                    if is_flip_augmented(ped_id_str):
                        cropped_image = cropped_image.transpose(Image.FLIP_LEFT_RIGHT)

                    b_factor, c_factor = get_photometric_factors(ped_id_str)
                    if b_factor != 1.0:
                        cropped_image = ImageEnhance.Brightness(cropped_image).enhance(b_factor)
                    if c_factor != 1.0:
                        cropped_image = ImageEnhance.Contrast(cropped_image).enhance(c_factor)

                    img_data = img_pad(cropped_image, mode='pad_resize', size=224)
                    image_array = img_to_array(img_data)
                    preprocessed_img = vgg16.preprocess_input(image_array)
                    preprocessed_batch.append(preprocessed_img)
                    batch_meta.append((i, j, img_save_path))

                # Single batched forward pass through VGG16 for this chunk.
                stacked_batch = np.stack(preprocessed_batch, axis=0)
                batch_features = convnet.predict_on_batch(stacked_batch)

                for k, (i, j, img_save_path) in enumerate(batch_meta):
                    img_features = batch_features[k:k + 1]
                    img_save_folder = os.path.dirname(img_save_path)
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder)
                    with open(img_save_path, 'wb') as fid:
                        pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)
                    features_grid[i][j] = img_features

                    if debug_every is not None:
                        debug_img_count[0] += 1
                        if debug_img_count[0] % debug_every == 0:
                            _debug_snap(f"after {debug_img_count[0]} cache-miss images")

                del stacked_batch, batch_features, preprocessed_batch

            feat_shape = tuple(int(d) for d in convnet.output_shape[1:])
            seq_len = len(img_sequences[0]) if n_seqs > 0 else 0
            buf_key = (n_seqs, seq_len) + feat_shape

            if not hasattr(self, '_feat_buffers'):
                self._feat_buffers = {}
            sequences = self._feat_buffers.get(buf_key)
            if sequences is None:
                sequences = np.empty((n_seqs, seq_len) + feat_shape, dtype=np.float32)
                self._feat_buffers[buf_key] = sequences

            for i in range(n_seqs):
                for j in range(seq_len):
                    sequences[i, j] = np.squeeze(features_grid[i][j])

            if debug_every is not None:
                _debug_snap("end of function")

            return sequences

    def _normalize_decoder_bbox(self, decoder_input, decoder_input_type):
        """Thin instance-level wrapper around utils.normalize_decoder_bbox.

        Kept as a method (rather than calling utils.normalize_decoder_bbox
        directly at every call site) so the `self.normalize_decoder_bbox`
        config flag is read in exactly ONE place. This directly addresses
        the migration risk flagged in the audit: PIEIntentExperiment
        inherits this method unchanged and calls it via `self.` from
        _process_test_group, so as long as that call goes through this
        wrapper (not a bare `normalize_decoder_bbox(...)` call), train
        and eval paths cannot silently drift out of sync on this flag.
        """
        return normalize_decoder_bbox(
            decoder_input, decoder_input_type,
            enabled=getattr(self, 'normalize_decoder_bbox', True))

    def get_train_val_data(self, data, data_type, seq_length, overlap):
        """Builds windowed encoder/decoder/output arrays for train/val
        from raw sequence data, and initializes the VGG16 context model."""
        tracks, images, bboxes, ped_ids = get_tracks(data, data_type, seq_length, overlap)

        encoder_input = concat_data(tracks, data_type['encoder_input_type'])
        decoder_input = concat_data(tracks, data_type['decoder_input_type'])
        output = concat_data(tracks, data_type['output_type'])

        if len(decoder_input) == 0:
            decoder_input = np.zeros(shape=np.array(bboxes).shape)
        else:
            decoder_input = self._normalize_decoder_bbox(decoder_input, data_type['decoder_input_type'])

        self.context_model = vgg16.VGG16(input_shape=(224, 224, 3),
                                          include_top=False,
                                          weights='imagenet')

        return {'images': images,
                'bboxes': bboxes,
                'ped_ids': ped_ids,
                'encoder_input': encoder_input,
                'decoder_input': decoder_input,
                'output': output}

    def get_test_data(self, data, train_params, seq_length, overlap, data_subset='test'):
        """Builds windowed test data and runs load_images_and_process to
        produce the encoder feature array for a given real split
        ('train'/'val'/'test', via data_subset)."""
        tracks, images, bboxes, ped_ids = get_tracks(data,
                                                    train_params['data_type'],
                                                    seq_length,
                                                    overlap)

        encoder_input = concat_data(tracks, train_params['data_type']['encoder_input_type'])
        decoder_input = concat_data(tracks, train_params['data_type']['decoder_input_type'])
        output = concat_data(tracks, train_params['data_type']['output_type'])

        if len(decoder_input) == 0:
            decoder_input = np.zeros(shape=np.array(bboxes).shape)
        else:
            decoder_input = self._normalize_decoder_bbox(
                decoder_input, train_params['data_type']['decoder_input_type'])

        if not hasattr(self, 'context_model') or self.context_model is None:
            self.context_model = vgg16.VGG16(input_shape=(224, 224, 3),
                                            include_top=False,
                                            weights='imagenet')

        test_img = self.load_images_and_process(images,
                                                bboxes,
                                                ped_ids,
                                                data_type=data_subset,
                                                save_path=self.get_path(type_save='data',
                                                                        data_type='features_context_pad_resize',
                                                                        model_name='vgg16_none',
                                                                        data_subset=data_subset))
        output = output[:, 0]
        return ([test_img, decoder_input], output)

    def get_model(self, model):
        train_model = self.pie_convlstm_encdec()
        return train_model

    def create_lstm_model(self,
                          name='convlstm_encdec',
                          r_state=True,
                          r_sequence=False):
        """Builds one LSTM layer using this instance's shared
        regularizer/dropout/activation settings."""
        return LSTM(units=self._num_hidden_units,
                    dropout=self._lstm_dropout,
                    recurrent_dropout=self._lstm_recurrent_dropout,
                    return_state=r_state,
                    return_sequences=r_sequence,
                    stateful=False,
                    bias_initializer='zeros',
                    kernel_regularizer=self._kernel_regularizer,
                    recurrent_regularizer=self._recurrent_regularizer,
                    bias_regularizer=self._bias_regularizer,
                    activation=self._activation,
                    name=name)

    def pie_convlstm_encdec(self):
        """Builds the full ConvLSTM encoder + LSTM decoder model:
        ConvLSTM2D on encoder_input -> flatten -> repeat + concat with
        decoder_input -> LSTM -> Dense(1, sigmoid)."""
        encoder_input = Input(shape=(self._sequence_length,) + self.context_model.output_shape[1:],
                                            name="encoder_input")
        interm_input = encoder_input

        encoder_model = ConvLSTM2D(filters=self._convlstm_num_filters,
                                   kernel_size=self._convlstm_kernel_size,
                                   kernel_regularizer=self._kernel_regularizer,
                                   recurrent_regularizer=self._recurrent_regularizer,
                                   bias_regularizer=self._bias_regularizer,
                                   dropout=self._lstm_dropout,
                                   recurrent_dropout=self._lstm_recurrent_dropout,
                                   return_sequences=False)(interm_input)
        encoder_output = Flatten(name='encoder_flatten')(encoder_model)

        decoder_input = Input(shape=(self._decoder_seq_length,
                                     self._decoder_input_size),
                              name='decoder_input')
        encoder_vec = RepeatVector(self._decoder_seq_length)(encoder_output)
        decoder_concat_inputs = Concatenate(axis=2)([encoder_vec, decoder_input])

        decoder_model = self.create_lstm_model(name='decoder_network',
                                               r_state=False,
                                               r_sequence=False)(decoder_concat_inputs)

        decoder_dense_output = Dense(self._decoder_dense_output_size,
                                     activation='sigmoid',
                                     kernel_regularizer=self._kernel_regularizer,
                                     bias_regularizer=self._bias_regularizer,
                                     name='decoder_dense')(decoder_model)

        self.train_model = Model(inputs=[encoder_input, decoder_input],
                                 outputs=decoder_dense_output)

        self.train_model.summary()
        return self.train_model

    def train(self,
             data_train,
             data_val,
             batch_size=128,
             epochs=400,
             data_test=None,
             optimizer_type='rmsprop',
             optimizer_params=None,
             loss=['binary_crossentropy'],
             metrics=['acc'],
             data_opts='',
             monitor_metric='val_auc',
             monitor_mode='max',
             early_stop_min_delta=0.0005,
             early_stop_patience=10,
             seed=None):
            """
            Builds train/val/test tf.data pipelines, compiles the model,
            runs model.fit() with early stopping/checkpointing/plateau LR
            scheduling and a per-epoch full-metrics callback, and returns
            the folder path where the model and configs were saved.
            """
            if optimizer_params is None:
                optimizer_params = {'lr': 0.00001, 'clip_type': 'clipvalue', 'clip_amount': 1.0}
            data_type = {'encoder_input_type': data_opts['encoder_input_type'],
                        'decoder_input_type': data_opts['decoder_input_type'],
                        'output_type': data_opts['output_type']}

            train_config = {'batch_size': batch_size,
                            'epoch': epochs,
                            'seed': seed,
                            'optimizer_type': optimizer_type,
                            'early_stop_min_delta': early_stop_min_delta,
                            'early_stop_patience': early_stop_patience,
                            'optimizer_params': optimizer_params,
                            'loss': [l if isinstance(l, str) else getattr(l, '__name__', repr(l)) for l in loss],
                            'metrics': metrics,
                            'learning_scheduler_mode': 'plateau',
                            'learning_scheduler_params': {'exp_decay_param': 0.3,
                                                        'step_drop_rate': 0.5,
                                                        'epochs_drop_rate': 20.0,
                                                        'plateau_patience': 3,
                                                        'min_lr': 0.0000001,
                                                        'monitor_value': monitor_metric},
                            'model': 'convlstm_encdec',
                            'data_type': data_type,
                            'overlap': data_opts['seq_overlap_rate'],
                            'dataset': 'pie'}
            
            self._model_type = 'convlstm_encdec'
            seq_length = data_opts['max_size_observe']
            train_d = self.get_train_val_data(data_train, data_type, seq_length, data_opts['seq_overlap_rate'])
            val_d = self.get_train_val_data(data_val, data_type, seq_length, data_opts['seq_overlap_rate'])

            self._encoder_seq_length = train_d['decoder_input'].shape[1]
            self._decoder_seq_length = train_d['decoder_input'].shape[1]
            self._sequence_length = self._encoder_seq_length

            self.context_model = vgg16.VGG16(input_shape=(224, 224, 3),
                                            include_top=False,
                                            weights='imagenet')

            feature_shape = self.context_model.output_shape[1:]
            print(f"feature_shape from context_model: {feature_shape}")

            start_gpu_watchdog()
            trace_log("train(): about to build datasets and call fit()")

            train_d['output'] = train_d['output'][:, 0]
            val_d['output'] = val_d['output'][:, 0]

            train_save_path = self.get_path(type_save='data',
                                            data_type='features' + '_' + data_opts['crop_type'] + '_' + data_opts['crop_mode'],
                                            model_name='vgg16_' + 'none',
                                            data_subset='train')
            val_save_path = self.get_path(type_save='data',
                                        data_type='features' + '_' + data_opts['crop_type'] + '_' + data_opts['crop_mode'],
                                        model_name='vgg16_' + 'none',
                                        data_subset='val')

            train_dataset = build_generator_dataset(
                self, train_d['images'], train_d['bboxes'], train_d['ped_ids'],
                train_d['decoder_input'], train_d['output'], train_save_path,
                seq_length, feature_shape, batch_size=batch_size,
                data_type_label='train', shuffle=True)
            train_eval_dataset = build_generator_dataset(
                                    self, train_d['images'], train_d['bboxes'], train_d['ped_ids'],
                                    train_d['decoder_input'], train_d['output'], train_save_path,
                                    seq_length, feature_shape, batch_size=batch_size,
                                    data_type_label='train', shuffle=False)
            val_dataset = build_generator_dataset(
                self, val_d['images'], val_d['bboxes'], val_d['ped_ids'],
                val_d['decoder_input'], val_d['output'], val_save_path,
                seq_length, feature_shape, batch_size=batch_size,
                data_type_label='val', shuffle=False)

            test_dataset = None
            if data_test is not None:
                test_d = self.get_train_val_data(data_test, data_type, seq_length, data_opts['seq_overlap_rate'])
                test_d['output'] = test_d['output'][:, 0]
                test_save_path = self.get_path(type_save='data',
                            data_type='features' + '_' + data_opts['crop_type'] + '_' + data_opts['crop_mode'],
                            model_name='vgg16_' + 'none',
                            data_subset='test')
                test_dataset = build_generator_dataset(
                    self, test_d['images'], test_d['bboxes'], test_d['ped_ids'],
                    test_d['decoder_input'], test_d['output'], test_save_path,
                    seq_length, feature_shape, batch_size=batch_size,
                    data_type_label='test', shuffle=False)

            # Seed right before model construction: this fixes the initial
            # weights and the dropout mask stream. It must come AFTER
            # balance_samples_count (which reseeds numpy globally to 42).
            if seed is not None:
                set_global_seed(seed)
                print(f"[seed] global seed = {seed} (weights/dropout/shuffle)")
            else:
                print("[seed] WARNING: seed=None -> run is NOT reproducible")
            train_model = self.pie_convlstm_encdec()
            rmsprop_kwargs = {'learning_rate': optimizer_params['lr']}
            clip_type = optimizer_params.get('clip_type', 'clipvalue')
            clip_amount = optimizer_params.get('clip_amount')
            if clip_type != 'none' and clip_amount is not None:
                rmsprop_kwargs[clip_type] = clip_amount

            weight_decay = optimizer_params.get('weight_decay')
            if weight_decay:
                rmsprop_kwargs['weight_decay'] = weight_decay

            optimizer = RMSprop(**rmsprop_kwargs)
            vprint(f'[optimizer] RMSprop kwargs = {rmsprop_kwargs}')

            train_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            vprint('TRAINING: loss={} metrics={}'.format(loss, metrics))

            model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")

            model_path, model_folder_path = self.get_path(type_save='models',
                                        model_name='convlstm_encdec',
                                        models_save_folder=model_folder_name,
                                        file_name='model.h5',
                                        save_root_folder=os.environ['PIE_PATH'])

            config_path, _ = self.get_path(type_save='models',
                                        model_name='convlstm_encdec',
                                        models_save_folder=model_folder_name,
                                        file_name='configs',
                                        save_root_folder=os.environ['PIE_PATH'])

            with open(config_path + '.pkl', 'wb') as fid:
                pickle.dump([self.get_model_config(),
                            train_config, data_opts],
                            fid, pickle.HIGHEST_PROTOCOL)
            print('Wrote configs to {}'.format(config_path))

            with open(config_path + '.txt', 'wt') as fid:
                fid.write("####### Data options #######\n")
                fid.write(str(data_opts))
                fid.write("\n####### Model config #######\n")
                fid.write(str(self.get_model_config()))
                fid.write("\n####### Training config #######\n")
                fid.write(str(train_config))

            hyperparams_path = os.path.join(model_folder_path, 'hyperparams.json')
            hyperparams = {
                'model_config': self.get_model_config(),
                'train_config': train_config,
                'data_opts': data_opts,
                'normalize_decoder_bbox': getattr(self, 'normalize_decoder_bbox', True),
            }
            with open(hyperparams_path, 'w', encoding='utf-8') as fid:
                json.dump(hyperparams, fid, indent=2, default=str)
            print('Wrote hyperparams to {}'.format(hyperparams_path))

            early_stop = EarlyStopping(monitor=monitor_metric,
                        mode=monitor_mode,
                        min_delta=early_stop_min_delta,
                        patience=early_stop_patience,
                        verbose=0)
            
            checkpoint = ModelCheckpoint(filepath=model_path,
                                        save_best_only=True,
                                        save_weights_only=False,
                                        monitor=train_config['learning_scheduler_params']['monitor_value'],
                                        mode=monitor_mode)
            plateau_sch = ReduceLROnPlateau(monitor=train_config['learning_scheduler_params']['monitor_value'],
                    mode=monitor_mode,
                    factor=train_config['learning_scheduler_params']['step_drop_rate'],
                    patience=train_config['learning_scheduler_params']['plateau_patience'],
                    min_lr=train_config['learning_scheduler_params']['min_lr'],
                    verbose=0)

            EVAL_EVERY_N_EPOCHS = 5

            class EpochMetricsTable(tf.keras.callbacks.Callback):
                """Computes and logs full train/val/test metrics after
                every epoch, and persists them to epoch_metrics.json."""

                def __init__(self, train_eval_dataset, train_y_true, val_dataset, val_y_true,
                    save_path, test_dataset=None, test_y_true=None,
                    eval_every=EVAL_EVERY_N_EPOCHS, total_epochs=None):
                    super().__init__()
                    self.train_eval_dataset = train_eval_dataset
                    self.train_y_true = train_y_true
                    self.val_dataset = val_dataset
                    self.val_y_true = val_y_true
                    self.test_dataset = test_dataset
                    if test_dataset is not None and test_y_true is None:
                        raise ValueError(
                            "EpochMetricsTable: test_dataset was provided but test_y_true was not."
                        )
                    self.test_y_true = test_y_true
                    self.save_path = save_path
                    self.eval_every = max(1, eval_every)
                    self.total_epochs = total_epochs
                    self.history = []

                def _should_run_full_train_eval(self, epoch):
                    return True

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}

                    val_prob = self.model.predict(self.val_dataset, verbose=0).ravel()
                    val_m = compute_full_metrics(self.val_y_true, val_prob)
                    logs['val_auc'] = val_m['roc_auc'] if not np.isnan(val_m['roc_auc']) else 0.0
                    train_loss = logs.get('loss')
                    val_loss   = logs.get('val_loss')

                    run_full_train = self._should_run_full_train_eval(epoch)

                    train_m = None
                    if run_full_train:
                        train_prob = self.model.predict(self.train_eval_dataset, verbose=0).ravel()
                        train_m = compute_full_metrics(self.train_y_true, train_prob)

                    test_m = None
                    test_loss = None
                    if self.test_dataset is not None:
                        test_prob = self.model.predict(self.test_dataset, verbose=0).ravel()
                        test_m = compute_full_metrics(self.test_y_true, test_prob)

                        test_eval = self.model.evaluate(self.test_dataset, verbose=0)
                        test_loss = float(test_eval[0]) if isinstance(test_eval, (list, tuple)) else float(test_eval)

                    print(f"epoch {epoch+1:>3} loss: train={train_loss:.4f} "
                        f"val={val_loss:.4f} "
                        + (f"test={test_loss:.4f}" if test_loss is not None else "")
                        + ("" if run_full_train else "  (train full metrics skipped this epoch)"))

                    if train_m is not None:
                        self._print_row(epoch, 'train', train_m)
                    self._print_row(epoch, 'val', val_m)
                    if test_m is not None:
                        self._print_row(epoch, 'test', test_m)

                    record = {'epoch': epoch + 1,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'test_loss': test_loss}
                    for k, v in val_m.items():
                        record[f'val_{k}'] = v
                    if train_m is not None:
                        for k, v in train_m.items():
                            record[f'train_{k}'] = v
                    if test_m is not None:
                        for k, v in test_m.items():
                            record[f'test_{k}'] = v

                    self.history.append(record)
                    self._save_history()

                    del val_prob, val_m
                    if train_m is not None:
                        del train_prob, train_m
                    if test_m is not None:
                        del test_prob, test_m
                    gc.collect()

                def _print_row(self, epoch, split_name, m):
                    print(f"epoch {epoch+1:>3} [{split_name:5}] "
                        f"accuracy={m['accuracy']:.4f} "
                        f"majority_acc={m['majority_acc']:.4f} "
                        f"precision={m['precision']:.4f} "
                        f"recall={m['recall']:.4f} "
                        f"f1={m['f1']:.4f} "
                        f"auc={m['roc_auc']:.4f} "
                        f"ap={m['pr_auc']:.4f} "
                        f"nll={m['nll']:.4f} "
                        f"ece={m['ece']:.4f}")

                def _save_history(self):
                    tmp_path = self.save_path + '.tmp'
                    with open(tmp_path, 'w', encoding='utf-8') as fid:
                        json.dump(self.history, fid, indent=2)
                    os.replace(tmp_path, self.save_path)

            epoch_metrics_path = os.path.join(model_folder_path, 'epoch_metrics.json')

            call_backs = [EpochMetricsTable(train_eval_dataset, train_d['output'],
                                            val_dataset, val_d['output'],
                                            save_path=epoch_metrics_path,
                                            test_dataset=test_dataset,
                                            test_y_true=(test_d['output'] if data_test is not None else None),
                                            eval_every=EVAL_EVERY_N_EPOCHS,
                                            total_epochs=epochs),
                            checkpoint, early_stop, plateau_sch,
                ]

            # Reseed numpy right before fit so the epoch-shuffle stream
            # (np.random.permutation in the generator) depends only on `seed`.
            if seed is not None:
                np.random.seed(seed)
            history = train_model.fit(train_dataset,
                                    epochs=epochs,
                                    validation_data=val_dataset,
                                    callbacks=call_backs,
                                    verbose=0)

            history_path, saved_files_path = self.get_path(type_save='models',
                                                        model_name='convlstm_encdec',
                                                        models_save_folder=model_folder_name,
                                                        file_name='history.pkl',
                                                        save_root_folder=os.environ['PIE_PATH'])

            with open(history_path, 'wb') as fid:
                pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)
            print('Wrote history to {}'.format(history_path))

            trace_log("train(): fit() completed -- stopping GPU watchdog")
            stop_gpu_watchdog()

            del train_d, val_d
            gc.collect()

            if getattr(self, '_cache_read_executor', None) is not None:
                self._cache_read_executor.shutdown(wait=True)
                self._cache_read_executor = None
                trace_log("train(): cache-read thread pool shut down")

            return saved_files_path

    def test_chunk(self,
                    data_test,
                    data_opts='',
                    model_path='',
                    visualize=False,
                    checkpoint_dir=None,
                    results_suffix='',
                    overlap_override=None,
                    vis_num_correct=10,
                    vis_num_incorrect=10,
                    vis_seed=42):
        """
        Evaluates the best checkpoint on data_test in CHUNK_SIZE-sample
        chunks (memory-safe), optionally resuming from a checkpoint dir,
        saves predictions to ped_intents<results_suffix>.pkl, and
        optionally saves visualizations. Returns (accuracy, f1) at 0.5.
        """
        with open(os.path.join(model_path, 'configs.pkl'), 'rb') as fid:
            try:
                configs = pickle.load(fid)
            except:
                configs = pickle.load(fid, encoding='bytes')
        train_params = configs[1]
        self.load_model_config(configs[0])
        self.context_model = vgg16.VGG16(input_shape=(224, 224, 3),
                                        include_top=False,
                                        weights='imagenet')
        try:
            test_model = load_model(os.path.join(model_path, 'model.h5'), compile=False)
        except:
            test_model = self.get_model(train_params['model'])
            test_model.load_weights(os.path.join(model_path, 'model.h5'))

        test_model.summary()

        overlap = overlap_override if overlap_override is not None else 1
        print(f"[test_chunk] using overlap={overlap} "
            f"({'override' if overlap_override is not None else 'default'})")

        num_samples = len(data_test['image'])
        CHUNK_SIZE = 100

        data_subset = results_suffix.lstrip('_') if results_suffix else 'test'

        test_target_data = []
        test_results = []
        ped_ids = []
        images = []
        bboxes = []
        vis_results = []
        next_start = 0
        checkpoint_path = None
        local_feat_dir = None
        drive_feat_dir = None

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'test_chunk_checkpoint{results_suffix}.pkl')
            if os.path.exists(checkpoint_path):
                print(f"Resuming from checkpoint: {checkpoint_path}")
                with open(checkpoint_path, 'rb') as fid:
                    ckpt = pickle.load(fid)
                test_target_data = ckpt['test_target_data']
                test_results = ckpt['test_results']
                ped_ids = ckpt['ped_ids']
                images = ckpt['images']
                bboxes = ckpt['bboxes']
                vis_results = ckpt['vis_results']
                next_start = ckpt['next_start']
                print(f"  {next_start}/{num_samples} samples already done, resuming.")

            local_feat_dir = self.get_path(type_save='data',
                                        data_type='features_context_pad_resize',
                                        model_name='vgg16_none',
                                        data_subset=data_subset)
            drive_feat_dir = os.path.join(checkpoint_dir, 'vgg16_feature_cache')
            if os.path.exists(drive_feat_dir):
                os.system(f'cp -rn "{drive_feat_dir}"/* "{local_feat_dir}"/ 2>/dev/null')

        for i in range(next_start, num_samples, CHUNK_SIZE):

            data_test_chunk = {}
            data_test_chunk['intention_binary'] = data_test['intention_binary'][i:min(i + CHUNK_SIZE, num_samples)]
            data_test_chunk['image'] = data_test['image'][i:min(i + CHUNK_SIZE, num_samples)]
            data_test_chunk['ped_id'] = data_test['ped_id'][i:min(i + CHUNK_SIZE, num_samples)]
            data_test_chunk['intention_prob'] = data_test['intention_prob'][i:min(i + CHUNK_SIZE, num_samples)]
            data_test_chunk['bbox'] = data_test['bbox'][i:min(i + CHUNK_SIZE, num_samples)]

            test_data_chunk, test_target_data_chunk = self.get_test_data(data_test_chunk,
                                                                        train_params,
                                                                        self._sequence_length,
                                                                        overlap,
                                                                        data_subset=data_subset)

            tracks, images_chunk, bboxes_chunk, ped_ids_chunk = get_tracks(data_test_chunk,
                                                                            train_params['data_type'],
                                                                            self._sequence_length,
                                                                            overlap)

            with tf.device('/CPU:0'):
                predict_dataset = tf.data.Dataset.from_tensor_slices(
                    {'encoder_input': test_data_chunk[0],
                    'decoder_input': test_data_chunk[1]}
                ).batch(32)

            test_results_chunk = test_model.predict(predict_dataset, verbose=1)

            test_target_data.extend(test_target_data_chunk)
            test_results.extend(test_results_chunk)
            images.extend(images_chunk)
            ped_ids.extend(ped_ids_chunk)
            bboxes.extend(bboxes_chunk)

            j = -1
            for imp, box, ped in zip(images_chunk, bboxes_chunk, ped_ids_chunk):
                j += 1
                vis_results.append({'imp': imp[-1],
                                    'bbox': box[-1],
                                    'ped_id': ped[-1][0],
                                    'res': test_results_chunk[j][0],
                                    'target': test_target_data_chunk[j]})

            next_processed = min(i + CHUNK_SIZE, num_samples)
            print(f"Processed {next_processed}/{num_samples} samples "
                f"({100 * next_processed / num_samples:.1f}%)")

            if checkpoint_path is not None:
                ckpt = {
                    'test_target_data': test_target_data,
                    'test_results': test_results,
                    'ped_ids': ped_ids,
                    'images': images,
                    'bboxes': bboxes,
                    'vis_results': vis_results,
                    'next_start': next_processed,
                }
                tmp_path = checkpoint_path + '.tmp'
                with open(tmp_path, 'wb') as fid:
                    pickle.dump(ckpt, fid, pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, checkpoint_path)
                print(f"  Checkpoint saved ({next_processed}/{num_samples})")

                os.system(f'cp -rn "{local_feat_dir}"/* "{drive_feat_dir}"/ 2>/dev/null'
                        if os.path.exists(drive_feat_dir) else
                        f'mkdir -p "{drive_feat_dir}" && cp -r "{local_feat_dir}"/* "{drive_feat_dir}"/ 2>/dev/null')

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

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        if visualize:
            save_visualizations(vis_results,
                                model_path=model_path,
                                results_suffix=results_suffix,
                                num_correct=vis_num_correct,
                                num_incorrect=vis_num_incorrect,
                                seed=vis_seed)

        return acc, f1
