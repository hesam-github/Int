\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
import tensorflow as tf
import numpy as np
import os
import pickle
import time
import subprocess
import threading
from datetime import datetime
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


CACHE_READ_WORKERS = 8


_TRACE_LOG_PATH = os.path.join(os.environ['PIE_PATH'], 'trace_main.log')
_GPU_LOG_PATH = os.path.join(os.environ['PIE_PATH'], 'trace_gpu.log')

_trace_log_fid = open(_TRACE_LOG_PATH, 'a', buffering=1, encoding='utf-8')
_trace_log_lock = threading.Lock()


def trace_log(msg):
    \
\
    line = f"[{datetime.now().strftime('%H:%M:%S.%f')}] {msg}"
    with _trace_log_lock:
        print(line)
        _trace_log_fid.write(line + "\n")
        _trace_log_fid.flush()
        os.fsync(_trace_log_fid.fileno())


_gpu_watchdog_stop_event = threading.Event()
_gpu_watchdog_thread = None


def _gpu_watchdog_loop():
    with open(_GPU_LOG_PATH, 'a', buffering=1, encoding='utf-8') as gfid:
        while not _gpu_watchdog_stop_event.is_set():
            ts = datetime.now().strftime('%H:%M:%S.%f')
            try:
                result = subprocess.run(
                    ['nvidia-smi',
                     '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5,
                )
                line = f"[{ts}] {result.stdout.strip()}"
            except Exception as e:
                line = f"[{ts}] nvidia-smi FAILED: {type(e).__name__}: {e}"
            gfid.write(line + "\n")
            gfid.flush()
            os.fsync(gfid.fileno())
            _gpu_watchdog_stop_event.wait(2.0)


def start_gpu_watchdog():
    \
\
\
    global _gpu_watchdog_thread
    if _gpu_watchdog_thread is not None and _gpu_watchdog_thread.is_alive():
        return
    _gpu_watchdog_stop_event.clear()
    _gpu_watchdog_thread = threading.Thread(target=_gpu_watchdog_loop, daemon=True)
    _gpu_watchdog_thread.start()
    trace_log(f"GPU watchdog thread started (polling nvidia-smi every 2s), "
              f"writing to {_GPU_LOG_PATH}")


def stop_gpu_watchdog():
    \
    _gpu_watchdog_stop_event.set()
    trace_log("GPU watchdog thread stop requested")


def _windowed_frame_generator(model, images, bboxes, ped_ids, decoder_input,
                              output, save_path, batch_size, data_type_label, shuffle):
    \
\
\
\
\
\
\
\
\
\
\
\
\
\
    n = len(images)
    order = np.random.permutation(n) if shuffle else np.arange(n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = order[start:end]

        batch_images = [images[i] for i in idx]
        batch_bboxes = [bboxes[i] for i in idx]
        batch_ped_ids = [ped_ids[i] for i in idx]

        batch_features = model.load_images_and_process(
            batch_images, batch_bboxes, batch_ped_ids,
            data_type=data_type_label, save_path=save_path,
        )

        batch_decoder = np.asarray(decoder_input[idx], dtype=np.float32)
        batch_output = np.asarray(output[idx], dtype=np.float32)

        assert batch_output.ndim == 2 and batch_output.shape[1] == 1, (
            f"Expected batch_output shape (batch, 1), got {batch_output.shape}. "
            f"This generator's output_signature assumes this exact shape -- "
            f"if the data pipeline changed, update both this assertion and "
            f"build_generator_dataset's output_signature together.")

        yield batch_features.astype(np.float32), batch_decoder, batch_output

def build_generator_dataset(model, images, bboxes, ped_ids, decoder_input,
                            output, save_path, seq_length, feature_shape,
                            batch_size, data_type_label, shuffle,
                            private_threadpool_size=32, prefetch_size=1):
    \
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
    def gen():
        for feat_batch, dec_batch, y_batch in _windowed_frame_generator(
                model, images, bboxes, ped_ids, decoder_input, output,
                save_path, batch_size, data_type_label, shuffle):
            yield (feat_batch, dec_batch), y_batch

    output_signature = (
        (
            tf.TensorSpec(shape=(None, seq_length) + feature_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(None, decoder_input.shape[1], decoder_input.shape[2]), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = private_threadpool_size
    options.experimental_threading.max_intra_op_parallelism = private_threadpool_size
    ds = ds.with_options(options)

    ds = ds.prefetch(prefetch_size)

    return ds

class PIEIntent(object):
    \
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
    def __init__(self,
                 num_hidden_units=128,
                 regularizer_val=0.001,
                 activation='tanh',
                 lstm_dropout=0.4,
                 lstm_recurrent_dropout=0.2,
                 convlstm_num_filters=64,
                 convlstm_kernel_size=2):


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

        self._model_name = 'convlstm_encdec'

    def get_path(self,
                 type_save='models',
                 models_save_folder='',
                 model_name='convlstm_encdec',
                 file_name='',
                 data_subset='',
                 data_type='',
                 save_root_folder=os.environ['PIE_PATH'] + '/data/'):
        \
\
\
\
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
        \
\
\
        config = dict()
        config['num_hidden'] = self._num_hidden_units
        config['reg_value'] = self.reg_value
        config['activation'] = self._activation
        config['sequence_length'] = self._sequence_length
        config['lstm_dropout'] = self._lstm_dropout
        config['lstm_recurrent_dropout'] = self._lstm_recurrent_dropout
        config['convlstm_num_filters'] = self._convlstm_num_filters
        config['convlstm_kernel_size'] = self._convlstm_kernel_size

        config['encoder_input_size'] = self._encoder_input_size

        config['decoder_input_size'] = self._decoder_input_size
        config['decoder_dense_output_size'] = self._decoder_dense_output_size

        config['encoder_seq_length'] = self._encoder_seq_length
        config['decoder_seq_length'] = self._decoder_seq_length

        print(config)
        return config

    def load_model_config(self, config):
        \
\
\
        self._num_hidden_units = config['num_hidden']

        self.reg_value = config['reg_value']
        self._activation = config['activation']
        self._encoder_input_size = config['encoder_input_size']
        self._encoder_seq_length = config['encoder_seq_length']
        self._sequence_length = config['sequence_length']

        self._lstm_dropout = config['lstm_dropout']
        self._lstm_recurrent_dropout = config['lstm_recurrent_dropout']
        self._convlstm_num_filters = config['convlstm_num_filters']
        self._convlstm_kernel_size = config['convlstm_kernel_size']

        self._encoder_input_size = config['decoder_input_size']
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
        \
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
        print("Generating {} features crop_type=context crop_mode=pad_resize \nsave_path={}, ".format(data_type, save_path))
        try:
            convnet = self.context_model
        except:
            raise Exception("No context model is defined")

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
                    miss_specs.append((i, j, imp, b, img_save_path))


        total_frames = len(hit_specs) + len(miss_specs)
        trace_log(f"load_images_and_process[{data_type}]: batch hits={len(hit_specs)} "
                  f"misses={len(miss_specs)} (total={total_frames})")
        if miss_specs:
            trace_log(f"load_images_and_process[{data_type}]: WARNING -- "
                      f"{len(miss_specs)} cache miss(es) in this batch; if the "
                      f"pre-caching pass was supposed to be 100% complete for "
                      f"this split, this is a real residual-miss finding, not "
                      f"expected allocator noise.")


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
            with ThreadPoolExecutor(max_workers=CACHE_READ_WORKERS) as executor:
                future_to_ij = {
                    executor.submit(_read_cached_feature, path): (i, j)
                    for (i, j, path) in hit_specs
                }
                for future in future_to_ij:
                    i, j = future_to_ij[future]
                    features_grid[i][j] = future.result()
                    done_count += 1
                    if done_count % progress_step == 0:
                        update_progress(done_count / len(hit_specs))
            update_progress(1.0)


        for (i, j, imp, b, img_save_path) in miss_specs:
            img_data = load_img(imp)
            bbox = jitter_bbox([b], imp, 'enlarge', 2)[0]
            bbox = squarify(bbox, 1, img_data.size[0])
            bbox = list(map(int, bbox[0:4]))
            cropped_image = img_data.crop(bbox)
            img_data = img_pad(cropped_image, mode='pad_resize', size=224)
            image_array = img_to_array(img_data)
            preprocessed_img = vgg16.preprocess_input(image_array)
            expanded_img = np.expand_dims(preprocessed_img, axis=0)
            img_features = convnet(expanded_img, training=False).numpy()
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


        sequences = []
        for i in range(n_seqs):
            img_seq = [np.squeeze(features_grid[i][j]) for j in range(len(img_sequences[i]))]
            sequences.append(img_seq)
        sequences = np.array(sequences)

        if debug_every is not None:
            _debug_snap("end of function (before np.array conversion overhead shown above)")

        return sequences

    def get_tracks(self, dataset, data_type, seq_length, overlap):
        \
\
\
        overlap_stride = seq_length if overlap == 0 else\
            int((1 - overlap) * seq_length)

        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        d_types = []
        for k in data_type.keys():
            d_types.extend(data_type[k])
        d = {}

        if 'bbox' in d_types:
            d['bbox'] = dataset['bbox']
        if 'intention_binary' in d_types:
            d['intention_binary'] = dataset['intention_binary']
        if 'intention_prob' in d_types:
            d['intention_prob'] = dataset['intention_prob']

        bboxes = dataset['bbox'].copy()
        images = dataset['image'].copy()
        ped_ids = dataset['ped_id'].copy()

        for k in d.keys():
            tracks = []
            for track in d[k]:
                tracks.extend([track[i:i + seq_length] for i in
                               range(0, len(track) - seq_length + 1, overlap_stride)])
            d[k] = tracks

        pid = []
        for p in ped_ids:
            pid.extend([p[i:i + seq_length] for i in
                        range(0, len(p) - seq_length + 1, overlap_stride)])
        ped_ids = pid

        im = []
        for img in images:
            im.extend([img[i:i + seq_length] for i in
                       range(0, len(img) - seq_length + 1, overlap_stride)])
        images = im

        bb = []
        for bbox in bboxes:
            bb.extend([bbox[i:i + seq_length] for i in
                       range(0, len(bbox) - seq_length + 1, overlap_stride)])

        bboxes = bb
        return d, images, bboxes, ped_ids

    def concat_data(self, data, data_type):
        \
\
\
\
        if not data_type:
            return []
        d = []
        for dt in data_type:
            d.append(np.array(data[dt]))
        if len(d) > 1:
            d = np.concatenate(d, axis=2)
        else:
            d = d[0]
        return d

    def _normalize_decoder_bbox(self, decoder_input, decoder_input_type):
            \
\
\
\
\
\
\
\
\
\
\
\
\
\
\
            if 'bbox' not in decoder_input_type:
                return decoder_input
            img_w, img_h = 1920.0, 1080.0
            decoder_input = np.asarray(decoder_input, dtype=np.float32).copy()
            decoder_input[..., [0, 2]] /= img_w
            decoder_input[..., [1, 3]] /= img_h
            return decoder_input

    def get_train_val_data(self, data, data_type, seq_length, overlap):
        \
\
\
\
        tracks, images, bboxes, ped_ids = self.get_tracks(data, data_type, seq_length, overlap)

        encoder_input = self.concat_data(tracks, data_type['encoder_input_type'])
        decoder_input = self.concat_data(tracks, data_type['decoder_input_type'])
        output = self.concat_data(tracks, data_type['output_type'])

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

    def get_test_data(self, data, train_params, seq_length, overlap):
        \
\
\
\
\
        tracks, images, bboxes, ped_ids = self.get_tracks(data,
                                                           train_params['data_type'],
                                                           seq_length,
                                                           overlap)

        encoder_input = self.concat_data(tracks, train_params['data_type']['encoder_input_type'])
        decoder_input = self.concat_data(tracks, train_params['data_type']['decoder_input_type'])
        output = self.concat_data(tracks, train_params['data_type']['output_type'])

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
                                                  data_type='test',
                                                  save_path=self.get_path(type_save='data',
                                                                          data_type='features_context_pad_resize',
                                                                          model_name='vgg16_none',
                                                                          data_subset='test'))
        output = output[:, 0]
        return ([test_img, decoder_input], output)

    def get_model(self, model):
        train_model = self.pie_convlstm_encdec()
        return train_model

    def create_lstm_model(self,
                          name='convlstm_encdec',
                          r_state=True,
                          r_sequence=False):
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
        \
\
\
        encoder_input = input_data = Input(shape=(self._sequence_length,) + self.context_model.output_shape[1:],
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
                                     name='decoder_dense')(decoder_model)

        decoder_output = decoder_dense_output

        self.train_model = Model(inputs=[encoder_input, decoder_input],
                                 outputs=decoder_output)

        self.train_model.summary()
        return self.train_model
    def train(self,
                data_train,
                data_val,
                batch_size=128,
                epochs=400,
                optimizer_type='rmsprop',
                optimizer_params={'lr': 0.00001, 'clipvalue': 1.0, 'decay': 0.0},
                loss=['binary_crossentropy'],
                metrics=['acc'],
                data_opts=''):
            \
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
            data_type = {'encoder_input_type': data_opts['encoder_input_type'],
                        'decoder_input_type': data_opts['decoder_input_type'],
                        'output_type': data_opts['output_type']}

            train_config = {'batch_size': batch_size,
                            'epoch': epochs,
                            'optimizer_type': optimizer_type,
                            'optimizer_params': optimizer_params,
                            'loss': loss,
                            'metrics': metrics,
                            'learning_scheduler_mode': 'plateau',
                            'learning_scheduler_params': {'exp_decay_param': 0.3,
                                                        'step_drop_rate': 0.5,
                                                        'epochs_drop_rate': 20.0,
                                                        'plateau_patience': 5,
                                                        'min_lr': 0.0000001,
                                                        'monitor_value': 'val_loss'},
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
            trace_log("train(): GPU watchdog started -- about to build "
                      "datasets and call fit()")

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

            val_dataset = build_generator_dataset(
                self, val_d['images'], val_d['bboxes'], val_d['ped_ids'],
                val_d['decoder_input'], val_d['output'], val_save_path,
                seq_length, feature_shape, batch_size=batch_size,
                data_type_label='val', shuffle=False)

            train_model = self.pie_convlstm_encdec()

            optimizer = RMSprop(learning_rate=optimizer_params['lr'],
                                clipvalue=optimizer_params['clipvalue'])

            train_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            print('TRAINING: loss={} metrics={}'.format(loss, metrics))

            model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")

            model_path, _ = self.get_path(type_save='models',
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

            early_stop = EarlyStopping(monitor='val_loss',
                                    min_delta=0.0001,
                                    patience=5,
                                    verbose=1)
            checkpoint = ModelCheckpoint(filepath=model_path,
                                        save_best_only=True,
                                        save_weights_only=False,
                                        monitor=train_config['learning_scheduler_params']['monitor_value'])
            plateau_sch = ReduceLROnPlateau(monitor=train_config['learning_scheduler_params']['monitor_value'],
                    factor=train_config['learning_scheduler_params']['step_drop_rate'],
                    patience=train_config['learning_scheduler_params']['plateau_patience'],
                    min_lr=train_config['learning_scheduler_params']['min_lr'],
                    verbose=1)


            class LogitDiagnostic(tf.keras.callbacks.Callback):
                def __init__(self, val_dataset):
                    super().__init__()
                    self.val_dataset = val_dataset
                    self._logit_model = None

                def on_train_begin(self, logs=None):
                    dense_layer = self.model.get_layer('decoder_dense')
                    pre_activation_logit = tf.matmul(dense_layer.input, dense_layer.kernel) + dense_layer.bias
                    self._logit_model = tf.keras.Model(
                        inputs=self.model.input,
                        outputs=pre_activation_logit)

                def on_epoch_end(self, epoch, logs=None):
                    batch = next(iter(self.val_dataset.take(1)))
                    (enc_in, dec_in), _ = batch
                    logits = self._logit_model.predict((enc_in, dec_in), verbose=0)
                    print(f"  [logit diag] epoch {epoch}: "
                        f"min={logits.min():.2f} max={logits.max():.2f} "
                        f"mean={logits.mean():.2f} std={logits.std():.4f}")

            call_backs = [checkpoint, early_stop, plateau_sch, LogitDiagnostic(val_dataset)]

            history = train_model.fit(train_dataset,
                                    epochs=epochs,
                                    validation_data=val_dataset,
                                    callbacks=call_backs,
                                    verbose=1)

            history_path, saved_files_path = self.get_path(type_save='models',
                                                        model_name='convlstm_encdec',
                                                        models_save_folder=model_folder_name,
                                                        file_name='history.pkl',
                                                        save_root_folder=os.environ['PIE_PATH'])

            with open(history_path, 'wb') as fid:
                pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)
            print('Wrote history to {}'.format(history_path))

            trace_log("train(): fit() completed successfully -- stopping GPU watchdog")
            stop_gpu_watchdog()

            del train_d, val_d
            gc.collect()

            return saved_files_path

    def test_chunk(self,
                   data_test,
                   data_opts='',
                   model_path='',
                   visualize=False,
                   checkpoint_dir=None):
        \
\
\
\
\
\
\
\
\
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
            test_model = load_model(os.path.join(model_path, 'model.h5'))
        except:
            test_model = self.get_model(train_params['model'])
            test_model.load_weights(os.path.join(model_path, 'model.h5'))

        test_model.summary()

        overlap = 1
        num_samples = len(data_test['image'])


        CHUNK_SIZE = 100

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
            checkpoint_path = os.path.join(checkpoint_dir, 'test_chunk_checkpoint.pkl')
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
                print(f"  {next_start}/{num_samples} samples already done, "
                      f"resuming from there.")

            local_feat_dir = self.get_path(type_save='data',
                                           data_type='features_context_pad_resize',
                                           model_name='vgg16_none',
                                           data_subset='test')
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
                                                                          overlap)

            tracks, images_chunk, bboxes_chunk, ped_ids_chunk = self.get_tracks(data_test_chunk,
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
                print(f"  Checkpoint saved to Drive ({next_processed}/{num_samples})")

                os.system(f'cp -rn "{local_feat_dir}"/* "{drive_feat_dir}"/ 2>/dev/null'
                          if os.path.exists(drive_feat_dir) else
                          f'mkdir -p "{drive_feat_dir}" && cp -r "{local_feat_dir}"/* "{drive_feat_dir}"/ 2>/dev/null')

        acc = accuracy_score(test_target_data, np.round(test_results))
        f1 = f1_score(test_target_data, np.round(test_results))

        save_results_path = os.path.join(model_path, 'ped_intents.pkl')
        if not os.path.exists(save_results_path):
            results = {'ped_id': ped_ids,
                       'images': images,
                       'results': test_results,
                       'gt': test_target_data}
            with open(save_results_path, 'wb') as fid:
                pickle.dump(results, fid, pickle.HIGHEST_PROTOCOL)

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        return acc, f1
