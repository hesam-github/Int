# pie_data_Hesam.py
"""
Interface for the PIE dataset:

A. Rasouli, I. Kotseruba, T. Kunic, and J. Tsotsos, "PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
Trajectory Prediction", ICCV 2019.

MIT License

Copyright (c) 2019 Amir Rasouli, Iuliia Kotseruba

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

-------------------------------------------------------------------------------
Modifications by Hesam Mardani, Isfahan University of Technology (1404):

1. extract_and_save_images() -- two-layer cache strategy added:
   Layer 1 (video level): if all required frames for a video already exist
   on disk, skip opening the video entirely (avoids cv2.VideoCapture).
   Layer 2 (frame level): if individual frames exist, skip cv2.imwrite.
   Rationale: on re-runs, the original code decoded every video from scratch
   even when nothing needed writing. With this change, re-runs over already-
   extracted sets (e.g. set01, set02, set03, set05) complete in seconds.

   Known minor inefficiency: frames_list remains a Python list, so the inner
   loop uses O(n) membership testing. Converting to set() before the while
   loop would make it O(1). Not yet applied because Layer 1 skips fully-
   extracted videos entirely, making the inner loop rarely reached on re-runs.

2. get_tracks() -- added as a module-level function (see note at its
   definition below) after being audited out of PIEIntent in pie_intent.py:
   it never used `self` there, and is purely a windowing/data-prep step
   over the sequence dict produced by generate_data_trajectory_sequence(),
   so it now lives next to the class that produces that dict's shape.

"""


import pickle
import cv2
import sys

import xml.etree.ElementTree as ET
import numpy as np

from os.path import join, abspath, isfile, isdir
from os import makedirs, listdir
from sklearn.model_selection import train_test_split, KFold

from functools import lru_cache
import hashlib
import json

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from utils import squarify, update_progress, PIE_IMG_WIDTH, PIE_IMG_HEIGHT

def get_tracks(dataset, data_type, seq_length, overlap):
    """Window tracks for every requested feature, not only bbox.

    Each frame step is forced to a 1-D vector, so a scalar occlusion and a
    4-number bbox can be concatenated on the feature axis.
    """
    overlap_stride = seq_length if overlap == 0 else int((1 - overlap) * seq_length)
    overlap_stride = 1 if overlap_stride < 1 else overlap_stride

    wanted = []
    for key_list in data_type.values():
        wanted.extend(key_list)

    d = {}
    for key in dict.fromkeys(wanted):
        if key not in dataset:
            raise KeyError(
                f"feature {key!r} is not in this sequence. "
                f"Available: {sorted(dataset.keys())}")
        d[key] = dataset[key]

    bboxes = dataset['bbox'].copy()
    images = dataset['image'].copy()
    ped_ids = dataset['ped_id'].copy()

    for k in list(d.keys()):
        tracks = []
        for track in d[k]:
            for i in range(0, len(track) - seq_length + 1, overlap_stride):
                window = [np.asarray(step, dtype=np.float32).reshape(-1)
                          for step in track[i:i + seq_length]]
                tracks.append(window)
        d[k] = tracks

    pid = []
    for p in ped_ids:
        pid.extend([p[i:i + seq_length]
                    for i in range(0, len(p) - seq_length + 1, overlap_stride)])
    ped_ids = pid

    im = []
    for img in images:
        im.extend([img[i:i + seq_length]
                   for i in range(0, len(img) - seq_length + 1, overlap_stride)])
    images = im

    bb = []
    for bbox in bboxes:
        bb.extend([bbox[i:i + seq_length]
                   for i in range(0, len(bbox) - seq_length + 1, overlap_stride)])
    bboxes = bb
    return d, images, bboxes, ped_ids

def print_annotated_pedestrian_fields(imdb, set_id='set01'):
    """Print pedestrian fields that are actually in one annotation XML."""
    xmls = sorted(
        name for name in listdir(join(imdb._annotation_path, set_id))
        if name.endswith('_annt.xml')
    )
    vid = xmls[0].replace('_annt.xml', '')
    ann = imdb._get_annotations(set_id, vid)
    ped = next(iter(ann['ped_annotations'].values()))
    per_frame = ['bbox', 'occlusion'] + sorted(ped['behavior'].keys())
    print(f"[annotations] video {set_id}/{vid}")
    print(f"[annotations] per-frame pedestrian fields: {per_frame}")
    print("[annotations] center is computed from bbox, not stored in the XML")
    attrs = sorted(ped.get('attributes', {}).keys())
    if attrs:
        print(f"[annotations] track-level attributes, not frame sequences: {attrs}")
    print("[annotations] obd_speed and gps_speed belong to the vehicle, not the pedestrian")
    return per_frame

class PIE(object):
    def __init__(self, regen_database=False, data_path=''):
        """
        Class constructor
        :param regen_database: Whether generate the database or not
        :param data_path: The path to wh
        """
        self._RANDOM_SEED = 42
        self._annotated_frames_cache = {}       # set_id -> {video_id: [num_frames, *frame_ids]}
        self._frame_numbers_cache = {}          # set_id -> {video_id: [num_frames, *frame_ids]}
        self._year = '2019'
        self._name = 'pie'
        self._image_ext = '.png'
        self._regen_database = regen_database

        # Paths
        self._pie_path = data_path if data_path else self._get_default_path()
        assert isdir(self._pie_path), \
            'pie path does not exist: {}'.format(self._pie_path)

        self._annotation_path = join(self._pie_path, 'annotations')
        self._annotation_attributes_path = join(self._pie_path, 'annotations_attributes')
        self._annotation_vehicle_path = join(self._pie_path, 'annotations_vehicle')

        self._clips_path = join(self._pie_path, 'PIE_clips')
        self._images_path = join(self._pie_path, 'images')

    # Path generators
    @property
    def cache_path(self):
        """
        Generates a path to save cache files
        :return: Cache file folder path
        """
        cache_path = abspath(join(self._pie_path, 'data_cache'))
        if not isdir(cache_path):
            makedirs(cache_path)
        return cache_path

    def _get_default_path(self):
        """
        Returns the default path where pie is expected to be installed.
        """
        return 'data/pie'

    def _get_image_set_ids(self, image_set):
        """
        Returns default image set ids
        :param image_set: Image set split
        :return: Set ids of the image set
        """
        image_set_nums = {'train': ['set01', 'set02', 'set04'],
                          'val': ['set05', 'set06'],
                          'test': ['set03'],
                          'all': ['set01', 'set02', 'set03',
                                  'set04', 'set05', 'set06']}
        return image_set_nums[image_set]


    @staticmethod
    @lru_cache(maxsize=None)
    def _build_image_path(images_root, sid, vid, fid):
        """
        Cached path builder. Pure function: same (root, sid, vid, fid) -> same path.
        (root, set, video, frame) tuples in the dataset.
        """
        return join(images_root, sid, vid, '{:05d}.png'.format(fid))

    def _get_image_path(self, sid, vid, fid):
        """
        Generates and returns the image path given ids
        :param sid: Set id
        :param vid: Video id
        :param fid: Frame id
        :return: Return the path to the given image
        """
        return self._build_image_path(self._images_path, sid, vid, fid)



# self._images_path = data/pie/images
# sid = set01
# vid = video_0003
# created path-->  data/pie/images/set01/video_0003
# fid = 27 -->{:05d}.png --> 00027.png

 
    def _print_dict(self, dic):
        """
        'full' mode: prints every key, unchanged (original behavior).
        'clean' mode: prints ONLY the keys actually consumed downstream for
        the CURRENT data_split_type -- so a dead key (e.g. max_size_predict,
        balance, or kfold_params when split_type != 'kfold') never appears.
        """
        from log_mode import is_full
        if is_full():
            for k, v in dic.items():
                print('%s: %s' % (str(k), str(v)))
            return

        relevant_keys = ['seq_type', 'min_track_size', 'seq_overlap_rate',
                        'max_size_observe', 'crop_type', 'crop_mode',
                        'decoder_input_type', 'output_type', 'data_split_type']
        split_type = dic.get('data_split_type')
        if split_type == 'random':
            relevant_keys.append('random_params')
        elif split_type == 'kfold':
            relevant_keys.append('kfold_params')
        for k in relevant_keys:
            if k in dic:
                print('%s: %s' % (str(k), str(dic[k])))
    # Data processing helpers
    def _get_width(self):
        """
        Returns image width
        :return: Image width
        """
        return PIE_IMG_WIDTH

    def _get_height(self):
        """
        Returns image height
        :return: Image height
        """
        return PIE_IMG_HEIGHT

    def _get_dim(self):
        """
        Returns the image dimensions
        :return: Image dimensions
        """
        return PIE_IMG_WIDTH, PIE_IMG_HEIGHT

    # Image processing helpers

    def get_annotated_frame_numbers(self, set_id):
        """
        Generates and returns a dictionary of videos and annotated frames for each video in the give set
        :param set_id: Set to generate annotated frames
        :return: A dictionary of form
                {<video_id>: [<number_of_frames>,<annotated_frame_id_0>,... <annotated_frame_id_n>]}

        Two-level cache:
        1. In-memory dict (this Python session)
        2. CSV file in annotations folder (across sessions)
        3. XML parse (cold path)
        Public signature and return structure unchanged.
        """
        # Level 1: memory
        if set_id in self._annotated_frames_cache:
            return self._annotated_frames_cache[set_id]

        print("Generating annotated frame numbers for", set_id)
        annotated_frames_file = join(
            self._pie_path, "annotations", set_id, set_id + '_annotated_frames.csv'
        )

        # Level 2: disk (CSV)
        if isfile(annotated_frames_file):
            with open(annotated_frames_file, 'rt') as f:
                annotated_frames = {
                    x.split(',')[0]: [int(fr) for fr in x.split(',')[1:]]
                    for x in f.readlines()
                }
            self._annotated_frames_cache[set_id] = annotated_frames
            return annotated_frames

        # Level 3: XML parse (cold path) -- unchanged logic from original
        annotated_frames = {
            v.split('_annt.xml')[0]: []
            for v in sorted(listdir(join(self._annotation_path, set_id)))
            if v.endswith("annt.xml")
        }
        for vid, annot_frames in sorted(annotated_frames.items()):
            _frames = []
            path_to_file = join(self._annotation_path, set_id, vid + '_annt.xml')
            tree = ET.parse(path_to_file)
            tracks = tree.findall('./track')
            for t in tracks:
                if t.get('label') != 'pedestrian':
                    continue
                boxes = t.findall('./box')
                for b in boxes:
                    if int(b.get('outside')) == 1:
                        continue
                    _frames.append(int(b.get('frame')))
            _frames = sorted(set(_frames))
            annot_frames.append(len(_frames))
            annot_frames.extend(_frames)

        with open(annotated_frames_file, 'wt') as fid:
            for vid, annot_frames in sorted(annotated_frames.items()):
                fid.write(vid)
                for fr in annot_frames:
                    fid.write("," + str(fr))
                fid.write('\n')

        self._annotated_frames_cache[set_id] = annotated_frames
        return annotated_frames


    def get_frame_numbers(self, set_id):
        """
        Generates and returns a dictionary of videos and  frames for each video in the give set
        :param set_id: Set to generate annotated frames
        :return: A dictionary of form
                {<video_id>: [<number_of_frames>,<frame_id_0>,... <frame_id_n>]}
        """
        if set_id in self._frame_numbers_cache:
            return self._frame_numbers_cache[set_id]

        print("Generating frame numbers for", set_id)
        frame_ids = {
            v.split('_annt.xml')[0]: []
            for v in sorted(listdir(join(self._annotation_path, set_id)))
            if v.endswith("annt.xml")
        }
        for vid, frames in sorted(frame_ids.items()):
            path_to_file = join(self._annotation_path, set_id, vid + '_annt.xml')
            tree = ET.parse(path_to_file)
            num_frames = int(tree.find("./meta/task/size").text)
            frames.extend(range(num_frames))
            frames.insert(0, num_frames)

        self._frame_numbers_cache[set_id] = frame_ids
        return frame_ids

    def extract_and_save_images(self, extract_frame_type='annotated'):
        """
        Extracts images from clips and saves on hard drive
        :param extract_frame_type: Whether to extract 'all' frames or only the ones that are 'annotated'
                             Note: extracting 'all' frames requires approx. 3TB space whereas
                                   'annotated' requires approx. 1TB
        """
        set_folders = [f for f in sorted(listdir(self._clips_path))]
        for set_id in set_folders:
            print('Extracting frames from', set_id)
            set_folder_path = join(self._clips_path, set_id)
            if extract_frame_type == 'annotated':
                extract_frames = self.get_annotated_frame_numbers(set_id)
            else:
                extract_frames = self.get_frame_numbers(set_id)

            set_images_path = join(self._pie_path, "images", set_id)
            for vid, frames in sorted(extract_frames.items()):
                print(vid)
                video_images_path = join(set_images_path, vid)
                num_frames = frames[0]
                frames_list = frames[1:]
                if not isdir(video_images_path):
                    makedirs(video_images_path)

                # Added By Hesam Mardani: a cache strategy for not decoding videos every time
                # Cache check: if every required frame for this video already
                # exists on disk, skip opening/decoding the video entirely.
                # This is the expensive step extract_and_save_images was
                # repeating on every re-run, even when nothing needed writing.
                already_extracted = all(
                    isfile(join(video_images_path, f'{f:05d}.png'))
                    for f in frames_list
                )
                if already_extracted:
                    print(f'Skipping {vid}: all {num_frames} frames already extracted')
                    continue

                vidcap = cv2.VideoCapture(join(set_folder_path, vid + '.mp4'))
                success, image = vidcap.read()
                frame_num = 0
                img_count = 0
                if not success:
                    print('Failed to open the video {}'.format(vid))
                while success:
                    if frame_num in frames_list:
                        update_progress(img_count / num_frames)
                        img_count += 1
                        filename = f"{frame_num:05d}.png"
                        if not isfile(join(video_images_path, filename)):
                            cv2.imwrite(join(video_images_path, filename), image)
                    success, image = vidcap.read()
                    frame_num += 1
                if num_frames != img_count:
                    print('num images don\'t match {}/{}'.format(num_frames, img_count))
                print('\n')

    # Annotation processing helpers
    def _map_text_to_scalar(self, label_type, value):
        """
        Maps a text label in XML file to scalars
        :param label_type: The label type
        :param value: The text to be mapped
        :return: The scalar value
        """
        map_dic = {'occlusion': {'none': 0, 'part': 1, 'full': 2},
                   'action': {'standing': 0, 'walking': 1},
                   'look': {'not-looking': 0, 'looking': 1},
                   'gesture': {'__undefined__': 0, 'hand_ack': 1, 'hand_yield': 2,
                               'hand_rightofway': 3, 'nod': 4, 'other': 5},
                   'cross': {'not-crossing': 0, 'crossing': 1, 'crossing-irrelevant': -1},
                   'crossing': {'not-crossing': 0, 'crossing': 1, 'irrelevant': -1},
                   'age': {'child': 0, 'young': 1, 'adult': 2, 'senior': 3},
                   'designated': {'ND': 0, 'D': 1},
                   'gender': {'n/a': 0, 'female': 1, 'male': 2},
                   'intersection': {'midblock': 0, 'T': 1, 'T-left': 2, 'T-right': 3, 'four-way': 4},
                   'motion_direction': {'n/a': 0, 'LAT': 1, 'LONG': 2},
                   'traffic_direction': {'OW': 0, 'TW': 1},
                   'signalized': {'n/a': 0, 'C': 1, 'S': 2, 'CS': 3},
                   'vehicle': {'car': 0, 'truck': 1, 'bus': 2, 'train': 3, 'bicycle': 4, 'bike': 5},
                   'sign': {'ped_blue': 0, 'ped_yellow': 1, 'ped_white': 2, 'ped_text': 3, 'stop_sign': 4,
                            'bus_stop': 5, 'train_stop': 6, 'construction': 7, 'other': 8},
                   'traffic_light': {'regular': 0, 'transit': 1, 'pedestrian': 2},
                   'state': {'__undefined__': 0, 'red': 1, 'yellow': 2, 'green': 3}}
        # cross: a frame based behaviour of pedestrian : 0 0 1 1 ->not_c not_c cross cross  4 frames
        # crossing: a general attribute for whole track : 1 it means at the end the pedestrian has crossed
        # OW: one way , TW: two way
        # C: crosswalk, S: signalized intersection (traffic light), CS: both
        return map_dic[label_type][value]

    def _map_scalar_to_text(self, label_type, value):
        """
        Maps a scalar value to a text label
        :param label_type: The label type
        :param value: The scalar to be mapped
        :return: The text label
        """
        map_dic = {'occlusion': {0: 'none', 1: 'part', 2: 'full'},
                   'action': {0: 'standing', 1: 'walking'},
                   'look': {0: 'not-looking', 1: 'looking'},
                   'hand_gesture': {0: '__undefined__', 1: 'hand_ack',
                                    2: 'hand_yield', 3: 'hand_rightofway',
                                    4: 'nod', 5: 'other'},
                   'cross': {0: 'not-crossing', 1: 'crossing', -1: 'crossing-irrelevant'},
                   'crossing': {0: 'not-crossing', 1: 'crossing', -1: 'irrelevant'},
                   'age': {0: 'child', 1: 'young', 2: 'adult', 3: 'senior'},
                   'designated': {0: 'ND', 1: 'D'},
                   'gender': {0: 'n/a', 1: 'female', 2: 'male'},
                   'intersection': {0: 'midblock', 1: 'T', 2: 'T-left', 3: 'T-right', 4: 'four-way'},
                   'motion_direction': {0: 'n/a', 1: 'LAT', 2: 'LONG'},
                   'traffic_direction': {0: 'OW', 1: 'TW'},
                   'signalized': {0: 'n/a', 1: 'C', 2: 'S', 3: 'CS'},
                   'vehicle': {0: 'car', 1: 'truck', 2: 'bus', 3: 'train', 4: 'bicycle', 5: 'bike'},
                   'sign': {0: 'ped_blue', 1: 'ped_yellow', 2: 'ped_white', 3: 'ped_text', 4: 'stop_sign',
                            5: 'bus_stop', 6: 'train_stop', 7: 'construction', 8: 'other'},
                   'traffic_light': {0: 'regular', 1: 'transit', 2: 'pedestrian'},
                   'state': {0: '__undefined__', 1: 'red', 2: 'yellow', 3: 'green'}}

        return map_dic[label_type][value]

    def _get_annotations(self, setid, vid):# Temporal informations / a General info which can change: looking , not looking
        #: reads the whole XML file of a video -> dictionary
        """
        Generates a dictionary of annotations by parsing the video XML file
        :param setid: The set id
        :param vid: The video id
        :return: A dictionary of annotations
        """
        path_to_file = join(self._annotation_path, setid, vid + '_annt.xml')
        # e.g: data/pie/annotations/set01/video_0001_annt.xml
        print(path_to_file)

        tree = ET.parse(path_to_file)
        ped_annt = 'ped_annotations'
        traffic_annt = 'traffic_annotations'

        annotations = {}
        annotations['num_frames'] = int(tree.find("./meta/task/size").text)
        annotations['width'] = int(tree.find("./meta/task/original_size/width").text)
        annotations['height'] = int(tree.find("./meta/task/original_size/height").text)
        annotations[ped_annt] = {}
        annotations[traffic_annt] = {}
        tracks = tree.findall('./track')
        for t in tracks:
            boxes = t.findall('./box')
            obj_label = t.get('label')
            obj_id = boxes[0].find('./attribute[@name=\"id\"]').text
            #./ means start from current node which in this loop is boxes[0]

            if obj_label == 'pedestrian':
                annotations[ped_annt][obj_id] = {'frames': [], 'bbox': [], 'occlusion': []}
                annotations[ped_annt][obj_id]['behavior'] = {'gesture': [], 'look': [], 'action': [], 'cross': []}
                for b in boxes: # each box is for 1 frame: frame=100,101,102,....
                    # Exclude the annotations that are outside of the frame
                    if int(b.get('outside')) == 1:
                        continue
                    annotations[ped_annt][obj_id]['bbox'].append(
                        [float(b.get('xtl')), float(b.get('ytl')),
                         float(b.get('xbr')), float(b.get('ybr'))])
                    occ = self._map_text_to_scalar('occlusion', b.find('./attribute[@name=\"occlusion\"]').text)# 'occlusion': [0,0,1,1,2,...]
                    annotations[ped_annt][obj_id]['occlusion'].append(occ)
                    annotations[ped_annt][obj_id]['frames'].append(int(b.get('frame')))
                    for beh in annotations['ped_annotations'][obj_id]['behavior']:
                        # Read behavior tags for each frame and add to the database
                        annotations[ped_annt][obj_id]['behavior'][beh].append(
                            self._map_text_to_scalar(beh, b.find('./attribute[@name=\"' + beh + '\"]').text))

            else:
                obj_type = boxes[0].find('./attribute[@name=\"type\"]')
                if obj_type is not None:
                    obj_type = self._map_text_to_scalar(obj_label,
                                                        boxes[0].find('./attribute[@name=\"type\"]').text)

                annotations[traffic_annt][obj_id] = {'frames': [], 'bbox': [], 'occlusion': [],
                                                     'obj_class': obj_label,
                                                     'obj_type': obj_type,
                                                     'state': []}

                for b in boxes:
                    # Exclude the annotations that are outside of the frame
                    if int(b.get('outside')) == 1:
                        continue
                    annotations[traffic_annt][obj_id]['bbox'].append(
                        [float(b.get('xtl')), float(b.get('ytl')),
                         float(b.get('xbr')), float(b.get('ybr'))])
                    annotations[traffic_annt][obj_id]['occlusion'].append(int(b.get('occluded')))
                    annotations[traffic_annt][obj_id]['frames'].append(int(b.get('frame')))
                    if obj_label == 'traffic_light':
                        annotations[traffic_annt][obj_id]['state'].append(self._map_text_to_scalar('state',
                                                                                                    b.find(
                                                                                                        './attribute[@name=\"state\"]').text))
        return annotations

    def _get_ped_attributes(self, setid, vid): # General Pedestrian information like: male,adult,intention_prob,...
        """
        Generates a dictionary of attributes by parsing the video XML file
        :param setid: The set id
        :param vid: The video id
        :return: A dictionary of attributes
        """
        path_to_file = join(self._annotation_attributes_path, setid, vid + '_attributes.xml')
        tree = ET.parse(path_to_file)

        attributes = {}
        pedestrians = tree.findall("./pedestrian")
        for p in pedestrians:
            ped_id = p.get('id')
            attributes[ped_id] = {}
            for k, v in p.items():
                if 'id' in k:
                    continue
                try:
                    if k == 'intention_prob':
                        attributes[ped_id][k] = float(v)
                    else:
                        attributes[ped_id][k] = int(v)
                except ValueError:
                    attributes[ped_id][k] = self._map_text_to_scalar(k, v)

        return attributes

    def _get_vehicle_attributes(self, setid, vid):
        """
        Generates a dictionary of vehicle attributes by parsing the video XML file
        :param setid: The set id
        :param vid: The video id
        :return: A dictionary of vehicle attributes (obd sensor recording)
        """
        path_to_file = join(self._annotation_vehicle_path, setid, vid + '_obd.xml')
        tree = ET.parse(path_to_file)

        veh_attributes = {}
        frames = tree.findall("./frame")

        for f in frames:
            dict_vals = {k: float(v) for k, v in f.attrib.items() if k != 'id'}
            veh_attributes[int(f.get('id'))] = dict_vals

        return veh_attributes
    def generate_database(self):
            """
            Generates and saves a database of the PIE dataset by integrating
            all annotations from XML files into a single nested dictionary.

            Dictionary structure:
            'set_id'(str): {
                'vid_id'(str): {
                    'num_frames': int
                    'width': int
                    'height': int
                    'traffic_annotations'(str): {
                        'obj_id'(str): {
                            'frames': list(int)
                            'occlusion': list(int)
                            'bbox': list([x1, y1, x2, y2]) (float)
                            'obj_class': str,
                            'obj_type': str,    # only for traffic lights, vehicles, signs
                            'state': list(int)  # only for traffic lights
                    'ped_annotations'(str): {
                        'ped_id'(str): {
                            'frames': list(int)
                            'occlusion': list(int)
                            'bbox': list([x1, y1, x2, y2]) (float)
                            'behavior'(str): {
                                'action': list(int)
                                'gesture': list(int)
                                'cross': list(int)
                                'look': list(int)
                            'attributes'(str): {
                                'age': int
                                'id': str
                                'num_lanes': int
                                'crossing': int
                                'gender': int
                                'crossing_point': int
                                'critical_point': int
                                'exp_start_point': int
                                'intersection': int
                                'designated': int
                                'signalized': int
                                'traffic_direction': int
                                'group_size': int
                                'motion_direction': int
                    'vehicle_annotations'(str): {
                        'frame_id'(int): {
                            'longitude': float
                            'yaw': float
                            'pitch': float
                            'roll': float
                            'OBD_speed': float
                            'GPS_speed': float
                            'latitude': float
                            'heading_angle': float
                            'accX': float
                            'accY': float
                            'accZ': float
                            'gyroX': float
                            'gyroY': float
                            'gyroZ': float

            :return: A database dictionary

            ── Changes from the reference repo (pie_data.py) ─────────────────

            Change 1 -- tqdm progress bar on cold path (M7, by Hesam Mardani):
                The reference repo prints one line per video during XML parsing,
                producing hundreds of unstructured lines. This version builds a
                (set, video) work list first so the total denominator is known,
                then wraps it in tqdm for a single clean progress bar.
                Graceful fallback: if tqdm is not installed (_HAS_TQDM=False),
                behaviour is identical to the reference.

            Change 2 -- stale-cache guard (by Hesam Mardani):
                The reference repo returns pie_database.pkl unconditionally with
                no check that the on-disk annotation folder still matches what
                is inside the cache. This silently causes data loss when a new
                PIE set is downloaded AFTER the cache was built -- the new set
                is ignored with no warning.

                This guard fires a WARNING (never auto-deletes, never raises)
                when the sets found in the annotation folder differ from the
                sets baked into the cached dictionary.

                WHY isdir() + startswith('set'):
                    listdir() returns ALL entries in the folder, including stray
                    OS files such as desktop.ini (Windows) or .DS_Store (macOS).
                    Without this filter those files appear in expected_sets and
                    trigger a false-positive WARNING every time the database is
                    loaded, even when the cache is perfectly valid.
                    isdir() keeps only subdirectories; startswith('set') keeps
                    only PIE set folders (set01 … set06).

                Example scenario this guard catches:
                    1. Project starts: set01, set02, set03, set05 on disk.
                    2. generate_database() -> pie_database.pkl built with keys
                    {'set01', 'set02', 'set03', 'set05'}.
                    3. User downloads and extracts set04 (665 new train tracks).
                    4. generate_database() is called again -- WITHOUT this guard
                    the old cache is returned silently; set04's 665 tracks
                    are never visible to PIEIntentionDataset.
                    5. WITH this guard the output is:

                    WARNING: cached database does not match annotation folder.
                        Sets on disk but NOT in cache : ['set04']
                        → These sets will be IGNORED until the cache is rebuilt.
                        Cache file : .../data_cache/pie_database.pkl
                        Action     : delete this file AND all seq_*.pkl files
                                    in the same folder, then re-run to rebuild.
            """
            print('---------------------------------------------------------')
            print("Generating database for pie")

            cache_file = join(self.cache_path, 'pie_database.pkl')
            if isfile(cache_file) and not self._regen_database:
                with open(cache_file, 'rb') as fid:
                    try:
                        database = pickle.load(fid)
                    except Exception:
                        database = pickle.load(fid, encoding='bytes')
                print('pie annotations loaded from {}'.format(cache_file))

                # Added By Hesam Mardani: stale-cache guard. Warns (never
                # auto-deletes) when the on-disk annotation folder no longer
                # matches the sets this cached database was built from -- the
                # exact silent-staleness failure mode hit earlier with set03.
                #
                # isdir() + startswith('set') filters out stray OS files
                # (desktop.ini, .DS_Store) that would cause false positives
                # if included in the comparison.
                expected_sets = set(
                    d for d in sorted(listdir(self._annotation_path))
                    if isdir(join(self._annotation_path, d))
                    and d.startswith('set')
                )
                cached_sets = set(database.keys())

                if cached_sets != expected_sets:
                    missing = expected_sets - cached_sets
                    extra   = cached_sets   - expected_sets
                    print('---------------------------------------------------------')
                    print('WARNING: cached database does not match the annotation folder.')
                    if missing:
                        print(f'  Sets on disk but NOT in cache : {sorted(missing)}')
                        print(f'  → These sets will be IGNORED until the cache is rebuilt.')
                    if extra:
                        print(f'  Sets in cache but NOT on disk : {sorted(extra)}')
                        print(f'  → Annotation folder may have changed or moved.')
                    print()
                    print(f'  Cache file : {cache_file}')
                    print( '  Action     : delete this file AND all seq_*.pkl files')
                    print( '               in the same folder, then re-run to rebuild.')
                    print('---------------------------------------------------------')

                return database

            set_ids = sorted(listdir(self._annotation_path))

            # Build the (set, video) work list first so we can show total progress.
            work_list = []
            for setid in set_ids:
                video_ids = [
                    v.split('_annt.xml')[0]
                    for v in sorted(listdir(join(self._annotation_path, setid)))
                    if v.endswith("annt.xml")
                ]
                for vid in video_ids:
                    work_list.append((setid, vid))

            iterator = tqdm(work_list, desc='Parsing XML') if _HAS_TQDM else work_list

            database = {}
            for setid, vid in iterator:
                if setid not in database:
                    database[setid] = {}
                if not _HAS_TQDM:
                    print('Getting annotations for %s, %s' % (setid, vid))
                database[setid][vid] = self._get_annotations(setid, vid)
                vid_attributes = self._get_ped_attributes(setid, vid)
                database[setid][vid]['vehicle_annotations'] = \
                    self._get_vehicle_attributes(setid, vid)
                for ped in database[setid][vid]['ped_annotations']:
                    database[setid][vid]['ped_annotations'][ped]['attributes'] = \
                        vid_attributes[ped]

            with open(cache_file, 'wb') as fid:
                pickle.dump(database, fid, pickle.HIGHEST_PROTOCOL)
            print('The database is written to {}'.format(cache_file))

            return database
    def get_data_stats(self):
        """
        Generates statistics for the dataset
        """
        annotations = self.generate_database()

        set_count = len(annotations.keys())

        ped_count = 0
        ped_box_count = 0
        video_count = 0
        total_frames = 0
        age = {'child': 0, 'adult': 0, 'senior': 0}
        gender = {'male': 0, 'female': 0}
        signalized = {'n/a': 0, 'C': 0, 'S': 0, 'CS': 0}
        traffic_direction = {'OW': 0, 'TW': 0}
        intersection = {'midblock': 0, 'T': 0, 'T-right': 0, 'T-left': 0, 'four-way': 0}
        crossing = {'crossing': 0, 'not-crossing': 0, 'irrelevant': 0}

        traffic_obj_types = {'vehicle': {'car': 0, 'truck': 0, 'bus': 0, 'train': 0, 'bicycle': 0, 'bike': 0},
                             'sign': {'ped_blue': 0, 'ped_yellow': 0, 'ped_white': 0, 'ped_text': 0, 'stop_sign': 0,
                                      'bus_stop': 0, 'train_stop': 0, 'construction': 0, 'other': 0},
                             'traffic_light': {'regular': 0, 'transit': 0, 'pedestrian': 0},
                             'crosswalk': 0,
                             'transit_station': 0}
        traffic_box_count = {'vehicle': 0, 'traffic_light': 0, 'sign': 0, 'crosswalk': 0, 'transit_station': 0}
        for sid, vids in annotations.items():
            video_count += len(vids)
            for vid, annots in vids.items():
                total_frames += annots['num_frames']
                for trf_ids, trf_annots in annots['traffic_annotations'].items():
                    obj_class = trf_annots['obj_class']
                    traffic_box_count[obj_class] += len(trf_annots['frames'])
                    if obj_class in ['traffic_light', 'vehicle', 'sign']:
                        obj_type = trf_annots['obj_type']
                        traffic_obj_types[obj_class][self._map_scalar_to_text(obj_class, obj_type)] += 1
                    else:
                        traffic_obj_types[obj_class] += 1
                for ped_ids, ped_annots in annots['ped_annotations'].items():
                    ped_count += 1
                    ped_box_count += len(ped_annots['frames'])
                    age[self._map_scalar_to_text('age', ped_annots['attributes']['age'])] += 1
                    if self._map_scalar_to_text('crossing', ped_annots['attributes']['crossing']) == 'crossing':
                        crossing[self._map_scalar_to_text('crossing', ped_annots['attributes']['crossing'])] += 1
                    else:
                        if ped_annots['attributes']['intention_prob'] > 0.5:
                            crossing['not-crossing'] += 1
                        else:
                            crossing['irrelevant'] += 1
                    intersection[
                        self._map_scalar_to_text('intersection', ped_annots['attributes']['intersection'])] += 1
                    traffic_direction[self._map_scalar_to_text('traffic_direction',
                                                               ped_annots['attributes']['traffic_direction'])] += 1
                    signalized[self._map_scalar_to_text('signalized', ped_annots['attributes']['signalized'])] += 1
                    gender[self._map_scalar_to_text('gender', ped_annots['attributes']['gender'])] += 1

        print('---------------------------------------------------------')
        print("Number of sets: %d" % set_count)
        print("Number of videos: %d" % video_count)
        print("Number of annotated frames: %d" % total_frames)
        print("Number of pedestrians %d" % ped_count)
        print("age:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(age.items())))
        print("gender:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(gender.items())))
        print("signal:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(signalized.items())))
        print("traffic direction:\n",
              '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(traffic_direction.items())))
        print("crossing:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(crossing.items())))
        print("intersection:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(intersection.items())))
        print("Number of pedestrian bounding boxes: %d" % ped_box_count)
        print("Number of traffic objects")
        for trf_obj, values in sorted(traffic_obj_types.items()):
            if isinstance(values, dict):
                print(trf_obj + ':\n', '\n '.join('{}: {}'.format(k, v) for k, v in sorted(values.items())),
                      '\n total: ', sum(values.values()))
            else:
                print(trf_obj + ': %d' % values)
        print("Number of pedestrian bounding boxes:\n",
              '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(traffic_box_count.items())),
              '\n total: ', sum(traffic_box_count.values()))

    def balance_samples_count(self, seq_data, label_type, random_seed=42):
        """
        Balances the number of positive and negative samples by randomly sampling
        from the more represented samples. Only works for binary classes.
        :param seq_data: The sequence data to be balanced.
        :param label_type: The lable type based on which the balancing takes place.
        The label values must be binary, i.e. only 0, 1.
        :param random_seed: The seed for random number generator.
        :return: Balanced data sequence.
        """
        for lbl in seq_data[label_type]:
            for i in lbl:
                if i[0] not in [0, 1]:
                    raise Exception("The label values used for balancing must be"
                                    " either 0 or 1")

        # balances the number of positive and negative samples
        print('---------------------------------------------------------')
        print("Balancing the number of positive and negative intention samples")

        gt_labels = [gt[0] for gt in seq_data[label_type]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        new_seq_data = {}
        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
            return seq_data
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(random_seed)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]
            # update the data
            for k in seq_data:
                seq_data_k = seq_data[k]
                if not isinstance(seq_data[k], list):
                    new_seq_data[k] = seq_data[k]
                else:
                    new_seq_data[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in new_seq_data[label_type]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(new_seq_data[label_type]) - num_pos_samples))
        return new_seq_data

    # Process pedestrian ids
    def _get_pedestrian_ids(self):
        """
        Returns all pedestrian ids
        :return: A list of pedestrian ids
        """
        annotations = self.generate_database()
        pids = []
        for sid in sorted(annotations):
            for vid in sorted(annotations[sid]):
                pids.extend(annotations[sid][vid]['ped_annotations'].keys())
        return pids

        
    # In __init__, i added:
    # self._RANDOM_SEED = 42

    def _get_random_pedestrian_ids(self, image_set, ratios=None, val_data=True, regen_data=False):
        """
        Generates and saves a random pedestrian ids
        :param image_set: The data split to return
        :param ratios: The ratios to split the data. There should be 2 ratios (or 3 if val_data is true)
        and they should sum to 1. e.g. [0.4, 0.6], [0.3, 0.5, 0.2]
        :param val_data: Whether to generate validation data
        :param regen_data: Whether to overwrite the existing data, i.e. regenerate splits
        :return: The random sample split
        """
        assert image_set in ['train', 'test', 'val']
        cache_file = join(self.cache_path, "random_samples.pkl")
        if isfile(cache_file) and not regen_data:
            print("Random sample currently exists.\n Loading from %s" % cache_file)
            with open(cache_file, 'rb') as fid:
                try:
                    rand_samples = pickle.load(fid)
                except Exception:
                    rand_samples = pickle.load(fid, encoding='bytes')
                assert image_set in rand_samples
                if val_data:
                    assert len(rand_samples['ratios']) == 3
                if ratios is not None:
                    assert ratios == rand_samples['ratios']
                print('The ratios are {}'.format(rand_samples['ratios']))
                print("Number of %s tracks %d" % (image_set, len(rand_samples[image_set])))
                return rand_samples[image_set]

        if ratios is None:
            ratios = [0.5, 0.4, 0.1] if val_data else [0.5, 0.5]

        assert sum(ratios) > 0.999999
        assert len(ratios) == (3 if val_data else 2)

        print("################ Generating Random training/testing data ################")
        ped_ids = self._get_pedestrian_ids()
        print("Total number of tracks %d" % len(ped_ids))
        print('The ratios are {}'.format(ratios))

        sample_split = {'ratios': ratios}
        # CHANGE: explicit random_state for reproducibility.
        train_samples, test_samples = train_test_split(
            ped_ids, train_size=ratios[0], random_state=self._RANDOM_SEED
        )
        print("Number of train tracks %d" % len(train_samples))

        if val_data:
            test_samples, val_samples = train_test_split(
                test_samples,
                train_size=ratios[1] / sum(ratios[1:]),
                random_state=self._RANDOM_SEED,
            )
            print("Number of val tracks %d" % len(val_samples))
            sample_split['val'] = val_samples

        print("Number of test tracks %d" % len(test_samples))
        sample_split['train'] = train_samples
        sample_split['test']  = test_samples

        with open(cache_file, 'wb') as fid:
            pickle.dump(sample_split, fid, pickle.HIGHEST_PROTOCOL)
            print('pie {} samples written to {}'.format('random', cache_file))
        return sample_split[image_set]


    def _get_kfold_pedestrian_ids(self, image_set, num_folds=5, fold=1):
        """
        Generates kfold pedestrian ids
        :param image_set: Image set split
        :param num_folds: Number of folds
        :param fold: The given fold
        :return: List of pedestrian ids for the given fold
        """

        assert image_set in ['train', 'test']
        assert fold <= num_folds
        print("################ Generating %d fold data ################" % num_folds)
        cache_file = join(self.cache_path, "%d_fold_samples.pkl" % num_folds)

        if isfile(cache_file):
            print("Loading %d-fold data from %s" % (num_folds, cache_file))
            with open(cache_file, 'rb') as fid:
                try:
                    fold_idx = pickle.load(fid)
                except Exception:
                    fold_idx = pickle.load(fid, encoding='bytes')
        else:
            ped_ids = self._get_pedestrian_ids()
            # CHANGE: explicit random_state for reproducibility.
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=self._RANDOM_SEED)
            fold_idx = {'pid': ped_ids}
            for count, (train_index, test_index) in enumerate(kf.split(ped_ids), start=1):
                fold_idx[count] = {'train': train_index.tolist(), 'test': test_index.tolist()}
            with open(cache_file, 'wb') as fid:
                pickle.dump(fold_idx, fid, pickle.HIGHEST_PROTOCOL)
                print('pie {}-fold samples written to {}'.format(num_folds, cache_file))

        print("Number of %s tracks %d" % (image_set, len(fold_idx[fold][image_set])))
        kfold_ids = [fold_idx['pid'][i] for i in fold_idx[fold][image_set]]
        return kfold_ids

    # Trajectory data generation
    def _get_data_ids(self, image_set, params):
        """
        Generates set ids and ped ids (if needed) for processing
        :param image_set: Image-set to generate data
        :param params: Data generation params
        :return: Set and pedestrian ids
        """
        _pids = None
        if params['data_split_type'] == 'default':
            set_ids = self._get_image_set_ids(image_set)
        else:
            set_ids = self._get_image_set_ids('all')
        if params['data_split_type'] == 'random':
            _pids = self._get_random_pedestrian_ids(image_set, **params['random_params'])
        elif params['data_split_type'] == 'kfold':
            _pids = self._get_kfold_pedestrian_ids(image_set, **params['kfold_params'])

        return set_ids, _pids

    def _height_check(self, height_rng, frame_ids, boxes, images, occlusion):
        """
        Checks whether the bounding boxes are within a given height limit. If not, it
        will adjust the length of bounding boxes in data sequences accordingly
        :param height_rng: Height limit [lower, higher]
        :param boxes: List of bounding boxes
        :param fields: A list of data field lists
        :return: The adjusted data sequences

        Filter bounding boxes by HEIGHT (y2 - y1), not width.
        Signature and return structure are unchanged from the original.

        bbox format: [x1, y1, x2, y2]
        width  = abs(b[0] - b[2])
        height = abs(b[1] - b[3])   <-- this is what we want to filter on
        """
        imgs, box, frames, occ = [], [], [], []
        for i, b in enumerate(boxes):
            bbox_height = abs(b[1] - b[3])   # fixed: y-axis, not x-axis
            if height_rng[0] <= bbox_height <= height_rng[1]:
                box.append(b)
                imgs.append(images[i])
                frames.append(frame_ids[i])
                occ.append(occlusion[i])
        return imgs, box, frames, occ


    def _height_check_v2(self, height_rng, boxes, *fields):
        """
        Checks whether the bounding boxes are within a given height limit. If not, it
        will adjust the length of bounding boxes in data sequences accordingly
        :param height_rng: Height limit [lower, higher]
        :param boxes: List of bounding boxes
        :param fields: A list of data field lists
        :return: The adjusted data sequences

        """
        assert len(boxes[0]) == 4, "First entry to the function should be bounding boxes"
        new_data = [[] for _ in range(len(fields) + 1)]
        for i, b in enumerate(boxes):
            bbox_height = abs(b[1] - b[3])   # fixed: y-axis, not x-axis
            if height_rng[0] <= bbox_height <= height_rng[1]:
                new_data[0].append(b)
                for j in range(1, len(new_data)):
                    new_data[j].append(fields[j - 1][i])
        return new_data
        
    def _get_center(self, box):
        """
        Calculates the center coordinate of a bounding box
        :param box: Bounding box coordinates
        :return: The center coordinate
        """
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


    def generate_data_trajectory_sequence(self, image_set, **opts):
        """
        Generates pedestrian tracks
        :param image_set: the split set to produce for. Options are train, test, val.
        :param opts:
                'fstride': Frequency of sampling from the data.
                'height_rng': The height range of pedestrians to use.
                'squarify_ratio': The width/height ratio of bounding boxes. A value between (0,1]. 0 the original
                                        ratio is used.
                'data_split_type': How to split the data. Options: 'default', predefined sets, 'random', randomly split the data,
                                        and 'kfold', k-fold data split (NOTE: only train/test splits).
                'seq_type': Sequence type to generate. Options: 'trajectory', generates tracks, 'crossing', generates
                                  tracks up to 'crossing_point', 'intention' generates tracks similar to human experiments
                'min_track_size': Min track length allowable.
                'random_params: Parameters for random data split generation. (see _get_random_pedestrian_ids)
                'kfold_params: Parameters for kfold split generation. (see _get_kfold_pedestrian_ids)
        :return: Sequence data

        Disk-cached version:

        Cache key = MD5( image_set + canonical-JSON(merged_params) ).
        Cache file = <cache_path>/seq_<image_set>_<hash>.pkl
        """

        params = {
            'fstride':          1,
            'sample_type':      'all',
            'height_rng':       [0, float('inf')],
            'squarify_ratio':   0,
            'data_split_type':  'default',
            'seq_type':         'intention',
            'min_track_size':   15,
            'random_params':    {'ratios': None, 'val_data': True, 'regen_data': False},
            'kfold_params':     {'num_folds': 5, 'fold': 1},
        }
        params.update(opts)
        # If a split regeneration was requested, do it NOW, before the cache key
        # is computed. Otherwise a cache hit below would return early and the
        # split file would never be rewritten. After regenerating, flip
        # regen_data to False so the rest of the function just loads the fresh file.
        if params['data_split_type'] == 'random' and params['random_params'].get('regen_data'):
            self._get_random_pedestrian_ids(image_set, **params['random_params'])
            params['random_params'] = {**params['random_params'], 'regen_data': False}

        # Build a stable cache key. float('inf') is not JSON-serializable by default;
        #  allow_nan=True (default) and replace inf with a sentinel string.
        def _stable_repr(value):
            if isinstance(value, float) and value == float('inf'):
                return '__inf__'
            return value

        # Module level (above the class): only params that actually change the
        # generated tracks belong in the cache key. If a _get_* method starts
        # consuming a new param, add it here.
        _SEQ_KEY_PARAMS = ('fstride', 'sample_type', 'height_rng', 'squarify_ratio',
                        'data_split_type', 'seq_type', 'min_track_size',
                        'random_params', 'kfold_params')
        key_params = {k: params[k] for k in _SEQ_KEY_PARAMS}

        # regen_data controls whether the split file is rewritten, not what the
        # tracks contain (the split content is already covered by split_fp below).
        key_params['random_params'] = {
            k: v for k, v in key_params['random_params'].items() if k != 'regen_data'}

        canonical = json.dumps(
            {k: _stable_repr(v) if not isinstance(v, dict)
                else {kk: _stable_repr(vv) for kk, vv in v.items()}
            for k, v in key_params.items()},
            sort_keys=True,
            default=str,
        )
        split_fp = None
        if params['data_split_type'] == 'random':
            split_file = join(self.cache_path, 'random_samples.pkl')
        elif params['data_split_type'] == 'kfold':
            split_file = join(self.cache_path, "%d_fold_samples.pkl" % params['kfold_params']['num_folds'])
        else:
            split_file = None

        if split_file is not None and isfile(split_file):
            with open(split_file, 'rb') as f:
                split_fp = hashlib.md5(f.read()).hexdigest()[:8]

        key_input = (image_set + '|' + canonical + '|' + str(split_fp) + '|intention_behavior_v2').encode('utf-8')
        key       = hashlib.md5(key_input).hexdigest()[:16]
        cache_file = join(self.cache_path, f'seq_{image_set}_{key}.pkl')


        if isfile(cache_file):
            print('---------------------------------------------------------')
            print(f'Loading cached sequence data from {cache_file}')
            with open(cache_file, 'rb') as fid:
                return pickle.load(fid)

        # Cold path -- original logic, unchanged
        print('---------------------------------------------------------')
        print("Generating trajectory sequence data")
        self._print_dict(params)
        annot_database = self.generate_database()
        if   params['seq_type'] == 'trajectory': sequence_data = self._get_trajectories(image_set, annot_database, **params)
        elif params['seq_type'] == 'crossing':   sequence_data = self._get_crossing    (image_set, annot_database, **params)
        elif params['seq_type'] == 'intention':  sequence_data = self._get_intention   (image_set, annot_database, **params)
        elif params['seq_type'] == 'all':        sequence_data = self._get_all         (image_set, annot_database, **params)
        else:
            raise Exception("Wrong sequence type is selected.")

        # Write cache for next time
        with open(cache_file, 'wb') as fid:
            pickle.dump(sequence_data, fid, pickle.HIGHEST_PROTOCOL)
        print(f'Sequence data cached to {cache_file}')

        return sequence_data

    def _get_trajectories(self, image_set, annotations, **params):
        """
        Generates trajectory data.
        :param image_set: Data split to use
        :param annotations: Annotations database
        :param params: Parameters to generate data (see generade_database)
        :return: A dictionary of trajectories
        """
        print('---------------------------------------------------------')
        print("Generating trajectory data")

        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        intent_seq = []
        obds_seq, gpss_seq, head_ang_seq, gpsc_seq, yrp_seq = [], [], [], [], []

        set_ids, _pids = self._get_data_ids(image_set, params)

        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                img_width = annotations[sid][vid]['width']
                pid_annots = annotations[sid][vid]['ped_annotations']
                vid_annots = annotations[sid][vid]['vehicle_annotations']
                for pid in sorted(pid_annots):
                    if params['data_split_type'] != 'default' and pid not in _pids:
                        continue
                    num_pedestrians += 1
                    frame_ids = pid_annots[pid]['frames']
                    boxes = pid_annots[pid]['bbox']
                    images = [self._get_image_path(sid, vid, f) for f in frame_ids]
                    occlusions = pid_annots[pid]['occlusion']

                    if height_rng[0] > 0 or height_rng[1] < float('inf'):
                        images, boxes, frame_ids, occlusions = self._height_check(height_rng,
                                                                                  frame_ids, boxes,
                                                                                  images, occlusions)

                    if len(boxes) / seq_stride < params['min_track_size']:  # max_obs_size: #90 + 45
                        continue

                    if sq_ratio:
                        boxes = [squarify(b, sq_ratio, img_width) for b in boxes]

                    image_seq.append(images[::seq_stride])
                    box_seq.append(boxes[::seq_stride])
                    center_seq.append([self._get_center(b) for b in boxes][::seq_stride])
                    occ_seq.append(occlusions[::seq_stride])

                    ped_ids = [[pid]] * len(boxes)
                    pids_seq.append(ped_ids[::seq_stride])

                    intent = [[pid_annots[pid]['attributes']['intention_prob']]] * len(boxes)
                    intent_seq.append(intent[::seq_stride])

                    gpsc_seq.append([(vid_annots[i]['latitude'], vid_annots[i]['longitude'])
                                     for i in frame_ids][::seq_stride])
                    obds_seq.append([[vid_annots[i]['OBD_speed']] for i in frame_ids][::seq_stride])
                    gpss_seq.append([[vid_annots[i]['GPS_speed']] for i in frame_ids][::seq_stride])
                    head_ang_seq.append([[vid_annots[i]['heading_angle']] for i in frame_ids][::seq_stride])
                    yrp_seq.append([(vid_annots[i]['yaw'], vid_annots[i]['roll'], vid_annots[i]['pitch'])
                                    for i in frame_ids][::seq_stride])

        print('Subset: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        return {'image': image_seq,
                'pid': pids_seq,
                'bbox': box_seq,
                'center': center_seq,
                'occlusion': occ_seq,
                'obd_speed': obds_seq,
                'gps_speed': gpss_seq,
                'heading_angle': head_ang_seq,
                'gps_coord': gpsc_seq,
                'yrp': yrp_seq,
                'intention_prob': intent_seq}

    def _get_crossing(self, image_set, annotations, **params):
        """
        Generates crossing data.
        :param image_set: Data split to use
        :param annotations: Annotations database
        :param params: Parameters to generate data (see generade_database)
        :return: A dictionary of trajectories
        """

        print('---------------------------------------------------------')
        print("Generating crossing data")

        print('Excluding tracks with less than {}'.format(params['min_track_size']))
        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        intent_seq = []
        obds_seq, gpss_seq, head_ang_seq, gpsc_seq, yrp_seq, acc_seq, gyro_seq = [], [], [], [], [],[], []
        activities = []
        act_seq, cross_seq, look_seq, gest_seq = [], [], [], []
        set_ids, _pids = self._get_data_ids(image_set, params)

        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                img_width = annotations[sid][vid]['width']
                pid_annots = annotations[sid][vid]['ped_annotations']
                vid_annots = annotations[sid][vid]['vehicle_annotations']
                for pid in sorted(pid_annots):
                    if params['data_split_type'] != 'default' and pid not in _pids:
                        continue
                    num_pedestrians += 1

                    frame_ids = pid_annots[pid]['frames']
                    event_frame = pid_annots[pid]['attributes']['crossing_point']
                    end_idx = frame_ids.index(event_frame)
                    boxes = pid_annots[pid]['bbox'][:end_idx + 1]
                    frame_ids = frame_ids[: end_idx + 1]
                    images = [self._get_image_path(sid, vid, f) for f in frame_ids]
                    occlusions = pid_annots[pid]['occlusion'][:end_idx + 1]

                    actions = pid_annots[pid]['behavior']['action'][:end_idx + 1]
                    crosses = pid_annots[pid]['behavior']['cross'][:end_idx + 1]
                    looks = pid_annots[pid]['behavior']['look'][:end_idx + 1]
                    gestures = pid_annots[pid]['behavior']['gesture'][:end_idx + 1]

                    if height_rng[0] > 0 or height_rng[1] < float('inf'):
                        boxes, frame_ids, images, occlusions, actions, crosses, looks, \
                        gestures = self._height_check_v2(height_rng,
                                                         boxes, frame_ids,
                                                         images, occlusions,
                                                         actions, crosses, looks,
                                                         gestures)

                    if len(boxes) / seq_stride < params['min_track_size']:
                        continue
                    if sq_ratio:
                        boxes = [squarify(b, sq_ratio, img_width) for b in boxes]

                    image_seq.append(images[::seq_stride])
                    box_seq.append(boxes[::seq_stride])
                    center_seq.append([self._get_center(b) for b in boxes][::seq_stride])
                    occ_seq.append([[o] for o in occlusions][::seq_stride])

                    ped_ids = [[pid]] * len(boxes)
                    pids_seq.append(ped_ids[::seq_stride])

                    intent = [[pid_annots[pid]['attributes']['intention_prob']]] * len(boxes)
                    intent_seq.append(intent[::seq_stride])

                    acts = [[int(pid_annots[pid]['attributes']['crossing'] > 0)]] * len(boxes)
                    activities.append(acts[::seq_stride])

                    act_seq.append([[a] for a in actions][::seq_stride])
                    cross_seq.append([[c] for c in crosses][::seq_stride])
                    look_seq.append([[l] for l in looks][::seq_stride])
                    gest_seq.append([[g] for g in gestures][::seq_stride])

                    gpsc_seq.append([(vid_annots[i]['latitude'], vid_annots[i]['longitude'])
                                     for i in frame_ids][::seq_stride])
                    acc_seq.append([(vid_annots[i]['accX'], vid_annots[i]['accY'], vid_annots[i]['accZ'])
                                     for i in frame_ids][::seq_stride])
                    gyro_seq.append([(vid_annots[i]['gyroX'], vid_annots[i]['gyroY'], vid_annots[i]['gyroZ'])
                                  for i in frame_ids][::seq_stride])
                    obds_seq.append([[vid_annots[i]['OBD_speed']] for i in frame_ids][::seq_stride])
                    gpss_seq.append([[vid_annots[i]['GPS_speed']] for i in frame_ids][::seq_stride])
                    head_ang_seq.append([[vid_annots[i]['heading_angle']] for i in frame_ids][::seq_stride])
                    yrp_seq.append([(vid_annots[i]['yaw'], vid_annots[i]['roll'], vid_annots[i]['pitch'])
                                    for i in frame_ids][::seq_stride])

        print('Subset: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        return {'image': image_seq,
                'pid': pids_seq,
                'bbox': box_seq,
                'center': center_seq,
                'occlusion': occ_seq,
                'action': act_seq,
                'gesture': gest_seq,
                'cross': cross_seq,
                'look': look_seq,
                'gyro': gyro_seq,
                'acc': acc_seq,
                'obd_speed': obds_seq,
                'gps_speed': gpss_seq,
                'heading_angle': head_ang_seq,
                'gps_coord': gpsc_seq,
                'yrp': yrp_seq,
                'intention_prob': intent_seq,
                'activities': activities,
                'image_dimension': self._get_dim()}

    def _get_intention(self, image_set, annotations, **params):
        """Intention sequences. Per-frame behavior names come from the XML."""
        print('---------------------------------------------------------')
        print("Generating intention data")

        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        intention_prob, intention_binary = [], []
        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        behavior_seq = None
        set_ids, _pids = self._get_data_ids(image_set, params)

        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                img_width = annotations[sid][vid]['width']
                pid_annots = annotations[sid][vid]['ped_annotations']
                for pid in sorted(pid_annots):
                    if params['data_split_type'] != 'default' and pid not in _pids:
                        continue
                    num_pedestrians += 1
                    exp_start_frame = pid_annots[pid]['attributes']['exp_start_point']
                    critical_frame = pid_annots[pid]['attributes']['critical_point']
                    frames = pid_annots[pid]['frames']

                    start_idx = frames.index(exp_start_frame)
                    end_idx = frames.index(critical_frame)

                    boxes = pid_annots[pid]['bbox'][start_idx:end_idx + 1]
                    frame_ids = frames[start_idx:end_idx + 1]
                    images = [self._get_image_path(sid, vid, f) for f in frame_ids]
                    occlusions = pid_annots[pid]['occlusion'][start_idx:end_idx + 1]
                    behavior = {
                        name: values[start_idx:end_idx + 1]
                        for name, values in pid_annots[pid]['behavior'].items()
                    }
                    if behavior_seq is None:
                        behavior_seq = {name: [] for name in behavior}

                    if height_rng[0] > 0 or height_rng[1] < float('inf'):
                        images, boxes, frame_ids, occlusions = self._height_check(
                            height_rng, frame_ids, boxes, images, occlusions)
                        keep = set(frame_ids)
                        for name in behavior:
                            behavior[name] = [
                                value for value, frame in zip(behavior[name], frames[start_idx:end_idx + 1])
                                if frame in keep
                            ]

                    if len(boxes) / seq_stride < params['min_track_size']:
                        continue

                    if sq_ratio:
                        boxes = [squarify(b, sq_ratio, img_width) for b in boxes]

                    int_prob = [[pid_annots[pid]['attributes']['intention_prob']]] * len(boxes)
                    int_bin = [[int(pid_annots[pid]['attributes']['intention_prob'] > 0.5)]] * len(boxes)

                    image_seq.append(images[::seq_stride])
                    box_seq.append(boxes[::seq_stride])
                    center_seq.append([self._get_center(b) for b in boxes][::seq_stride])
                    occ_seq.append([[o] for o in occlusions][::seq_stride])
                    for name in behavior_seq:
                        behavior_seq[name].append([[v] for v in behavior[name]][::seq_stride])

                    intention_prob.append(int_prob[::seq_stride])
                    intention_binary.append(int_bin[::seq_stride])
                    pids_seq.append(([[pid]] * len(boxes))[::seq_stride])

        print('Subset: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        data = {'image': image_seq,
                'bbox': box_seq,
                'center': center_seq,
                'occlusion': occ_seq,
                'intention_prob': intention_prob,
                'intention_binary': intention_binary,
                'ped_id': pids_seq}
        data.update(behavior_seq or {})
        return data
    def _get_all(self, image_set, annotations, **params):
        """
        Generates crossing data.
        :param image_set: Data split to use
        :param annotations: Annotations database
        :param params: Parameters to generate data (see generade_database)
        :return: A dictionary of trajectories
        """

        print('---------------------------------------------------------')
        print("Generating all data")
        num_pedestrians = 0
        image_seq, pids_seq = [], []
        traffic_seq = []
        box_seq, center_seq, occ_seq = [], [], []
        intent_seq = []
        signalized_seq = []
        obds_seq, gpss_seq, head_ang_seq, gpsc_seq, yrp_seq, acc_seq, gyro_seq = [], [], [], [], [],[], []
        activities = []
        action_seq = []
        gesture_seq = []
        motion_dir_seq = []
        look_seq = []
        cross_seq = []
        event_frames = []
        event_frames_idx = []
        intent_frames_idx = []
        set_ids, _pids = self._get_data_ids(image_set, params)
        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                pid_annots = annotations[sid][vid]['ped_annotations']
                vid_annots = annotations[sid][vid]['vehicle_annotations']
                traffic_annotations = annotations[sid][vid]['traffic_annotations']
                traffic_annots = {}
                # Create a empty traffic dic based on the ids of annotated frames
                for k in vid_annots:
                    traffic_annots[k] = {'ped_sign': 0, 'ped_crossing': 0, 'traffic_light': 0,
                                         'stop_sign': 0, 'traffic_direction': 0}
                # import pdb
                for k, v in traffic_annotations.items():
                    obj_class = v['obj_class']
                    if obj_class in ['crosswalk', 'traffic_light', 'sign']:
                        for f_idx, f in enumerate(v['frames']):
                            if obj_class == 'sign':
                                if v['obj_type'] < 4:
                                    traffic_annots[f]['ped_sign'] = 1
                                elif v['obj_type'] == 4:
                                    traffic_annots[f]['stop_sign'] = 1
                            if obj_class == 'traffic_light':
                                           traffic_annots[f]['traffic_light'] = v['state'][f_idx] 

                            if obj_class == 'crosswalk':
                                traffic_annots[f]['ped_crossing'] = 1
                for pid in sorted(pid_annots):
                    if params['data_split_type'] != 'default' and pid not in _pids:
                        continue
                    num_pedestrians += 1
                    frame_ids = pid_annots[pid]['frames']

                    # Intnetion indicies
                    exp_start_frame = pid_annots[pid]['attributes']['exp_start_point']
                    critical_frame = pid_annots[pid]['attributes']['critical_point']
                    int_start_idx = frame_ids.index(exp_start_frame)
                    int_end_idx = frame_ids.index(critical_frame)
                    intent_frames_idx.append((int_start_idx, int_end_idx))

                    # Crossing events
                    event_frame = pid_annots[pid]['attributes']['crossing_point']
                    event_frames.append(event_frame)
                    end_idx = frame_ids.index(event_frame)
                    event_frames_idx.append(end_idx)
                    boxes = pid_annots[pid]['bbox']
                    images = [self._get_image_path(sid, vid, f) for f in frame_ids]
                    occlusions = pid_annots[pid]['occlusion']
                    action = pid_annots[pid]['behavior']['action']
                    look = pid_annots[pid]['behavior']['look']
                    cross = pid_annots[pid]['behavior']['cross']
                    gesture = pid_annots[pid]['behavior']['gesture']
                    image_seq.append(images)
                    box_seq.append(boxes)

                    center_seq.append([self._get_center(b) for b in boxes])
                    occ_seq.append(occlusions)
                    action_seq.append([[a] for a in action])
                    look_seq.append([[l] for l in look])
                    cross_seq.append([[c] for c in cross])
                    gesture_seq.append([[g] for g in gesture])

                    ped_ids = [[pid]] * len(boxes)
                    pids_seq.append(ped_ids)
                    intent = [[pid_annots[pid]['attributes']['intention_prob']]] * len(boxes)
                    intent_seq.append(intent)
                    signalized = [[pid_annots[pid]['attributes']['signalized']]] * len(boxes)
                    signalized_seq.append(signalized)
                    acts = [[int(pid_annots[pid]['attributes']['crossing'] > 0)]] * len(boxes)
                    activities.append(acts)

                    acc_seq.append([(vid_annots[i]['accX'], vid_annots[i]['accY'], vid_annots[i]['accZ'])
                                     for i in frame_ids])
                    gyro_seq.append([(vid_annots[i]['gyroX'], vid_annots[i]['gyroY'], vid_annots[i]['gyroZ'])
                                  for i in frame_ids])
                    # Removed outer list
                    gpsc_seq.append([(vid_annots[i]['latitude'], vid_annots[i]['longitude'])
                                     for i in frame_ids])
                    obds_seq.append([[vid_annots[i]['OBD_speed']] for i in frame_ids])
                    gpss_seq.append([[vid_annots[i]['GPS_speed']] for i in frame_ids])
                    head_ang_seq.append([[vid_annots[i]['heading_angle']] for i in frame_ids])
                    # Removed outer list
                    yrp_seq.append([(vid_annots[i]['yaw'], vid_annots[i]['roll'], vid_annots[i]['pitch'])
                                    for i in frame_ids])
                    road_type = {'road_type': 0,
                                 'num_lanes': pid_annots[pid]['attributes'].get('num_lanes'), 
                                 'traffic_direction': pid_annots[pid]['attributes'].get('traffic_direction')}
                    traffic_seq.append([[{**traffic_annots[i], **road_type}]
                                        for i in frame_ids])
        print('Subset: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        return {'image': image_seq,
                'pid': pids_seq,
                'bbox': box_seq,
                'actions': action_seq,
                'looks': look_seq,
                'cross': cross_seq,
                'gesture': gesture_seq,
                'signalized': signalized_seq,
                'center': center_seq,
                'occlusion': occ_seq,
                'obd_speed': obds_seq,
                'gps_speed': gpss_seq,
                'gyro': gyro_seq,
                'acc': acc_seq,
                'heading_angle': head_ang_seq,
                'gps_coord': gpsc_seq,
                'yrp': yrp_seq,
                'event_frames': event_frames,
                'event_frames_idx': event_frames_idx,
                'intent_frames_idx': intent_frames_idx,
                'intention_prob': intent_seq,
                'activities': activities,
                'traffic': traffic_seq,
                'image_dimension': self._get_dim()}
