# utils.py
import sys
import PIL
from PIL import Image
import numpy as np


# Single source of truth for the PIE dataset's fixed frame size.
# Previously hardcoded independently in three places (PIE._get_width/
# _get_height/_get_dim in pie_data_Hesam.py, PIEIntent._normalize_decoder_bbox
# in pie_intent.py, and the img_w/img_h defaults in augmentation.py) --
# the exact same duplication pattern discovered earlier with squarify.
PIE_IMG_WIDTH = 1920
PIE_IMG_HEIGHT = 1080


def bbox_sanity_check(bbox , img):
    img_width ,img_height = img.size  
    if bbox[0]< 0:
        bbox[0]=0.0
    if bbox[2]>=img_width:
        bbox[2] = img_width-1
    if bbox[1]<0:
        bbox[1] = 0.0
    if bbox[3]>=img_height:
        bbox[3] = img_height - 1
    return bbox

def squarify(bbox , squarify_ratio , img_width):
    bbox = list(bbox) 
    width = abs(bbox[2]- bbox[0])
    height = abs(bbox[3]-bbox[1])
    width_change = squarify_ratio * height - width

    bbox[0] = bbox[0] - width_change/2
    bbox[2] = bbox[2] + width_change/2   # So the width increases by exactly width_change

    if bbox[0]<0:
        bbox[0] = 0
    if bbox[2]>img_width:
        bbox[0] = bbox[0] - bbox[2] + img_width
        bbox[2] = img_width
    return bbox

def img_pad(img, mode = 'warp' , size = 224):
    assert (mode in ['warp', 'pad_same' , 'same','pad_same','pad_resize','pad_fit']),'Pad mode %s is invalid.'%mode
    image = img.copy()
    if mode == 'warp':
        warpped_image = image.resize((size,size), PIL.Image.BILINEAR)
        return warpped_image
    elif mode == 'same':
        return image
    elif mode in ['pad_same','pad_resize','pad_fit']:
        img_size = image.size
        ratio = float(size)/max(img_size)
        if mode == 'pad_resize' or (mode == 'pad_fit' and (img_size[0]>size or img_size[1]>size)):
            img_size = tuple([int(img_size[0]*ratio) ,int(img_size[1]*ratio)])
            image = image.resize(img_size , PIL.Image.NEAREST)
        padded_image = PIL.Image.new('RGB',(size,size))
        padded_image.paste(image,((size-img_size[0])//2 , (size - img_size[1])//2))
        return padded_image
    

def update_progress(progress):
    barlength = 20
    status = ""
    if isinstance(progress,int):
        progress = float(progress)
    block = int(round(barlength*progress))
    text = f"\r[{'#'*block + '-'*(barlength-block)}] {progress*100:.1f} % {status}"
    text = text.ljust(80)  # pad to a fixed width so old trailing chars get erased

    sys.stdout.write(text)
    sys.stdout.flush()



def jitter_bbox(bbox,img_path,mode,ratio):
    assert (mode in ['same','enlarge','move','random_enlarge','random_move']), \
        'mode %s is invalid.' % mode
    if mode =='same':
        return bbox
    img = Image.open(img_path)  
    img_width , img_height = img.size 

    if mode in ['random_enlarge','enlarge']:
        jitter_ratio = abs(ratio)
    else:
        jitter_ratio = ratio
    if mode == 'random_enlarge':
        jitter_ratio = np.random.random_sample() * jitter_ratio

    elif mode == 'random_move':
        # for ratio between (-jitter_ratio, jitter_ratio)
        # for sampling the formula is [a,b), b > a,
        # random_sample * (b-a) + a
        jitter_ratio = np.random.random_sample()*2*jitter_ratio - jitter_ratio
    
    jit_boxes = []
    for b in bbox:
        b = list(b)
        bbox_width = b[2]-b[0]
        bbox_height = b[3] - b[1]

        # here jitter_ratio is applied:
        width_change = bbox_width * jitter_ratio
        height_change = bbox_height * jitter_ratio

        # change must be same so the bbox be square_like and reduce the distortion
        if width_change <height_change:
            height_change = width_change
        else:
            width_change = height_change
        # ['same','enlarge','move','random_enlarge','random_move']
        '''
        did augmentation helped? so we compare:
        - same (no augmentation)
        - jittered (enlarge/move/random)'''
        if mode in ['enlarge' , 'random_enlarge']:
            b[0] = b[0] - width_change//2
            b[1] = b[1] - height_change//2
        else:
            b[0] = b[0] + width_change//2
            b[1] = b[1] + height_change//2
        
        b[2] = b[2] + width_change//2
        b[3] = b[3] + height_change//2

        # sanity chech for the bbox after jittering or being the same
        b = bbox_sanity_check(b,img)
        jit_boxes.append(b)
    return jit_boxes


def concat_data(data, data_type):
    """Concatenates the requested data_type keys along the feature axis.

    Moved from PIEIntent.concat_data -- pure array-assembly, no
    dependency on model architecture, annotations, or `self` of any kind.
    :param data: A dict mapping type names (e.g. 'bbox', 'intention_binary')
        to lists of per-track arrays, as produced by get_tracks().
    :param data_type: The list of keys from `data` to concatenate, in order,
        along axis=2 (e.g. data_opts['encoder_input_type']).
    :return: A single concatenated np.array, or [] if data_type is empty.
    """
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


def normalize_decoder_bbox(decoder_input, decoder_input_type,
                            img_w=PIE_IMG_WIDTH, img_h=PIE_IMG_HEIGHT,
                            enabled=True):
    """Min-max scales bbox decoder_input to roughly [0,1] by dividing
    by the PIE frame size, unless `enabled` is False.

    Moved from PIEIntent._normalize_decoder_bbox. The only previous use
    of `self` in that method was reading a simple on/off config flag
    (`getattr(self, 'normalize_decoder_bbox', True)`) -- that flag is now
    the explicit `enabled` parameter, passed in by the caller instead of
    being read implicitly from an instance attribute.

    IMPORTANT: this function is called from three places --
    PIEIntent.get_train_val_data, PIEIntent.get_test_data, and
    PIEIntentExperiment._process_test_group. All three must pass the same
    `enabled` value (typically `getattr(self, 'normalize_decoder_bbox', True)`
    from the PIEIntent instance) or train/eval paths will silently
    normalize inconsistently -- the same class of bug previously found
    with duplicated, drifted copies of squarify().
    """
    if not enabled:
        return decoder_input
    if 'bbox' not in decoder_input_type:
        return decoder_input
    decoder_input = np.asarray(decoder_input, dtype=np.float32).copy()
    decoder_input[..., [0, 2]] /= float(img_w)
    decoder_input[..., [1, 3]] /= float(img_h)
    return decoder_input
