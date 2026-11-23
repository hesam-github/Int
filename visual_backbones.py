"""Frozen image extractors. Cached map is always (7, 7, C), C native."""
import numpy as np

def preprocess_imagenet_torch(image_rgb):
    """img_to_array is RGB in 0..255. ViT uses mean/std, not VGG caffe."""
    x = np.asarray(image_rgb, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return (x - mean) / std

def build_extractor(name):
    """Returns (predict_on_batch, preprocess, cache_token, channels)."""
    name = str(name).lower()
    if name == "vgg16":
        from keras.applications import vgg16
        model = vgg16.VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
        return model.predict_on_batch, vgg16.preprocess_input, "vgg16", 512
    if name == "resnet50":
        from keras.applications import resnet50
        model = resnet50.ResNet50(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
        return model.predict_on_batch, resnet50.preprocess_input, "resnet50", 2048
    if name == "vit_b16":
        import torch
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        net.eval()

        def encode(batch):
            t = torch.from_numpy(np.transpose(batch, (0, 3, 1, 2))).float()
            with torch.no_grad():
                x = net._process_input(t)
                cls = net.class_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = net.encoder(x)[:, 1:, :]
                b, n, c = x.shape
                h = int(n ** 0.5)
                x = x.reshape(b, h, h, c).permute(0, 3, 1, 2)
                x = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))
            return np.asarray(x.permute(0, 2, 3, 1).cpu().numpy(), dtype=np.float32)

        return encode, preprocess_imagenet_torch, "vit_b16", 768
    raise ValueError("backbone must be vgg16, resnet50, or vit_b16")