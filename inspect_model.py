import sys

from keras.models import load_model
from feature_routing import CUSTOM_OBJECTS

model = load_model(sys.argv[1], compile=False, custom_objects=CUSTOM_OBJECTS)
for layer in model.layers:
    shapes = [tuple(w.shape) for w in layer.get_weights()]
    print(f"{layer.name:<24}{type(layer).__name__:<24}{shapes}")