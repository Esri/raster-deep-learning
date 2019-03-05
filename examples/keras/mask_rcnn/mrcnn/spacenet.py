import os
import sys

sys.path.append(os.path.dirname(__file__))
from config import Config
import model as modellib

class SpaceNetConfig(Config):
    NAME = 'spacenet'
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1+1
    USE_MINI_MASK = False

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    MEAN_PIXEL = [0.0, 0.0, 0.0]

    STEPS_PER_EPOCH = 500

class InferenceConfig(SpaceNetConfig):
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1

config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference', config=config, model_dir='./')