import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
import importlib
from skimage.measure import find_contours
import keras.backend as K
import tensorflow as tf

class MatterMaskRCNN:
    def initialize(self, model, model_as_file):
        K.clear_session()

        if model_as_file:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        else:
            self.json_info = json.loads(model)

        model_path = self.json_info['ModelFile']
        if model_as_file and not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(os.path.dirname(model), model_path))

        config_module = self.json_info['ModelConfiguration']['Config']
        if not os.path.isabs(config_module):
            config_module = os.path.abspath(os.path.join(os.path.dirname(model), config_module))

        sys.path.append(os.path.dirname(config_module))
        config_module_name = os.path.basename(config_module)

        if config_module_name in sys.modules:
            del sys.modules[config_module_name]

        self.config = getattr(importlib.import_module(config_module_name), 'config')

        architecture_module = self.json_info['ModelConfiguration']['Architecture']
        if not os.path.isabs(architecture_module):
            architecture_module = os.path.abspath(os.path.join(os.path.dirname(model), architecture_module))

        sys.path.append(os.path.dirname(architecture_module))
        architecture_module_name = os.path.basename(architecture_module)

        if (architecture_module_name != config_module_name) and (architecture_module_name in sys.modules):
            del sys.modules[architecture_module_name]

        self.model = getattr(importlib.import_module(architecture_module_name), 'model')

        self.model.load_weights(model_path, by_name=True)

        self.graph = tf.get_default_graph()

    def getParameterInfo(self, required_parameters):
        return required_parameters

    def getConfiguration(self, **scalars):
        self.padding = int(scalars['padding'])

        return {
            'extractBands': tuple(self.json_info['ExtractBands']),
            'padding': int(scalars['padding']),
            'tx': self.json_info['ImageWidth'] - 2 * self.padding,
            'ty': self.json_info['ImageHeight'] - 2 * self.padding
        }

class ChildImageClassifier(MatterMaskRCNN):
    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        image = pixelBlocks['raster_pixels']
        _, height, width = image.shape
        image = np.transpose(image, [1,2,0])

        with self.graph.as_default():
            results = self.model.detect([image], verbose=1)

        masks = results[0]['masks']
        class_ids = results[0]['class_ids']
        output_raster = np.zeros((masks.shape[0], masks.shape[1], 1), dtype=props['pixelType'])
        mask_count = masks.shape[-1]
        for i in range(mask_count):
            mask = masks[:, :, i]
            output_raster[np.where(mask==True)] = class_ids[i]

        return np.transpose(output_raster, [2,0,1])


class ChildObjectDetector(MatterMaskRCNN):
    def vectorize(self,  **pixelBlocks):
        image = pixelBlocks['raster_pixels']
        _, height, width = image.shape
        image = np.transpose(image, [1,2,0])

        with self.graph.as_default():
            results = self.model.detect([image], verbose=1)

        masks = results[0]['masks']
        mask_count = masks.shape[-1]
        coord_list = []
        for m in range(mask_count):
            mask = masks[:, :, m]
            padded_mask = np.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5, fully_connected='high')

            if len(contours) != 0:
                verts = contours[0] - 1
                coord_list.append(verts)

        if self.padding != 0:
            coord_list[:] = [item - self.padding for item in coord_list]

        return coord_list, results[0]['scores'], results[0]['class_ids']