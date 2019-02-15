import importlib
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
from attribute_table import attribute_table
import prf_utils

class ImageClassifier:
    def __init__(self):
        self.name = 'Image Classifier'
        self.description = 'Image classification python raster function to inference a tensorflow ' \
                           'deep learning model'

    def initialize(self, **kwargs):
        if 'model' not in kwargs:
            return

        model = kwargs['model']
        model_as_file = True
        try:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        sys.path.append(os.path.dirname(__file__))
        framework = self.json_info['Framework']
        if 'ModelConfiguration' in self.json_info:
            if isinstance(self.json_info['ModelConfiguration'], str):
                ChildImageClassifier = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration'])), 'ChildImageClassifier')
            else:
                ChildImageClassifier = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration']['Name'])), 'ChildImageClassifier')
        else:
            raise Exception("Invalid model configuration")

        if 'device' in kwargs:
            device = kwargs['device']
            if device < -1:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                device = prf_utils.get_available_device()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        self.child_image_classifier = ChildImageClassifier()
        self.child_image_classifier.initialize(model, model_as_file)

    def getParameterInfo(self):
        required_parameters = [
            {
                'name': 'raster',
                'dataType': 'raster',
                'required': True,
                'displayName': 'Raster',
                'description': 'Input Raster'
            },
            {
                'name': 'model',
                'dataType': 'string',
                'required': True,
                'displayName': 'Input Model Definition (EMD) File',
                'description': 'Input model definition (EMD) JSON file'
            },
            {
                'name': 'device',
                'dataType': 'numeric',
                'required': False,
                'displayName': 'Device ID',
                'description': 'Device ID'
            },
        ]

        if 'ModelPadding' not in self.json_info:
            required_parameters.append(
                {
                    'name': 'padding',
                    'dataType': 'numeric',
                    'value': 0,
                    'required': False,
                    'displayName': 'Padding',
                    'description': 'Padding'
                },
            )

        if 'BatchSize' not in self.json_info:
            required_parameters.append(
                {
                    'name': 'batch_size',
                    'dataType': 'numeric',
                    'required': False,
                    'value': 1,
                    'displayName': 'Batch Size',
                    'description': 'Batch Size'
                },
            )

        return self.child_image_classifier.getParameterInfo(required_parameters)

    def getConfiguration(self, **scalars):
        configuration = self.child_image_classifier.getConfiguration(**scalars)
        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])
        configuration['inheritProperties'] = 2|4|8
        configuration['inputMask'] = True
        return configuration

    def updateRasterInfo(self, **kwargs):
        kwargs['output_info']['bandCount'] = 1
        #todo: type is determined by the value range of classes in the json file
        kwargs['output_info']['pixelType'] = 'i4'
        class_info = self.json_info['Classes']
        attribute_table['features'] = []
        for i, c in enumerate(class_info):
            attribute_table['features'].append(
                {
                    'attributes':{
                        'OID':i+1,
                        'Value':c['Value'],
                        'Class':c['Name'],
                        'Red':c['Color'][0],
                        'Green':c['Color'][1],
                        'Blue':c['Color'][2]
                    }
                }
            )
        kwargs['output_info']['rasterAttributeTable'] = json.dumps(attribute_table)

        return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        # set pixel values in invalid areas to 0
        raster_mask = pixelBlocks['raster_mask']
        raster_pixels = pixelBlocks['raster_pixels']
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks['raster_pixels'] = raster_pixels

        pixelBlocks['output_pixels'] = self.child_image_classifier.updatePixels(tlc, shape, props, **pixelBlocks).astype(props['pixelType'], copy=False)
        return pixelBlocks
