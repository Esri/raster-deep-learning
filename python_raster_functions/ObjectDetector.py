'''
Copyright 2018 Esri

Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.

You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and

limitations under the License.​
'''

import importlib
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from fields import fields
from features import features
import numpy as np
import prf_utils

class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4


class ObjectDetector:
    def __init__(self):
        self.name = 'Object Detector'
        self.description = 'This python raster function applies deep learning model to detect objects in imagery'

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

        if 'device' in kwargs:
            device = kwargs['device']
            if device < -1:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                device = prf_utils.get_available_device()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        sys.path.append(os.path.dirname(__file__))
        framework = self.json_info['Framework']
        if 'ModelConfiguration' in self.json_info:
            if isinstance(self.json_info['ModelConfiguration'], str):
                ChildModelDetector = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration'])), 'ChildObjectDetector')
            else:
                ChildModelDetector = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration']['Name'])), 'ChildObjectDetector')
        else:
            raise Exception("Invalid model configuration")

        self.child_object_detector = ChildModelDetector()
        self.child_object_detector.initialize(model, model_as_file)

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
            {
                'name': 'padding',
                'dataType': 'numeric',
                'value': 0,
                'required': False,
                'displayName': 'Padding',
                'description': 'Padding'
            },
            {
                'name': 'score_threshold',
                'dataType': 'numeric',
                'value': 0.6,
                'required': False,
                'displayName': 'Confidence Score Threshold [0.0, 1.0]',
                'description': 'Confidence score threshold value [0.0, 1.0]'
            },
        ]

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

        return self.child_object_detector.getParameterInfo(required_parameters)

    def getConfiguration(self, **scalars):
        configuration = self.child_object_detector.getConfiguration(**scalars)
        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])
        configuration['inheritProperties'] = 2|4|8
        configuration['inputMask'] = True
        return configuration

    def getFields(self):
        return json.dumps(fields)

    def getGeometryType(self):
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):
        # set pixel values in invalid areas to 0
        raster_mask = pixelBlocks['raster_mask']
        raster_pixels = pixelBlocks['raster_pixels']
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks['raster_pixels'] = raster_pixels

        polygon_list, scores, classes = self.child_object_detector.vectorize(**pixelBlocks)

        # bounding_boxes = bounding_boxes.tolist()
        scores = scores.tolist()
        classes = classes.tolist()
        features['features'] = []

        for i in range(len(polygon_list)):
            rings = [[]]
            for j in range(polygon_list[i].shape[0]):
                rings[0].append(
                    [
                        polygon_list[i][j][1],
                        polygon_list[i][j][0]
                    ]
                )

            features['features'].append({
                'attributes': {
                    'OID': i + 1,
                    'Class': self.json_info['Classes'][classes[i] - 1]['Name'],
                    'Confidence': scores[i]
                },
                'geometry': {
                    'rings': rings
                }
            })
        return {'output_vectors': json.dumps(features)}
