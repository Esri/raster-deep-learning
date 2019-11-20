import importlib
from importlib import reload, import_module
import json
import os
import sys
import arcpy
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


class ObjectClassifier:
    def __init__(self):
        self.name = 'Object classifier'
        self.description = 'This python raster function applies deep learning model to ' \
                           'classify objects from overlaid imagery'

    def initialize(self, **kwargs):

        if 'model' not in kwargs:
            return

        # Read esri model definition (emd) file
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
            modelconfig = self.json_info['ModelConfiguration']
            if isinstance(modelconfig, str):
                if modelconfig not in sys.modules:
                    ChildModelDetector = getattr(import_module(
                        '{}.{}'.format(framework, modelconfig)), 'ChildObjectDetector')
                else:
                    ChildModelDetector = getattr(reload(
                        '{}.{}'.format(framework, modelconfig)), 'ChildObjectDetector')
            else:
                modelconfig = self.json_info['ModelConfiguration']['Name']
                ChildModelDetector = getattr(importlib.import_module(
                    '{}.{}'.format(framework, modelconfig)), 'ChildObjectDetector')
        else:
            raise Exception("Invalid model configuration")

        # if 'device' in kwargs:
        #     device = kwargs['device']
        #     if device < -1:
        #         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #         device = prf_utils.get_available_device()
        #     os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
        # else:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        device = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = prf_utils.get_available_device()

        if device is not None:
            if device >= 0:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                arcpy.env.processorType = "CPU"
        else:
            arcpy.env.processorType = "CPU"

        self.child_object_detector = ChildModelDetector()
        self.child_object_detector.initialize(model, model_as_file)


    def getParameterInfo(self):

        # PRF needs values of these parameters from gp tool user,
        # either from gp tool UI or emd (a json) file.
        required_parameters = [
            {
                # To support mini batch, it is required that Classify Objects Using Deep Learning geoprocessing Tool
                # passes down a stack of raster tiles to PRF for model inference, the keyword required here is 'rasters'.
                'name': 'rasters',
                'dataType': 'rasters',
                'value': None,
                'required': True,
                'displayName': "Rasters",
                'description': 'The collection of overlapping rasters to objects to be classified'
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
            }
        ]

        if 'BatchSize' not in self.json_info:
             required_parameters.append(
                 {
                     'name': 'batch_size',
                     'dataType': 'numeric',
                     'required': False,
                     'value': 4,
                     'displayName': 'Batch Size',
                     'description': 'Batch Size'
                 }
             )

        return self.child_object_detector.getParameterInfo(required_parameters)


    def getConfiguration(self, **scalars):

        # The information PRF returns to the GP tool,
        # the information is either from emd or defined in getConfiguration method.

        configuration = self.child_object_detector.getConfiguration(**scalars)

        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])

        configuration['inheritProperties'] = 2|4|8
        configuration['inputMask'] = True

        return configuration

    def getFields(self):

        if 'Label' not in fields['fields']:
            fields['fields'].append(
                {
                    'name': 'Label',
                    'type': 'esriFieldTypeString',
                    'alias': 'Label'
                }
            )
        return json.dumps(fields)

    def getGeometryType(self):
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):

        # set pixel values in invalid areas to 0
        rasters_mask = pixelBlocks['rasters_mask']
        rasters_pixels = pixelBlocks['rasters_pixels']

        for i in range(0, len(rasters_pixels)):
            rasters_pixels[i][np.where(rasters_mask[i] == 0)] = 0

        pixelBlocks['rasters_pixels'] = rasters_pixels

        polygon_list, scores, labels = self.child_object_detector.vectorize(**pixelBlocks)

        features['features'] = []

        features['fieldAliases'].update({
            'Label': 'Label'
        })
        features['fields'].append(
            {
                'name': 'Label',
                'type': 'esriFieldTypeString',
                'alias': 'Label'
            }
        )
        for i in range(len(polygon_list)):

            rings = [[]]
            for j in range(len(polygon_list[i])):
                rings[0].append(
                    [
                        polygon_list[i][j][1],
                        polygon_list[i][j][0]
                    ]
                )

            features['features'].append({
                'attributes': {
                    'OID': i + 1,
                    'Confidence': str(scores[i]),
                    'Label': labels[i]
                },
                'geometry': {
                    'rings': rings
                }
            })

        return {'output_vectors': json.dumps(features)}
