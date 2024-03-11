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

limitations under the License.
'''

import sys, os, pathlib
import json

# Add current folder to the system path
sys.path.append(os.path.dirname(__file__))

# Helper functions for writing python raster functions
import prf_utils

import numpy as np

# Only for Tensorflow 1 or 2, check if Tensorflow 1 or 2 is installed or not
try:
    import tensorflow as tf
    if tf.__version__[0] == '2':
        # Pro's Deep Learning Installer contains only Tensorflow 2, this line is needed to run TF1 models, diabling certain incompatible TF2 functions
        tf.compat.v1.disable_v2_behavior()

        HAS_TF2 = True
        HAS_TF1 = False
    else:
        HAS_TF2 = False
        HAS_TF1 = True

except Exception as e:
    HAS_TF2 = False
    HAS_TF1 = False

# Only for Keras with Tensorflow as backend, check if Keras is installed or not
try:
    import keras
    import keras.backend as K
    import keras.backend.tensorflow_backend as tb 
    if HAS_TF2 or HAS_TF1:
        HAS_KERAS = True
    else:
        HAS_KERAS = False

except Exception as e:
    HAS_KERAS = False

# Only for PyTorch,  check if PyTorch is installed or not
try:
    import torch
    HAS_PYTORCH = True

except Exception as e:
    HAS_PYTORCH = False

# For Object Classification, for output features and their fields
features = {
    'displayFieldName': '',
    'fieldAliases': {
        'FID': 'FID',
        'Class': 'Class',
        'Confidence': 'Confidence'
    },
    'geometryType': 'esriGeometryPolygon',
    'fields': [
        {
            'name': 'FID',
            'type': 'esriFieldTypeOID',
            'alias': 'FID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        }
    ],
    'features': []
}

fields = {
    'fields': [
        {
            'name': 'OID',
            'type': 'esriFieldTypeOID',
            'alias': 'OID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        },
        {
            'name': 'Shape',
            'type': 'esriFieldTypeGeometry',
            'alias': 'Shape'
        }
    ]
}

# For Object Detection and Object Classification. Geometry Type accepted in ArcGIS Pro
class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4


class ObjectClassifier:
    def __init__(self):
        self.name = 'Object Classifier'
        self.description = 'This python raster function applies deep learning model to classify objects from overlaid imagery'

    def initialize(self, **kwargs):
        '''
        This method is called first when inference starts. What it does is: loading the model emd file into inference GP tool, setting correct processor type(GPU or CPU), etc.
        '''

        # Read esri model definition (emd) file, and its content is stored in self.json_info for later access in this python raster function
        if 'model' not in kwargs:
            return

        model = kwargs['model'] # kwargs['model'] is the emd file
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

        # Add current folder to the system path
        sys.path.append(os.path.dirname(__file__))

        # Set processor type to be GPU or CPU
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = prf_utils.get_available_device()

        if device is not None:
            if device >= 0:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

                # Only for PyTorch
                # Set processor type to be GPU in PyTorch
                if not HAS_PYTORCH:
                    raise Exception("PyTorch is not installed. Install it using 'conda install deep-learning-essentials -y'")
                torch.cuda.set_device(device)

                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                arcpy.env.processorType = "CPU"

        # self.model_path stores the actual DL model file path from its emd file's 'ModelFile' attribute value and self.model_is_loaded flags for loading status for current DL model file
        model_path = self.json_info['ModelFile']
        if model_as_file and not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(os.path.dirname(model), model_path))

        self.model_path = model_path
        self.model_is_loaded = False

    def load_model(self):
        '''
        Fill this method to write your own model loading python code.

        Tip: you can access emd information through self.json_info.
        '''

        '''
        # The following is the model loading code for a particular Object Classifier model implemented in Keras using Tensorflow as backend. It is just an example.
        # load the trained model
        if not HAS_KERAS:
            raise Exception('Keras(2.3.1 or above), Tensorflow(version 2.1.0 or above) and scikit-image(version 0.15.0 or above) '
                            'libraries are not installed. Install them using "conda install tensorflow-gpu=2.1.0 scikit-image=0.15.0 keras-gpu=2.2.4 ".')
        K.clear_session()
        tb._SYMBOLIC_SCOPE.value = True # keras compatible with tensorflow 2.1.0
        self.model = keras.models.load_model(model_path)
        self.graph = tf.compat.v1.get_default_graph()
        '''

        '''
        The following is the model loading code for using PyTorch to train a feature classifier, with arcgis.learn
        if arcpy.env.processorType == "GPU" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            arcgis.env._processorType = "GPU"
        else:
            self.device = torch.device("cpu")
            arcgis.env._processorType = "CPU"

        # Using arcgis.learn FeatureClassifer from_model function.
        self.cf = FeatureClassifier.from_model(emd_path=model)
        self._learnmodel = self.cf
        self.model = self.cf.learn.model
        self.model = self.cf.learn.model.to(self.device)
        self.model.eval()
        '''

    def getParameterInfo(self):
        '''
        This method is called after initialize() and provides information on each input parameter expected by the raster function. This method must be defined for the class to be recognized as a valid raster function. The first three parameters are mandatory as they define the input raster, emd, and device.
        After emd file or deep learning model package (dlpk) file is specified in inference GP tool, all the parameters (except raster, model and device) along with their default values, will be populated in the tool. Their default values are defined in this method or read from emd information through self.json_info.

        Tip: you can access emd information through self.json_info.
        '''
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
                     'value': 64,
                     'displayName': 'Batch Size',
                     'description': 'Batch Size'
                 }
             )

        # Todo: add your inference parameters here. Reference: https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#getparameterinfo
        # Below is an example adding new parameters
        '''
        required_parameters.extend(
            [ 
                {
                    'name': 'new_parameter1',
                    'dataType': 'numeric',
                    'value': 0.2,
                    'required': True,
                    'displayName': 'New Parameter 1',
                    'description': 'new parameter 1'
                },
                {
                    'name': 'new_parameter2',
                    'dataType': 'numeric',
                    'value': 0.2,
                    'required': True,
                    'displayName': 'New Parameter 2',
                    'description': 'new parameter 2'
                }
            ]
        )
        '''

        return required_parameters

    def getConfiguration(self, **scalars):
        '''
        This method is used to set the properties/keys like padding, batch size, tile size, input bands and many more. This method, if defined, manages how the output raster is constructed. Decisions in this method may be based on the user-updated values of one or more scalar (non-raster) parameters. This method is invoked after .getParameterInfo() but before .updateRasterInfo() by which time all rasters will have been loaded.

        The scalars value contains all the parameter values that can be accessed by the parameter name. Some of the recognized properties/keys and their descriptions are listed here
        https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#getConfiguration
        The full list of supported properties/keys is: padding,fixedTileSize, tx,ty,CropSizeFixed,batch_size,extractBands,dataRange,inheritProperties,invalidateProperties,resampling,BlackenAroundFeature,keyMetadata,compositeRasters,samplingFactor,_noUpdatePixels,resamplingType,supportsBandSelection, maskUpdated, byRef

        Remember to save the parameter values here as self.PARAMETER_NAME if you want to use the parameter values in other methods.
        '''
        if "batch_size" in scalars:
            self.batch_size = int(scalars["batch_size"])
        elif "BatchSize" in self.json_info:
            self.batch_size = int(self.json_info["BatchSize"])
        else:
            self.batch_size = 64

        if "padding" in scalars:
            self.padding = int(scalars["padding"])
        elif "Padding" in self.json_info:
            self.padding = int(self.json_info["Padding"])
        else:
            self.padding = 0

        if "CropSizeFixed" in scalars:
            self.CropSizeFixed = int(scalars["CropSizeFixed"])
        elif "CropSizeFixed" in self.json_info:
            self.CropSizeFixed = int(self.json_info["CropSizeFixed"])
        else:
            self.CropSizeFixed = 1
        
        if "BlackenAroundFeature" in scalars:
            self.BlackenAroundFeature = int(scalars["BlackenAroundFeature"])
        elif "BlackenAroundFeature" in self.json_info:
            self.BlackenAroundFeature = int(self.json_info["BlackenAroundFeature"])
        else:
            self.batch_size = 1

        self.rectangle_height, self.rectangle_width = prf_utils.calculate_rectangle_size_from_batch_size(self.batch_size)
        ty, tx = prf_utils.get_tile_size(self.json_info['ImageHeight'], self.json_info['ImageWidth'], self.padding, self.rectangle_height, self.rectangle_width)

        configuration = {
            'CropSizeFixed': self.CropSizeFixed,
            'BlackenAroundFeature': self.BlackenAroundFeature,
            'padding': self.padding,
            'batch_size': self.batch_size,
            'tx': tx,
            'ty': ty,
        }
        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])
        if "ExtractBands" in self.json_info:
            configuration["extractBands"] = tuple(self.json_info["ExtractBands"])
        configuration['inheritProperties'] = 2|4|8
        configuration['inputMask'] = True

        '''
        # The following is an example of how to save the parameter values here as self.PARAMETER_NAME if you want to use the parameter values in other methods.
        if "pvalue" in scalars:
            self.pvalue = float(scalars["pvalue"])
        elif "pvalue" in self.json_info:
            self.pvalue = float(self.json_info["pvalue"])
        else:
            self.pvalue = 1
        '''

        self.scalars = scalars

        return configuration

    def getFields(self):
        '''
        Use this method to return the JSON string fields of the output feature class. Fields are defined in fields.py or you can customize it to suit your own needs.
        '''
        fields['fields'].append(
            {
                'name': 'Label',
                'type': 'esriFieldTypeString',
                'alias': 'Label'
            }
        )
        return json.dumps(fields)

    def getGeometryType(self):
        '''
        Use this method if you use the Detect Objects Using Deep Learning tool and you want to declare the feature geometry type of the output detected objects. Typically, the output is a polygon feature class if the model is to draw bounding boxes around objects.
        '''
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):
        '''
        Use this method if you use the Classify Objects Using Deep Learning tool. This method returns a dictionary in which the "output_vectors" property is a string of features in image space in JSON format. A typical workflow is below:

        1.Obtain the input image from pixelBlocks and transform to the shape of the model's input.
        2.Run the deep learning model on the input image tile. 
        3.Post-process the model's output as necessary. 
        4.Generate a feature JSON object, wrap it as a string in a dictionary and return the dictionary.
        '''

        # Each pixelBlocks is cropped from the input raster with a shape of (tx+2*padding) by (ty+2*padding)
        raster_mask = pixelBlocks['raster_mask']
        raster_pixels = pixelBlocks['raster_pixels']
        
        # Set pixel values in invalid areas to 0
        raster_pixels[np.where(raster_mask == 0)] = 0

        # Get pixel blocks - tuple of 3-d rasters: ([bands,height,width],[bands,height.width],...)
        pixelBlocks['raster_pixels'] = raster_pixels
        batch_images = np.asarray(pixelBlocks["rasters_pixels"])

        # Call the deep learning framework specific load_model() defined previously to load the model file
        if not self.model_is_loaded:
            self.load_model()
            self.model_is_loaded = True
#change
        try:
            polygon_list, scores, labels = self.inference(batch_images, **self.scalars)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # arcpy.AddError('Runtime Error: ran out of GPU memory, please try a smaller batch size')
                raise RuntimeError("Ran out of GPU memory, please try a smaller batch size")
                return None
            else:
                # arcpy.AddError('Runtime Error:" + str(e) + "Inferencing was not successful.')
                raise RuntimeError("Inferencing was not successful.")
                return None

        features['features'] = []

        features['fieldAliases'].update({
            'Label': 'Label'
        })

        Labelfield = {
                'name': 'Label',
                'type': 'esriFieldTypeString',
                'alias': 'Label'
        }

        if not Labelfield in features['fields']:
            features['fields'].append(Labelfield)

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
                    'Label': labels[i],
                    'Classname': labels[i]
                },
                'geometry': {
                    'rings': rings
                }
            })

        return {'output_vectors': json.dumps(features)}

    def inference(self, batch_images, **scalars):
        '''
        Fill this method to write your own inference python code, you can refer to the model instance that is created in the load_model method. Expected results format is described in the returns as below.

        :param batch: numpy array with shape (B, D, H, W), B is batch size, H, W is specified and equal to ImageHeight and ImageWidth in the emd file and D is the number of bands and equal to the length of ExtractBands in the emd. If BatchInference is set to False in emd, B is constant 1.
        :param scalars: inference parameters, accessed by the parameter name, i.e. score_threshold=float(scalars['threshold']). If you want to have more inference parameters, add it to the list of the following getParameterInfo method.
        :return: 
          1. rings: python list representing bounding boxes whose length is equal to B, each element is [N,4] numpy array representing [ymin, xmin, ymax, xmax] with respect to the upper left corner of the image tile.
          2. confidences: python list representing the score of each bounding box whose length is equal to B, each element is a [N,] numpy array
          3. labels: python list representing the class of each bounding box whose length is equal to B, each element is [N,] numpy array and its dype is np.uint8
        '''
        #Todo: fill in this method to inference your model and return bounding boxes, scores and classes

        '''
        # The following is an inference code example for using PyTorch to train a feature classifier, with arcgis.learn

        # Get the shape of the 4-d numpy array
        batch_size, bands, height, width = batch_images.shape

        rings = []
        labels, confidences = [], []

        # Transpose the image dimensions to [batch, height, width, bands],
        # normalize and transpose back to [batch, bands, height, width]
        batch_images = norm(batch.transpose(0, 2, 3, 1)).transpose(
            0, 3, 1, 2
        )

        # Convert to torch tensor, set device and convert to float
        batch_images = torch.tensor(batch_images).to(self.device).float()

        # the second element in the passed tuple is hardcoded to make fastai's pred_batch work
        predictions = self.cf.learn.pred_batch(
            batch=(batch_images, torch.tensor([40]).to(self.device))
        )
        # predictions: torch.tensor(B,C), where B is the batch size and C is the number of classes

        # Using emd to map the class
        class_map = [c["Name"] for c in self.json_info["Classes"]]

        # torch.max returns the max value and the index of the max as a tuple
        confidences, class_idxs = torch.max(predictions, dim=1)
        confidences = confidences.tolist()
        labels = [class_map[c] for c in class_idxs]

        # Appending this ring for all the features in the batch
        rings = [
            [[0, 0], [0, width - 1], [height - 1, width - 1], [height - 1, 0]]
            for i in range(batch)
        ]
        '''
        return rings, confidences, labels
