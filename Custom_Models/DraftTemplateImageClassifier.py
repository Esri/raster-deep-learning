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

# For Image Classification, for output attribute table
if 'attribute_table' in sys.modules:
    del sys.modules['attribute_table']
from attribute_table import attribute_table

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

class ImageClassifier:
    def __init__(self):
        self.name = 'Image Classifier'
        self.description = 'This python raster function applies deep learning model to do pixel-based classification in imagery'

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

        # self.class_values stores the class and values from its emd
        self.class_values = set([row['Value'] for row in self.json_info.get('Classes', [])])

    def load_model(self):
        '''
        Fill this method to write your own model loading python code.

        Tips: you can access emd information through self.json_info.
        '''

        '''
        # The following is the model loading code for a particular DeepLab model implemented for TensorFlow 1.0. It is just an example.
        if not HAS_TF2:
            raise Exception('Tensorflow(version 2.1.0 or above) libraries are not installed. Install Tensorflow using "conda install tensorflow-gpu=2.1.0".')

        self.classification_graph = tf.Graph()
        with self.classification_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        '''

    def getParameterInfo(self):
        '''
        This method is called after initialize() and provides information on each input parameter expected by the raster function. This method must be defined for the class to be recognized as a valid raster function. The first three parameters are mandatory as they define the input raster, emd, and device.
        After emd file or deep learning model package (dlpk) file is specified in inference GP tool, all the parameters (except raster, model and device) along with their default values, will be populated in the tool. Their default values are defined in this method or read from emd information through self.json_info.

        Tip: you can access emd information through self.json_info.
        '''
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
            }
        ]

        required_parameters.extend(
            [
                {
                    'name': 'padding',
                    'dataType': 'numeric',
                    'required': False,
                    'value': 0 if "Padding" not in self.json_info else int(self.json_info["Padding"]),
                    'displayName': 'Padding',
                    'description': 'Padding'                    
                },
                {
                    'name': 'batch_size',
                    'dataType': 'numeric',
                    'required': False,
                    'value': 1 if "BatchSize" not in self.json_info else int(self.json_info["BatchSize"]),
                    'displayName': 'Batch Size',
                    'description': 'Batch Size'
                    
                },
                {
                    "name": "predict_background",
                    "dataType": "string",
                    "required": False,
                    "value": "True",
                    "displayName": "Predict Background",
                    "description": "If False, will never predict the background/NoData Class."
                }
            ]       
        )

        if 0 not in self.class_values and len(self.class_values)>1:
            for param in required_parameters:
                if param['name'] == 'predict_background':
                    param['value'] = 'False'
                    break

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
        if "padding" in scalars:
            self.padding = int(scalars["padding"])
        elif "Padding" in self.json_info:
            self.padding = int(self.json_info["Padding"])
        else:
            self.padding = 0

        if "batch_size" in scalars:
            self.batch_size = int(scalars["batch_size"])
        elif "BatchSize" in self.json_info:
            self.batch_size = int(self.json_info["BatchSize"])
        else:
            self.batch_size = 1

        self.rectangle_height, self.rectangle_width = prf_utils.calculate_rectangle_size_from_batch_size(self.batch_size)
        
        # ImageHeight and ImageWidth information which is passed through the getConfiguration method, ensures that ArcGIS delivers the pixel blocks of the correct size
        # ty, tx are tile sizes in x and y axis
        ty, tx = prf_utils.get_tile_size(self.json_info['ImageHeight'], self.json_info['ImageWidth'], self.padding, self.rectangle_height, self.rectangle_width)
        self.fixedTileSize = 0 if "fixedTileSize" not in self.json_info else int(self.json_info["fixedTileSize"])

        # all of the supported properties/keys should be added to the following "configuration" dictionary for inference tool to use
        configuration = {
            'padding': self.padding,
            'batch_size': self.batch_size,
            'tx': tx,
            'ty': ty,
            'fixedTileSize': self.fixedTileSize
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

    def updateRasterInfo(self, **kwargs):
        '''
        This method, if defined, updates information that defines the output raster. It's invoked after .getConfiguration() and each time the dataset containing the python raster function is initialized.
        https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#updateRasterInfo
        '''
        kwargs['output_info']['bandCount'] = 1

        # Output pixel type is determined by the value range of classes in the json file
        classes_info = self.json_info['Classes']
        class_values = [class_info['Value'] for class_info in classes_info]
        class_min, class_max = min(class_values), max(class_values)

        if class_min >= 0:
            if class_max > np.iinfo(np.uint16).max:
                kwargs['output_info']['pixelType'] = 'u4'
            elif class_max > np.iinfo(np.uint8).max:
                kwargs['output_info']['pixelType'] = 'u2'
            else:
                kwargs['output_info']['pixelType'] = 'u1'
        else:
            if class_min < np.iinfo(np.int16).min or class_max > np.iinfo(np.int16).max:
                kwargs['output_info']['pixelType'] = 'i4'
            elif class_min < np.iinfo(np.int8).min or class_max > np.iinfo(np.int8).max:
                kwargs['output_info']['pixelType'] = 'i2'
            else:
                kwargs['output_info']['pixelType'] = 'i1'

        attribute_table['features'] = []
        for i, c in enumerate(classes_info):
            attribute_table['features'].append(
                {
                    'attributes':{
                        'OID':i+1,
                        'Value':c['Value'],
                        'Classvalue':c['Value'],
                        'Classname':c['Name'],
                        'Red':c['Color'][0],
                        'Green':c['Color'][1],
                        'Blue':c['Color'][2]
                    }
                }
            )
        kwargs['output_info']['rasterAttributeTable'] = json.dumps(attribute_table)

        return kwargs

    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        '''
        This method, if defined, updates dataset-level or band-level key metadata. When a request for a dataset's key metadata is made, this method allows the python raster function to invalidate or overwrite specific requests.
        https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#updateKeyMetadata
        '''

        '''
        # The following is an example of updateKeyMetadata()
        if bandIndex == -1:
            keyMetadata['datatype'] = 'Scientific'
            keyMetadata['datatype'] = 'Windchill'
        elif bandIndex == 0:
            keyMetadata['wavelengthmin'] = None     # reset inapplicable band-specific key metadata 
            keyMetadata['wavelengthmax'] = None
            keyMetadata['bandname'] = 'Winchill'
        return keyMetadata
        '''

    def selectRasters(self, tlc, shape, props):
        '''
        This method, if defined, enables the Python raster function to define a subset of input rasters from which pixels are read before being fed into the updatePixels method.
        https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#selectRasters
        '''

        '''
        # The following is an example of selectRasters()
        cellSize = props['cellSize']
        v = 0.5 * (cellSize[0] + cellSize[1])
        if v < self.threshold:
            return ('r1',)
        else: return ('r2',)
        '''

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        '''
        This method, if defined, provides output pixels based on pixel blocks associated with all input rasters. A python raster function that doesn't actively modify output pixel values doesn't need to define this method.
        https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#updatePixels
        '''

        '''
        Use this method if you use the Classify Pixels Using Deep Learning tool for semantic segmentation. This method returns the classified raster wrapped in a dictionary. The typical workflow is below:

        Obtain the input image from pixelBlocks and transform to the shape of the model's input.
        Run the deep learning model on the input image tile.
        Post-process the model's output as necessary.
        Generate a classified raster, wrap it in a dictionary and return the dictionary.
        '''
        if not self.model_is_loaded:
            self.load_model(self.model_path)
            self.model_is_loaded = True

        # set pixel values in invalid areas to 0
        raster_mask = pixelBlocks['raster_mask']
        raster_pixels = pixelBlocks['raster_pixels']
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks['raster_pixels'] = raster_pixels

        input_image = pixelBlocks['raster_pixels']
        _, height, width = input_image.shape
        batch, batch_height, batch_width = prf_utils.tile_to_batch(input_image, 
                                                                   self.json_info['ImageHeight'], 
                                                                   self.json_info['ImageWidth'], 
                                                                   self.padding, 
                                                                   fixed_tile_size=True, 
                                                                   batch_height=self.rectangle_height, 
                                                                   batch_width=self.rectangle_width)

        semantic_predictions = self.inference(batch, **self.scalars)

        height, width = semantic_predictions.shape[2], semantic_predictions.shape[3]
        if self.padding != 0:
            semantic_predictions = semantic_predictions[:, :, self.padding:height - self.padding, self.padding:width - self.padding]

        pixelBlocks['output_pixels'] = prf_utils.batch_to_tile(semantic_predictions, batch_height, batch_width)
        return pixelBlocks

    def inference(self, batch, **scalars):
        '''
        Fill this method to write your own inference python code, you can refer to the model instance that is created in the load_model method. Expected results format is described in the returns as below.

        :param batch: numpy array with shape (B, D, H, W), B is batch size, H, W is specified and equal to ImageHeight and ImageWidth in the emd file and D is the number of bands and equal to the length of ExtractBands in the emd. If BatchInference is set to False in emd, B is constant 1.
        :param scalars: inference parameters, accessed by the parameter name, i.e. score_threshold=float(scalars['threshold']). If you want to have more inference parameters, add it to the list of the following getParameterInfo method.
        :return: semantic segmentation, numpy array in the shape [B, 1, H, W] and type np.uint8, B is the batch size, H and W are the tile size, equal to ImageHeight and ImageWidth in the emd file respectively.
        '''

        '''
        # The following is an inference code example for a particular DeepLab model implemented for TensorFlow 1.0.
        # batch is transposed from shape (B, D, H, W) to shape (B, H, W, D)
        batch = np.transpose(batch, (0,2,3,1))

        config = tf.ConfigProto()
        if 'PerProcessGPUMemoryFraction' in self.json_info:
            config.gpu_options.per_process_gpu_memory_fraction = float(self.json_info['PerProcessGPUMemoryFraction'])

        with self.classification_graph.as_default():
            with tf.Session(config=config) as sess:
                feed_dict = {
                    'ImageTensor:0': batch
                }
                fetches = {
                    'SemanticPredictions': 'SemanticPredictions:0'
                }

                output_dict = sess.run(fetches, feed_dict=feed_dict)

        semantic_predictions = np.expand_dims(output_dict['SemanticPredictions'], axis=1)

        return semantic_predictions
        '''