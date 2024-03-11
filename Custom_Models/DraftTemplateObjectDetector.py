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

# For Object Detection, for output features and their fields
from fields import fields
from features import features

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

# Only for Object Detection API with TensorFlow 2
from object_detection.utils import config_util
from object_detection.builders import model_builder

# Only for PyTorch,  check if PyTorch is installed or not
try:
    import torch
    HAS_PYTORCH = True

except Exception as e:
    HAS_PYTORCH = False

# For Object Detection and Object Classification. Geometry Type accepted in ArcGIS Pro
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
        # The following is the model loading code for a particular SSD model implemented in PyTorch. It is just an example.
        if not HAS_PYTORCH:
            raise Exception('PyTorch(version 1.1.0 or above) libraries are not installed. Install PyTorch using "conda install pytorch=1.1.0 ".')

        f_model = resnet34
        head_reg = SSD_MultiHead(k, -4.)
        models = ConvnetBuilder(f_model, 0, 0, 0, custom_head=head_reg)
        self.model = models.model
        self.model = util.load_model(self.model, model_path)
        self.model.eval()
        '''

        '''
        # The following is the model loading code for TensorFlow 1.0, to import graph def from frozen pb file:
        if not HAS_TF1:
            raise Exception('Tensorflow 1 libraries are not installed. Install Tensorflow using "conda install tensorflow-gpu".')
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        '''

        '''
        # The following is the model loading code for Tensorflow 2 Object Detection API 2.0 models
        if not HAS_TF2:
            raise Exception("Tensorflow 2(version 2.1.0 or above) is not installed. Install Tensorflow using 'conda install deep-learning-essentials -y'")

        # load model code for Tensorflow 2 Object Detection API
        pipeline_config = os.path.join(self.model_path, 'pipeline.config')
        checkpoint_dir = os.path.join(self.model_path, 'checkpoint')

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs['model']
        self.detection_model = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)

        # Generally you want to put the last ckpt from training in here
        filenames = list(pathlib.Path(checkpoint_dir).glob('*.index'))
        filenames.sort()  
        ckpt.restore(str(filenames[-1]).replace('.index','')).expect_partial()
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
                    'name': 'threshold',
                    'dataType': 'numeric',
                    'required': False,
                    'value': 0.5 if "Threshold" not in self.json_info else float(self.json_info["Threshold"]),
                    'displayName': 'Confidence Score Threshold [0.0, 1.0]',
                    'description': 'Confidence score threshold value [0.0, 1.0]'                    
                },
                {
                    'name': 'batch_size',
                    'dataType': 'numeric',
                    'required': False,
                    'value': 64 if "BatchSize" not in self.json_info else int(self.json_info["BatchSize"]),
                    'displayName': 'Batch Size',
                    'description': 'Batch Size'
                    
                }
            ]       
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
        This method is used to set the properties/keys like padding, batch size, tile size, input bands, (detection) threshold and many more. This method, if defined, manages how the output raster is constructed. Decisions in this method may be based on the user-updated values of one or more scalar (non-raster) parameters. This method is invoked after .getParameterInfo() but before .updateRasterInfo() by which time all rasters will have been loaded.

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
            self.batch_size = 64

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

        # The following is an example of how to save the parameter values here as self.PARAMETER_NAME if you want to use the parameter values in other methods.
        if "threshold" in scalars:
            self.thres = float(scalars["threshold"])
        elif "Threshold" in self.json_info:
            self.thres = float(self.json_info["Threshold"])
        else:
            self.thres = 0.5

        self.scalars = scalars

        return configuration

    def getFields(self):
        '''
        Use this method to return the JSON string fields of the output feature class. Fields are defined in fields.py or you can customize it to suit your own needs.
        '''
        return json.dumps(fields)

    def getGeometryType(self):
        '''
        Use this method if you use the Detect Objects Using Deep Learning tool and you want to declare the feature geometry type of the output detected objects. Typically, the output is a polygon feature class if the model is to draw bounding boxes around objects.
        '''
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):
        '''
        Use this method if you use the Detect Objects Using Deep Learning tool. This method returns a dictionary in which the "output_vectors" property is a string of features in image space in JSON format. A typical workflow is below:

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
        pixelBlocks['raster_pixels'] = raster_pixels

        # Call the deep learning framework specific load_model() defined previously to load the model file
        if not self.model_is_loaded:
            self.load_model()
            self.model_is_loaded = True

        input_images = pixelBlocks['raster_pixels']

        if self.fixedTileSize == 1:
            fixed_tile_size = True
        else:
            fixed_tile_size = False

        batch, batch_height, batch_width = \
            prf_utils.tile_to_batch(input_images,
                                    self.json_info['ImageHeight'],
                                    self.json_info['ImageWidth'],
                                    self.padding,
                                    fixed_tile_size=fixed_tile_size)

        # Call the deep learning framework specific inference code to run inference on your own model
        batch_bounding_boxes, batch_scores, batch_classes = self.inference(batch, **self.scalars)

        # Convert a batch of object detection results to the format of detections in a flat tile
        polygon_list, scores, classes = prf_utils.batch_detection_results_to_tile_results(
            batch_bounding_boxes,
            batch_scores,
            batch_classes,
            self.json_info['ImageHeight'],
            self.json_info['ImageWidth'],
            self.padding,
            batch_width
        )

        scores = scores.tolist()
        classes = classes.tolist()
        features['features'] = []

        # Obtain label class name and value dictionary from emd file
        clsvalue_index_dict = {int(class_info['Value']): index for index, class_info in enumerate(self.json_info['Classes'])}

        for i in range(len(polygon_list)):
            rings = [[]]
            for j in range(polygon_list[i].shape[0]):
                rings[0].append(
                    [
                        polygon_list[i][j][1],
                        polygon_list[i][j][0]
                    ]
                )
            cls_idx = clsvalue_index_dict.get(int(classes[i]), -1)
            if cls_idx == -1:
                raise Exception("Detected object's class not found in emd file. Please check again.")


            features['features'].append({
                'attributes': {
                    'OID': i + 1,
                    'Classname': self.json_info['Classes'][cls_idx]['Name'],
                    'Classvalue': self.json_info['Classes'][cls_idx]['Value'],
                    'Value': self.json_info['Classes'][cls_idx]['Value'],
                    'Confidence': scores[i]
                },
                # Create feature json object and fill out the geometry
                'geometry': {
                    'rings': rings
                }
            })
        # Wrap the json object as a string in dictionary, this is the final output of the entire python raster function
        return {'output_vectors': json.dumps(features)}

    def inference(self, batch, **scalars):
        '''
        Fill this method to write your own inference python code and to inference your own model, and return bounding boxes, scores and classes. 
        Expected results format is described in the returns as below.

        :param batch: numpy array with shape (B, D, H, W), B is batch size, H, W is specified and equal to ImageHeight and ImageWidth in the emd file and D is the number of bands and equal to the length of ExtractBands in the emd.
        :param scalars: inference parameters, accessed by the parameter name, i.e. self.thres=float(scalars['threshold']). If you want to have more inference parameters, add it to the list of the previous getParameterInfo method.
        :return: 
          1. bounding boxes: python list representing bounding boxes whose length is equal to B, each element is [N,4] numpy array representing [ymin, xmin, ymax, xmax] with respect to the upper left corner of the image tile.
          2. scores: python list representing the score of each bounding box whose length is equal to B, each element is a [N,] numpy array
          3. classes: python list representing the class of each bounding box whose length is equal to B, each element is [N,] numpy array and its dype is np.uint8
        '''

        '''
        # The following is an inference code example for Tensorflow 2 Object Detection API
        # Obtain the input image\images from batch, then batch is transposed from shape (B, D, H, W) to shape (B, H, W, D) so that Tensorflow can recognize its shape
        batch = np.transpose(batch, (0,2,3,1)) 

        label_id_offset = 1
        def get_model_detection_function(model):
            """Get a tf.function for detection."""

            @tf.function
            def detect_fn(image):
                """Detect objects in image."""

                image, shapes = model.preprocess(image)
                # run the model on the input image
                prediction_dict = model.predict(image, shapes)
                detections = model.postprocess(prediction_dict, shapes)

                return detections, prediction_dict, tf.reshape(shapes, [-1])

            return detect_fn

        detect_fn = get_model_detection_function(self.detection_model)

        batch_bounding_boxes, batch_scores, batch_classes = [], [], []

        batch_size = batch.shape[0]
        for index in range(batch_size):
            input_tensor = tf.convert_to_tensor(np.expand_dims(batch[index], 0), dtype=tf.float32)
            detections, predictions_dict, shapes = detect_fn(input_tensor)

            bounding_boxes = detections['detection_boxes'][0].numpy()
            # detection_boxes: [ , ymin, xmin, ymax, xmax]
            bounding_boxes[:, [0, 2]] = bounding_boxes[:, [0, 2]] * self.json_info['ImageHeight']
            bounding_boxes[:, [1, 3]] = bounding_boxes[:, [1, 3]] * self.json_info['ImageWidth']
            scores = detections['detection_scores'][0].numpy()
            classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)

            keep_indices = np.where(scores > self.thres)
            batch_bounding_boxes.append(bounding_boxes[keep_indices])
            batch_scores.append(scores[keep_indices])
            batch_classes.append(classes[keep_indices])
        '''

        '''
        # The following is an inference code example for Tensorflow 1 Object Detection API     


        # Obtain the input image\images from batch, then batch is transposed from shape (B, D, H, W) to shape (B, H, W, D) so that Tensorflow can recognize its shape
        batch = np.transpose(batch, (0, 2, 3, 1))
        config = tf.compat.v1.ConfigProto()
        with self.detection_graph.as_default():
            with tf.compat.v1.Session(config=config) as sess:
                feed_dict = {
                    'image_tensor:0': batch
                }
                fetches = {
                    'boundingboxes': 'detection_boxes:0',
                    'scores': 'detection_scores:0',
                    'classes': 'detection_classes:0'
                }
                # run the model on the input image/images
                output_dict = sess.run(fetches, feed_dict=feed_dict)

        bounding_boxes = output_dict['boundingboxes']
        scores = output_dict['scores']
        classes = output_dict['classes']

        # The scaled coordinates(from 0 to 1.0) of bounding_boxes: [ymin, xmin, ymax, xmax] is scaled back to the image's actual size
        bounding_boxes[:, :, [0, 2]] = bounding_boxes[:, :, [0, 2]] * self.json_info['ImageHeight']
        bounding_boxes[:, :, [1, 3]] = bounding_boxes[:, :, [1, 3]] * self.json_info['ImageWidth']

        batch_bounding_boxes, batch_scores, batch_classes = [], [], []

        batch_size = bounding_boxes.shape[0]
        for batch_idx in range(batch_size):
            # Detection threshold value can be either already defined in getConfiguration() as self.thres or can be obtained from scalars via self.thres = float(scalars['threshold'])
            keep_indices = np.where(scores[batch_idx] > self.thres)
            batch_bounding_boxes.append(bounding_boxes[batch_idx][keep_indices])
            batch_scores.append(scores[batch_idx][keep_indices])
            batch_classes.append(classes[batch_idx][keep_indices])
        '''
        return batch_bounding_boxes, batch_scores, batch_classes