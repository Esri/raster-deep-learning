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

import os
import sys

import numpy as np
import tensorflow as tf

prf_root_dir = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(prf_root_dir)
from Templates.TemplateBaseClassifier import TemplateBaseClassifier

class ChildImageClassifier(TemplateBaseClassifier):
    # template method to fill in
    def load_model(self, model_path):
        '''
        Fill this method to write your own model loading python code
        save it in self object if you would like to reference it later.

        Tips: you can access emd information through self.json_info.
        '''
        # Todo: fill in this method to load your model
        self.classification_graph = tf.Graph()
        with self.classification_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def getParameterInfo(self, required_parameters):
        required_parameters.extend(
            [
                # Todo: add your inference parameters here
                # https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#getparameterinfo
            ]
        )
        required_parameters[:] = [d for d in required_parameters if not d.get('name') == 'batch_size']

        return required_parameters

    # template method to fill in
    def inference(self, batch, **scalars):
        '''
        Fill this method to write your own inference python code, you can refer to the model instance that is created
        in the load_model method. Expected results format is described in the returns as below.

        Tips: you can access emd information through self.json_info.

        :param batch: numpy array with shape (B, H, W, D), B is batch size, H, W is specified and equal to
                      ImageHeight and ImageWidth in the emd file and D is the number of bands and equal to the length
                      of ExtractBands in the emd.
        :param scalars: inference parameters, accessed by the parameter name,
                       i.e. score_threshold=float(kwargs['score_threshold']). If you want to have more inference
                       parameters, add it to the list of the following getParameterInfo method.
        :return: semantic segmentation, numpy array in the shape [B, 1, H, W] and type np.uint8, B is the batch size,
                 H and W are the tile size, equal to ImageHeight and ImageWidth in the emd file respectively.
        '''
        #Todo: fill in this method to inference your model and return bounding boxes, scores and classes
        batch = np.transpose(batch, [0, 2, 3, 1])

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

        semantic_predictions = output_dict['SemanticPredictions']
        return np.expand_dims(semantic_predictions, axis=1)
