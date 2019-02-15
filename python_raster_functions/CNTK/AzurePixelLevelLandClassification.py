import os
import sys

import cntk as C
import numpy as np

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
        self.model = C.load_model(model_path)

    def getParameterInfo(self, required_parameters):
        required_parameters.extend(
            [
                # Todo: add your inference parameters here
                # https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#getparameterinfo
            ]
        )
        return required_parameters

    # template method to fill in
    def inference(self, batch, **kwargs):
        '''
        Fill this method to write your own inference python code, you can refer to the model instance that is created
        in the load_model method. Expected results format is described in the returns as below.

        Tips: you can access emd information through self.json_info.

        :param batch: numpy array with shape (B, H, W, D), B is batch size, H, W is specified and equal to
                      ImageHeight and ImageWidth in the emd file and D is the number of bands and equal to the length
                      of ExtractBands in the emd.
        :param kwargs: inference parameters, accessed by the parameter name,
                       i.e. score_threshold=float(kwargs['score_threshold']). If you want to have more inference
                       parameters, add it to the list of the following getParameterInfo method.
        :return: semantic segmentation, numpy array in the shape [B, 1, H, W] and type np.uint8, B is the batch size,
                 H and W are the tile size, equal to ImageHeight and ImageWidth in the emd file respectively if Padding
                 is not set
        '''
        # Todo: fill in this method to inference your model and return bounding boxes, scores and classes
        batch=batch.astype(np.float32)

        output = self.model.eval(
            {
                self.model.arguments[0]: batch
            }
        )
        semantic_predictions = np.argmax(output, axis=1)
        semantic_predictions = np.expand_dims(semantic_predictions, axis=1)

        return semantic_predictions
