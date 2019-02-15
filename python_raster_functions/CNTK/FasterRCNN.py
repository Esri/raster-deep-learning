import os
import sys

import cntk as C
import numpy as np

sys.path.append(os.path.dirname(__file__))
from utils.rpn.bbox_transform import regress_rois
from utils.nms_wrapper import apply_nms_to_single_image_results

prf_root_dir = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(prf_root_dir)
from Templates.TemplateBaseDetector import TemplateBaseDetector

class ChildObjectDetector(TemplateBaseDetector):
    def load_model(self, model_path):
        '''
        Fill this method to write your own model loading python code
        save it self object if you would like to reference it later.

        Tips: you can access emd information through self.json_info.
        '''
        #Todo: fill in this method to load your model
        self.model = C.load_model(model_path)


    def getParameterInfo(self, required_parameters):
        required_parameters.extend(
            [
                # Todo: add your inference parameters here
                # https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#getparameterinfo
                {
                    'name': 'nms_threshold',
                    'dataType': 'numeric',
                    'value': 0.2,
                    'required': True,
                    'displayName': 'nms threshold',
                    'description': 'non maximum suppression(nms) threshold'
                },
            ]
        )
        return required_parameters

    def inference(self, batch, **scalars):
        '''
        Fill this method to write your own inference python code, you can refer to the model instance that is created
        in the load_model method. Expected results format is described in the returns as below.

        :param batch: numpy array with shape (B, H, W, D), B is batch size, H, W is specified and equal to
                      ImageHeight and ImageWidth in the emd file and D is the number of bands and equal to the length
                      of ExtractBands in the emd. If BatchInference is set to False in emd, B is constant 1.
        :param scalars: inference parameters, accessed by the parameter name,
                       i.e. score_threshold=float(scalars['score_threshold']). If you want to have more inference
                       parameters, add it to the list of the following getParameterInfo method.
        :return: bounding boxes, python list representing bounding boxes whose length is equal to B, each element is
                                 [N,4] numpy array representing [ymin, xmin, ymax, xmax] with respect to the upper left
                                 corner of the image tile.
                 scores, python list representing the score of each bounding box whose length is equal to B, each element
                         is [N,] numpy array
                 classes, python list representing the class of each bounding box whose length is equal to B, each element
                         is [N,] numpy array and its dype is np.uint8
        '''
        # Todo: fill in this method to inference your model and return bounding boxes, scores and classes
        nms_threshold = float(scalars['nms_threshold'])

        batch = batch.astype(np.float32)
        batch_size, bands, height, width = batch.shape

        dims = np.array([width, height,
                         width, height,
                         width, height], np.float32)
        output = self.model.eval(
            {
                self.model.arguments[0]: batch,
                self.model.arguments[1]: [dims]
            }
        )
        out_dict = dict([(k.name, k) for k in output])

        out_cls_pred = output[out_dict['cls_pred']][0]
        out_rpn_rois = output[out_dict['rpn_rois']][0]
        out_bbox_regr = output[out_dict['bbox_regr']][0]

        labels = out_cls_pred.argmax(axis=1)
        scores = out_cls_pred.max(axis=1)
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, dims)

        nmsKeepIndices = apply_nms_to_single_image_results(regressed_rois, labels, scores,
                                                           False,
                                                           0,
                                                           nms_threshold=nms_threshold,
                                                           conf_threshold=self.score_threshold)

        filtered_bboxes = regressed_rois[nmsKeepIndices]
        filtered_scores = scores[nmsKeepIndices]
        filtered_labels = labels[nmsKeepIndices]

        positive_label_indices = np.where(filtered_labels > 0)
        filtered_bboxes = filtered_bboxes[positive_label_indices]
        filtered_scores = filtered_scores[positive_label_indices]
        filtered_labels = filtered_labels[positive_label_indices]

        filtered_bboxes = filtered_bboxes[:, [1, 0, 3, 2]]

        return [filtered_bboxes], [filtered_scores], [filtered_labels]
