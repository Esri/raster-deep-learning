import os
import sys

prf_root_dir = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(prf_root_dir)
from Templates.TemplateBaseDetector import TemplateBaseDetector

sys.path.append(os.path.dirname(__file__))

# from util import load_model, norm, denorm, export_img, get_tile_images, get_img, \
#     get_cropped_tiles, predict_, detect_objects, suppress_close_pools, overlap, predict_classf, \
#     get_nms_preds
import util
from model import ConvnetBuilder, SSD_MultiHead, resnet34, k

class ChildObjectDetector(TemplateBaseDetector):
    def load_model(self, model_path):
        '''
        Fill this method to write your own model loading python code
        save it self object if you would like to reference it later.

        Tips: you can access emd information through self.json_info.
        '''
        # Todo: fill in this method to load your model
        f_model = resnet34

        head_reg = SSD_MultiHead(k, -4.)
        models = ConvnetBuilder(f_model, 0, 0, 0, custom_head=head_reg)
        self.model = models.model
        self.model = util.load_model(self.model, model_path)
        self.model.eval()

    def getParameterInfo(self, required_parameters):
        required_parameters.extend(
            [
                # Todo: add your inference parameters here
                # https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#getparameterinfo
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
                       i.e. score_threshold=float(kwargs['score_threshold']). If you want to have more inference
                       parameters, add it to the list of the following getParameterInfo method.
        :return: bounding boxes, python list representing bounding boxes whose length is equal to B, each element is
                                 [N,4] numpy array representing [ymin, xmin, ymax, xmax] with respect to the upper left
                                 corner of the image tile.
                 scores, python list representing the score of each bounding box whose length is equal to B, each element
                         is [N,] numpy array
                 classes, python list representing the class of each bounding box whose length is equal to B, each element
                         is [N,] numpy array and its dype is np.uint8
        '''
        #Todo: fill in this method to inference your model and return bounding boxes, scores and classes
        return util.detect_objects_image_space(self.model, batch, self.score_threshold)
