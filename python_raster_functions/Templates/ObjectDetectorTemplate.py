import os
import sys

prf_root_dir = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(prf_root_dir)
from Templates.TemplateBaseDetector import TemplateBaseDetector

class ChildObjectDetector(TemplateBaseDetector):
    # template method to fill in
    def load_model(self, model_path):
        '''
        Fill this method to write your own model loading python code
        save it self object if you would like to reference it later.

        Tips: you can access emd information through self.json_info.

        TensorFlow example to import graph def from frozen pb file:
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        '''
        #Todo: fill in this method to load your model
        pass

    def getParameterInfo(self, required_parameters):
        required_parameters.extend(
            [
                # Todo: add your inference parameters here
                # https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#getparameterinfo
                '''
                Example:
                {
                    'name': 'nms_threshold',
                    'dataType': 'numeric',
                    'value': 0.2,
                    'required': True,
                    'displayName': 'nms threshold',
                    'description': 'non maximum suppression(nms) threshold'
                },
                '''
            ]
        )
        return required_parameters

    # template method to fill in
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
                         is [N,] numpy array and its dtype is np.uint8

        Tensorflow Example:
        score_threshold = float(scalars['score_threshold'])

        config = tf.ConfigProto()
        if 'PerProcessGPUMemoryFraction' in self.json_info:
            config.gpu_options.per_process_gpu_memory_fraction = float(self.json_info['PerProcessGPUMemoryFraction'])

        with self.detection_graph.as_default():
            with tf.Session(config=config) as sess:
                feed_dict = {
                    'image_tensor:0': batch
                }
                fetches = {
                    'boundingboxes': 'detection_boxes:0',
                    'scores': 'detection_scores:0',
                    'classes': 'detection_classes:0'
                }

                output_dict = sess.run(fetches, feed_dict=feed_dict)

        bounding_boxes = output_dict['boundingboxes']
        scores = output_dict['scores']
        classes = output_dict['classes']

        bounding_boxes[:, :, [0, 2]] = bounding_boxes[:, :, [0, 2]] * self.json_info['ImageHeight']
        bounding_boxes[:, :, [1, 3]] = bounding_boxes[:, :, [1, 3]] * self.json_info['ImageWidth']

        batch_bounding_boxes, batch_scores, batch_classes = [], [], []

        batch_size = bounding_boxes.shape[0]
        for batch_idx in range(batch_size):
            keep_indices = np.where(scores[batch_idx]>score_threshold)
            batch_bounding_boxes.append(bounding_boxes[batch_idx][keep_indices])
            batch_scores.append(scores[batch_idx][keep_indices])
            batch_classes.append(classes[batch_idx][keep_indices])

        return batch_bounding_boxes, batch_scores, batch_classes
        '''
        #Todo: fill in this method to inference your model and return bounding boxes, scores and classes
        pass
