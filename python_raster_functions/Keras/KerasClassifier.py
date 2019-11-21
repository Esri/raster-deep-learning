import json
import keras.backend as K
from keras.models import load_model
import sys
import os
import random
import numpy as np
import tensorflow as tf
import skimage.color
import skimage.io
import skimage.transform
from distutils.version import LooseVersion
import arcpy

sys.path.append(os.path.dirname(__file__))

class ChildObjectDetector:
    def initialize(self, model, model_as_file):

        K.clear_session()

        if model_as_file:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        else:
            self.json_info = json.loads(model)

        model_path = self.json_info['ModelFile']
        if model_as_file and not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(os.path.dirname(model), model_path))

        if arcpy.env.processorType != "GPU":
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

        # load the trained model
        self.model = load_model(model_path)
        self.graph = tf.get_default_graph()

    def getParameterInfo(self, required_parameters):
        required_parameters.append(
            {
                'name': 'resampled_image_size',
                'dataType': 'numeric',
                'required': False,
                'value': 64,
                'displayName': 'Resampled Image Size',
                'description': 'Unified image size to feed in to the deep learning model'
            }
        )
        return required_parameters

    def getConfiguration(self, **scalars):
        self.resampleSize = int(scalars['resampled_image_size'])

        if 'BatchSize' not in self.json_info and 'batch_size' not in scalars:
            self.batch_size = 1
        elif 'BatchSize' not in self.json_info and 'batch_size' in scalars:
            self.batch_size = int(scalars['batch_size'])
        else:
            self.batch_size = int(self.json_info['BatchSize'])

        return {
            # CropSizeFixed is a boolean value parameter (1 or 0) in the emd file, representing whether the size of
            # tile cropped around the feature is fixed or not.
            # 1 -- fixed tile size, crop fixed size tiles centered on the feature. The tile can be bigger or smaller
            # than the feature;
            # 0 -- Variable tile size, crop out the feature using the smallest fitting rectangle. This results in tiles
            # of varying size, both in x and y. the ImageWidth and ImageHeight in the emd file are still passed and used
            # as a maximum cropped size limit. If the feature is bigger than the defined ImageWidth/ImageHeight, the tiles are cropped
            # the same way as in the "fixed tile size" option using the maximum cropped size limit.
            'CropSizeFixed': int(self.json_info['CropSizeFixed']),

            # BlackenAroundFeature is a boolean value paramater (1 or 0) in the emd file, representing whether blacken
            # the pixels outside the feature in each image tile.
            # 1 -- Blacken
            # 0 -- Not blacken
            'BlackenAroundFeature': int(self.json_info['BlackenAroundFeature']),

            'extractBands': tuple(self.json_info['ExtractBands']),
            'tx': self.json_info['ImageWidth'],
            'ty': self.json_info['ImageHeight'],
            'batch_size': self.batch_size
        }

    def vectorize(self, **pixelBlocks):

        # Pixelblocks - A tuple of 3-d rasters: ([bands,height,width],[bands,height.width],...)

        # Convert tuple to 4-d numpy array
        batch_images = np.asarray(pixelBlocks['rasters_pixels'])

        # Get the shape of the 4-d numpy array
        batch, bands, height, width = batch_images.shape

        # Transpose the image dimensions to [batch, height, width, bands]
        batch_images = np.transpose(batch_images, [0, 2, 3, 1])

        rings = []
        clf = self.model
        image_dim = self.resampleSize

        batch_images_resized = np.zeros((batch, image_dim, image_dim, bands), dtype=batch_images.dtype)

        for i in range(0, batch):

            # img.astype(image_dtype), window, scale, padding, crop
            image_resize, _, _, _, _ = self.resize_image(batch_images[i], min_dim=image_dim, max_dim=image_dim,
                                                     mode="square")
            [h, w, b] = image_resize.shape
            if h < image_dim or w < image_dim:
                image_resize_modified = np.zeros(shape=(image_dim, image_dim, b), dtype=batch_images.dtype)
                image_resize_modified[0:h, 0:w, :] = image_resize
            else:
                image_resize_modified = image_resize

            batch_images_resized[i] = image_resize_modified

            # Return boundaries of image tiles as the geometries. These geometries are ignored by the
            # GP tool if the input feature class is given.
            # Otherwise, if the input feature class is not given, it assumes that each image tile contains only
            # one object,these polygons are used as the output feature boundaries.
            rings.append([[[0, 0], [0, width - 1], [height - 1, width - 1], [height-1, 0]]])

        with self.graph.as_default():
            labels, confidences = self.predict(im=batch_images_resized, clf=clf)

        return rings, confidences, labels

    def predict(self, im, clf, image_dim = 64):
        # Need image to be in shape (?, 64, 64, 3)
        pr = clf.predict_on_batch(im)

        labels, confidences = [], []
        for i in range(0, len(pr)):
            if np.argmax(pr[i]) == 0:
                labels.append(self.json_info['Classes'][0]['Name'])
                confidences.append(pr[i][0])

            elif np.argmax(pr[i]) == 1:
                labels.append(self.json_info['Classes'][1]['Name'])
                confidences.append(pr[i][1])

        return labels, confidences


    def resize_image(self, img, min_dim=None, max_dim=None, min_scale=None, mode="square"):
            """Resizes an image keeping the aspect ratio unchanged.

            min_dim: if provided, resizes the image such that it's smaller
                dimension == min_dim
            max_dim: if provided, ensures that the image longest side doesn't
                exceed this value.
            min_scale: if provided, ensure that the image is scaled up by at least
                this percent even if min_dim doesn't require it.
            mode: Resizing mode.
                none: No resizing. Return the image unchanged.
                square: Resize and pad with zeros to get a square image
                    of size [max_dim, max_dim].
                pad64: Pads width and height with zeros to make them multiples of 64.
                       If min_dim or min_scale are provided, it scales the image up
                       before padding. max_dim is ignored in this mode.
                       The multiple of 64 is needed to ensure smooth scaling of feature
                       maps up and down the 6 levels of the FPN pyramid (2**6=64).
                crop: Picks random crops from the image. First, scales the image based
                      on min_dim and min_scale, then picks a random crop of
                      size min_dim x min_dim. Can be used in training only.
                      max_dim is not used in this mode.

            Returns:
            image: the resized image
            window: (y1, x1, y2, x2). If max_dim is provided, padding might
                be inserted in the returned image. If so, this window is the
                coordinates of the image part of the full image (excluding
                the padding). The x2, y2 pixels are not included.
            scale: The scale factor used to resize the image
            padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
            """
            # Keep track of image dtype and return results in the same dtype
            image_dtype = img.dtype
            # Default window (y1, x1, y2, x2) and default scale == 1.
            h, w = img.shape[:2]

            window = (0, 0, h, w)
            scale = 1
            padding = [(0, 0), (0, 0), (0, 0)]
            crop = None

            if mode == "none":
                return img, window, scale, padding, crop

            # Scale?
            if min_dim:
                # Scale up but not down
                scale = max(1, min_dim / min(h, w))
            if min_scale and scale < min_scale:
                scale = min_scale

            # Does it exceed max dim?
            if max_dim and mode == "square":
                image_max = max(h, w)
                if round(image_max * scale) > max_dim:
                    scale = max_dim / image_max

            # Resize image using bilinear interpolation
            if scale != 1:
                img = self.resize(img, (round(h * scale), round(w * scale)), preserve_range=True)

            # Need padding or cropping?
            if mode == "square":
                # Get new height and width
                h, w = img.shape[:2]
                top_pad = (max_dim - h) // 2
                bottom_pad = max_dim - h - top_pad
                left_pad = (max_dim - w) // 2
                right_pad = max_dim - w - left_pad
                padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
                image = np.pad(img, padding, mode='constant', constant_values=0)
                window = (top_pad, left_pad, h + top_pad, w + left_pad)
            elif mode == "pad64":
                h, w = img.shape[:2]
                # Both sides must be divisible by 64
                assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
                # Height
                if h % 64 > 0:
                    max_h = h - (h % 64) + 64
                    top_pad = (max_h - h) // 2
                    bottom_pad = max_h - h - top_pad
                else:
                    top_pad = bottom_pad = 0
                # Width
                if w % 64 > 0:
                    max_w = w - (w % 64) + 64
                    left_pad = (max_w - w) // 2
                    right_pad = max_w - w - left_pad
                else:
                    left_pad = right_pad = 0
                padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
                image = np.pad(img, padding, mode='constant', constant_values=0)
                window = (top_pad, left_pad, h + top_pad, w + left_pad)
            elif mode == "crop":
                # Pick a random crop
                h, w = img.shape[:2]
                y = random.randint(0, (h - min_dim))
                x = random.randint(0, (w - min_dim))
                crop = (y, x, min_dim, min_dim)
                img = img[y:y + min_dim, x:x + min_dim]
                window = (0, 0, min_dim, min_dim)
            else:
                raise Exception("Mode {} not supported".format(mode))
            return img.astype(image_dtype), window, scale, padding, crop


    def resize(self, img, output_shape, order=1, mode='constant', cval=0, clip=True,
                   preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
            """A wrapper for Scikit-Image resize().

            Scikit-Image generates warnings on every call to resize() if it doesn't
            receive the right parameters. The right parameters depend on the version
            of skimage. This solves the problem by using different parameters per
            version. And it provides a central place to control resizing defaults.
            """
            if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
                # New in 0.14: anti_aliasing. Default it to False for backward
                # compatibility with skimage 0.13.
                return skimage.transform.resize(
                    img, output_shape,
                    order=order, mode=mode, cval=cval, clip=clip,
                    preserve_range=preserve_range, anti_aliasing=anti_aliasing,
                    anti_aliasing_sigma=anti_aliasing_sigma)
            else:
                return skimage.transform.resize(
                    img, output_shape,
                    order=order, mode=mode, cval=cval, clip=clip,
                    preserve_range=preserve_range)
