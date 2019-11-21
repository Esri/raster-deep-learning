from __future__ import division
import os
import sys
import json
import warnings
from fastai.vision import *
from torchvision import models as torchvision_models
import arcgis
from arcgis.learn import FeatureClassifier
import arcpy
import torch
from fastai.metrics import accuracy
import tempfile
from pathlib import Path
prf_root_dir = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(prf_root_dir)
import numpy as np

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

imagenet_mean = 255 * np.array(imagenet_stats[0], dtype=np.float32)
imagenet_std = 255 * np.array(imagenet_stats[1], dtype=np.float32)

def norm(x, mean=imagenet_mean, std=imagenet_std):
    return (x - mean)/std

def denorm(x, mean=imagenet_mean, std=imagenet_std):
    return x * std + mean

class ChildObjectDetector:

    def initialize(self, model, model_as_file):
        if model_as_file:
            with open(model, 'r') as f:
                self.emd = json.load(f)
        else:
            self.emd = json.loads(model)

        if arcpy.env.processorType == "GPU" and torch.cuda.is_available():
            self.device = torch.device('cuda')
            arcgis.env._processorType = "GPU"
        else:
            self.device = torch.device('cpu')
            arcgis.env._processorType = "CPU"

        # Using arcgis.learn FeatureClassifer from_model function.
        self.cf = FeatureClassifier.from_model(emd_path=model)
        self.model = self.cf.learn.model
        self.model.eval()

    def getParameterInfo(self, required_parameters):
        return required_parameters

    def getConfiguration(self, **scalars):

        if 'BatchSize' not in self.emd and 'batch_size' not in scalars:
            self.batch_size = 1
        elif 'BatchSize' not in self.emd and 'batch_size' in scalars:
            self.batch_size = int(scalars['batch_size'])
        else:
            self.batch_size = int(self.emd['BatchSize'])

        return {
            # CropSizeFixed is a boolean value parameter (1 or 0) in the emd file, representing whether the size of
            # tile cropped around the feature is fixed or not.
            # 1 -- fixed tile size, crop fixed size tiles centered on the feature. The tile can be bigger or smaller
            # than the feature;
            # 0 -- Variable tile size, crop out the feature using the smallest fitting rectangle. This results in tiles
            # of varying size, both in x and y. the ImageWidth and ImageHeight in the emd file are still passed and used
            # as a maximum size. If the feature is bigger than the defined ImageWidth/ImageHeight, the tiles are cropped
            # the same way as in the fixed tile size option using the maximum size.
            'CropSizeFixed': int(self.emd['CropSizeFixed']),

            # BlackenAroundFeature is a boolean value paramater (1 or 0) in the emd file, representing whether blacken
            # the pixels outside the feature in each image tile.
            # 1 -- Blacken
            # 0 -- Not blacken
            'BlackenAroundFeature': int(self.emd['BlackenAroundFeature']),

            'extractBands': tuple(self.emd['ExtractBands']),
            'tx': self.emd['ImageWidth'],
            'ty': self.emd['ImageHeight'],
            'batch_size': self.batch_size
        }

    def vectorize(self, **pixelBlocks):

        # Get pixel blocks - tuple of 3-d rasters: ([bands,height,width],[bands,height.width],...)
        # Convert tuple to 4-d numpy array
        batch_images = np.asarray(pixelBlocks['rasters_pixels'])

        # Get the shape of the 4-d numpy array
        batch, bands, height, width = batch_images.shape

        # Transpose the image dimensions to [batch, height, width, bands]
        batch_images = np.transpose(batch_images, [0, 2, 3, 1])

        rings = []
        labels, confidences = [], []

        # Convert to torch tensor and transpose the dimensions to [batch, bands, height, width]
        batch_images = torch.tensor(norm(batch_images).transpose(0, 3, 1, 2)).to(self.device)

        # the second element in the passed tuple is hardcoded to make fastai's pred_batch work
        predictions = self.cf.learn.pred_batch(batch=(batch_images, torch.tensor([40]).to(self.device)))

        # torch.max returns the max value and the index of the max as a tuple
        confidences, class_idxs = torch.max(predictions, dim=1)

        # Using emd to map the class
        class_map = [c['Name'] for c in self.emd["Classes"]]
        labels = [class_map[c] for c in class_idxs]

        # Appending this ring for all the features in the batch
        rings = [[[[0, 0], [0, width - 1], [height - 1, width - 1], [height - 1, 0]]] for i in range(self.batch_size)]

        return rings, confidences.tolist(), labels
