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

import json
import os
import sys

prf_root_dir = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(prf_root_dir)
import prf_utils

class TemplateBaseDetector:
    def initialize(self, model, model_as_file):
        if model_as_file:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        else:
            self.json_info = json.loads(model)

        model_path = self.json_info['ModelFile']
        if model_as_file and not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(os.path.dirname(model), model_path))

        # load model
        self.load_model(model_path)

    def getConfiguration(self, **scalars):

        if 'BatchSize' not in self.json_info and 'batch_size' not in scalars:
            self.batch_size = 1
        elif 'BatchSize' not in self.json_info and 'batch_size' in scalars:
            self.batch_size = int(scalars['batch_size'])
        else:
            self.batch_size = int(self.json_info['BatchSize'])

        self.padding = int(scalars['padding'])
        self.score_threshold = float(scalars['score_threshold'])

        self.scalars = scalars

        self.rectangle_height, self.rectangle_width = prf_utils.calculate_rectangle_size_from_batch_size(self.batch_size)
        ty, tx = prf_utils.get_tile_size(self.json_info['ImageHeight'], self.json_info['ImageWidth'],
                                         self.padding, self.rectangle_height, self.rectangle_width)

        return {
            'extractBands': tuple(self.json_info['ExtractBands']),
            'padding': int(scalars['padding']),
            'tx': tx,
            'ty': ty,
            'fixedTileSize': 0
        }

    def vectorize(self, **pixelBlocks):
        input_image = pixelBlocks['raster_pixels']
        _, height, width = input_image.shape
        batch, batch_height, batch_width = \
            prf_utils.tile_to_batch(input_image,
                                    self.json_info['ImageHeight'],
                                    self.json_info['ImageWidth'],
                                    self.padding,
                                    fixed_tile_size=False)

        batch_bounding_boxes, batch_scores, batch_classes = self.inference(batch, **self.scalars)
        return prf_utils.batch_detection_results_to_tile_results(
            batch_bounding_boxes,
            batch_scores,
            batch_classes,
            self.json_info['ImageHeight'],
            self.json_info['ImageWidth'],
            self.padding,
            batch_width
        )
