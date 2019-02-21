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

class TemplateBaseClassifier:
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

        if 'ModelPadding' in self.json_info:
            self.model_padding = self.json_info['ModelPadding']
            self.padding = self.model_padding
        else:
            self.model_padding = 0
            self.padding = int(scalars['padding'])

        self.scalars = scalars

        self.rectangle_height, self.rectangle_width = prf_utils.calculate_rectangle_size_from_batch_size(self.batch_size)
        ty, tx = prf_utils.get_tile_size(self.json_info['ImageHeight'], self.json_info['ImageWidth'],
                                         self.padding, self.rectangle_height, self.rectangle_width)

        return {
            'extractBands': tuple(self.json_info['ExtractBands']),
            'padding': self.padding,
            'tx': tx,
            'ty': ty,
            'fixedTileSize': 1
        }

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        input_image = pixelBlocks['raster_pixels']
        _, height, width = input_image.shape
        batch, batch_height, batch_width = \
            prf_utils.tile_to_batch(input_image,
                                    self.json_info['ImageHeight'],
                                    self.json_info['ImageWidth'],
                                    self.padding,
                                    fixed_tile_size=True,
                                    batch_height=self.rectangle_height,
                                    batch_width=self.rectangle_width)

        semantic_predictions = self.inference(batch, **self.scalars)

        height, width = semantic_predictions.shape[2], semantic_predictions.shape[3]
        if self.model_padding == 0 and self.padding != 0:
            semantic_predictions = semantic_predictions[:, :, self.padding:height - self.padding,
                                   self.padding:width - self.padding]

        semantic_predictions = prf_utils.batch_to_tile(semantic_predictions, batch_height, batch_width)

        return semantic_predictions
