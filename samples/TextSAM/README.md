# TextSAM DLPK For ArcGIS Pro

This sample showcases a deep learning package (DLPK) to generate masks for an object using text prompts in ArcGIS Pro. It is achieved by using Segment Anything Model (SAM) and GroundingDINO. Grounding DINO is an open set object detector that can find objects given a text prompt. The bounding boxes representing the detected objects are then fed into Segment Anything Model as prompts to generate masks for those. Finally, the masks are converted to polygons by this model and returned as GIS features. Both models are called sequentially within this deep learning package.

## Prerequisites
1. ArcGIS Pro 2.3 or later

2. An ArcGIS Image Analyst license is required to run inferencing tools.

3. CPU or NVIDIA GPU + CUDA CuDNN

4. Set up the [Python deep learning environment in ArcGIS](https://developers.arcgis.com/python/guide/deep-learning/). 

5. Get familiar with the format and requirements of the *[Esri model definition file (emd)](../../docs/writing_model_definition.md)*.

6. (Optional) Get familiar with *Python raster function*.
*[Anatomy of a Python Raster Function](https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#anatomy-of-a-python-raster-function)*.  
 
7. (Optional) If you are interested in writing custom deep learning Python raster function to integrate additional deep learning
models into ArcGIS, the
*[Custom Python raster function guide](../../docs/writing_deep_learning_python_raster_functions.md)* provides details 
on the necessary functions and how to implement model inference calls using Python raster function.

## Steps for creating DLPK
1. Clone GroundingDINO repository:
   ```
   git clone https://github.com/giswqs/GroundingDINO.git 
   ```
   
2. Install the GroundingDINO from the repository downloaded above by running the command below:
   ```
   pip install --no-deps -e . 
   ```

3. Install the supervision package from pypi using the command:
   ```
   pip install --no-deps supervision==0.6.0
   ```

4. Install the segment anything from pypi using the command:
   ```
   pip install segment-anything
   ```

5. Download the Segment Anything Model (SAM) checkpoint from the [repo](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) and point to it in the [TextSAM.py](TextSAM.py) file.

6. Download the GroundingDINO checkpoint and Config file from the [repo](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#luggage-checkpoints) and point to it in the [TextSAM.py](TextSAM.py) file.

7. Finally, to create a standalone DLPK, you may create a 7zip archive containing the model weights, source code of grounding_dino and segment_anything, along with the EMD and .py inference function from this repo. The archive should be given a DLPK extension.

## Issues

Did you find a bug or do you want to request a new feature?  Please let us know by submitting an issue.

## Contributing

Esri welcomes contributions from anyone and everyone. Please see our [guidelines for contributing](https://github.com/esri/contributing).

## Licensing
Copyright 2019 Esri

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

A copy of the license is available in the repository's [license.txt](../../license.txt) file.