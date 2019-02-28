# Deep Learning Python Raster Functions For ArcGIS

Deep learning model inferences in ArcGIS are implemented on top of the python raster function framework. 
This repository is served as a harbor to provide guidance on deep learning python raster functions in ArcGIS,
 and most important to educate the users on creating custom python raster functions to integrate more deep learning 
 models into ArcGIS.   

## Prerequisites
1. The users need to install either of the following ArcGIS platforms:
    - *ArcGIS Pro 2.3 or later*.
    - *ArcGIS Enterprise 10.7 or later*.

2. Esri Image Analyst license

3. CPU or NVIDIA GPU + CUDA CuDNN

## Getting started
1. Clone raster-deep-learning repository: 
   ```
   git clone https://github.com/Esri/raster-deep-learning.git 
   ```
   
2. Download sample images and sample trained deep learning models (Link is TBD)

3. Set up python deep learning environment in ArcGIS, and install required deep learning libraries in *[requirements.txt](requirements.txt)*.
   
   - ArcGIS Pro users can install through *[ArcGIS Pro Python Package Manager](http://pro.arcgis.com/en/pro-app/arcpy/get-started/what-is-conda.htm)*.
   
   - Alternatively, ArcGIs Pro users can use this batch file *[env_setup.bat](env_setup.bat)* at root directory to automatically install 
   all the required deep learning libraries for the sample cases in this repository into ArcGIS Pro. 
   Note: use it with caution, in default batch installation will take 5 GB disk space on C: drive. To execute the batch file, 
   right click the batch file and run as administrator. This batch file takes less than 5 minutes to run on the author's test machine. 
   
   - For ArcGIS Enterprise customers, the deep learning libraries need to be installed to the directory 
   *C:\ArcGIS\Server\framework\runtime\ArcGIS\bin\Python* on each raster analytics server machine using command lines.
   Here is an example to install tensorflow:
       ```
       Step1: Change Directory to ArcGIS python scripts directory.
       cd C:\ArcGIS\Server\framework\runtime\ArcGIS\bin\Python\Scripts
       
       Step2: Clone a new ArcGIS python environment for deep learning.
       conda create --name deeplearning_env_name --clone arcgispro-py3
       
       Step3: Activate the new python environment.
       activate deeplearning_env_name
       
       Step4: Install tensorflow or tensorflow-gpu in the new python environment. 
       conda install tensorflow-gpu
       
       Step5: Swap it to the activate python environment of the ArcGIS servers. 
       proswap deeplearning_env_name
       ```

4. Understand *[Esri model definition file (emd)](docs/writing_model_definition.md)*.

5. (Optional) Understand *python raster functions* through this document
*[Anatomy of a Python Raster Function](https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#anatomy-of-a-python-raster-function)*.  
 
6. (Optional) If you are interested in writing custom deep learning python raster functions to integrate deep learning
models into ArcGIS, 
*[Custom python raster functions guide](docs/writing_deep_learning_python_raster_functions.md)* provides details 
on the functions to be called and how to implement model inference call in python raster function.    

## Examples

We provide deep learning model inference python raster functions (PRFs) for a list of open source deep learning model configurations.

*Note: Click on the model configuration name will go to the corresponding GitHub repository or project landing page of this model config.*

* *[Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)*.

    [**[Sample case](examples/tensorflow/object_detection/coconut_tree_detection/README.md) | 
    [Sample emd file](examples/tensorflow/object_detection/coconut_tree_detection/tensorflow_objectdetectionapi_coconuttree.emd) |
    [Build-in PRF](python_raster_functions/TensorFlow/ObjectDetectionAPI.py)**]

* *[Tensorflow DeepLab for Semantic Image Segmentation](https://github.com/tensorflow/models/tree/master/research/deeplab)*
    
    [**[Sample case](examples/tensorflow/image_classification/land_cover_classification/README.md) | 
    [Sample emd file](examples/tensorflow/image_classification/land_cover_classification/tensorflow_deeplab_landclassification.emd) |
    [Build-in PRF](python_raster_functions/TensorFlow/DeepLab.py)**]
    
* [Cognitive Toolkit (CNTK) object detection using Faster R-CNN](https://docs.microsoft.com/en-us/cognitive-toolkit/object-detection-using-faster-r-cnn)
    
    [**[Sample case](examples/cntk/object_detection/coconut_tree/README.md) | 
    [Sample emd file](examples/cntk/object_detection/coconut_tree/cntk_fasterrcnn_coconut_tree.emd) |
    [Build-in PRF](python_raster_functions/CNTK/FasterRCNN.py)**]
    
* [Azure pixel-level land cover classification on Cognitive Toolkit (CNTK)](https://github.com/Azure/pixel_level_land_classification)

    [**[Sample case](examples/cntk/image_classification/land_classification/README.md) | 
    [Sample emd file](examples/cntk/image_classification/land_classification/azure_pixel_level_land_classification.emd) |
    [Build-in PRF](python_raster_functions/CNTK/AzurePixelLevelLandClassification.py)**]
    
* [Keras Mask R-CNN for object detection and instance segmentation](https://github.com/matterport/Mask_RCNN)
  
    [**[Sample case](examples/keras/mask_rcnn/README.md) | 
    [Sample emd file](examples/keras/mask_rcnn/mask_rcnn.emd) |
    [Build-in PRF](python_raster_functions/Keras/MaskRCNN.py)**]

* [Fast.ai SSD implementation on PyTorch](https://github.com/Esri/arcgis-python-api/tree/master/talks/uc2018/Plenary/pools)

    [**[Sample case](examples/pytorch/README.md) | 
    [Sample emd file](examples/pytorch/pytorch_fastai_ssd.emd) |
    [Build-in PRF](python_raster_functions/PyTorch/FastaiSSD.py)**]

## Features
Python raster function templates are provided for bounding box based object detection and pixel-level image classification
(Image Segmentation in Computer Vision) 

## [Frequently Asked Questions](docs/questions_and_answers.md)

## Resources

* [The raster function Wiki](https://github.com/Esri/raster-functions/wiki)

## Issues

Find a bug or want to request a new feature?  Please let us know by submitting an issue.

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

A copy of the license is available in the repository's [license.txt]( license.txt) file.