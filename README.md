# Deep Learning Python raster function For ArcGIS

Deep learning model inferencing in ArcGIS is implemented on top of the Python raster function framework. 
This repository serves to provide guidance on deep learning Python raster functions in ArcGIS,
 and to educate users on how to create custom Python raster functions to integrate additional deep learning 
 models with the ArcGIS platform.   

## Prerequisites
1. Deep learning in ArcGIS requires one of the following products:
    - *ArcGIS Pro 2.3 or later*.
    - *ArcGIS Enterprise 10.7 or later*.

2. An ArcGIS Image Analyst license is required to run inferencing tools.

3. CPU or NVIDIA GPU + CUDA CuDNN

## Getting started
1. Clone raster-deep-learning repository: 
   ```
   git clone https://github.com/Esri/raster-deep-learning.git 
   ```
   
2. [Download sample data, including imagery and trained deep learning models](https://www.arcgis.com/apps/MinimalGallery/index.html?appid=99c39f7512d54881bc365583c76c7da6).
   
   - Use the link above to download the sample data and models for four different model frameworks.
3. Set up the Python deep learning environment in ArcGIS. 

   Starting from ArcGIS Pro 2.6 and ArcGIS Enterprise 10.8.1, *[Deep Learning Libraries Installers for ArcGIS](https://github.com/esri/deep-learning-frameworks)* are provided, 
   which include a broad collection of components, such as PyTorch, TensorFlow, Fast.ai, that are required to run Deep Learning workflows in this repository. 
   Users can download the installers from this GitHub repository: https://github.com/Esri/deep-learning-frameworks.   

   For users on ArcGIS Pro 2.5 and ArcGIS Enterprise 10.8, or older. Please install the required deep learning libraries listed in *[requirements.txt](requirements.txt)* using the following steps:
   
   - Install Python environments in ArcGIS Pro using the *[ArcGIS Pro Python Package Manager](http://pro.arcgis.com/en/pro-app/arcpy/get-started/what-is-conda.htm)*.
   
   - Alternatively, ArcGIS Pro users can use this batch file *[env_setup.bat](env_setup.bat)* at the root directory to automatically install 
   all the required deep learning libraries for the sample cases in this repository. 
   NOTE: By default, the batch installation will take up 5 GB of disk space on the C:/ drive (or the disk where ArcGIS Pro is installed). 
   To execute the batch file, right click the batch file and choose to Run as Administrator. This installation takes less than 5 minutes to run on the author's test machine. 
   
   - For ArcGIS Enterprise users, the deep learning libraries need to be installed to the Python environment on each raster analytics server machine: 
   *C:\ArcGIS\Server\framework\runtime\ArcGIS\bin\Python*. 
   The batch file *[env_setup_server_tensorflow.bat](env_setup_server_tensorflow.bat)* at root directory automates the TensorFlow environment setup process for ArcGIS Enterprise 10.7, specifically. 
   It is not recommended to install different deep learning frameworks (TensorFlow, Keras, PyTorch, etc) into one Enterprise Python environment.   
   NOTE: By default, batch installation will take ~2-3 GB of disk space on the C:/ drive (or the disk where ArcGIS server is installed). To execute the batch file, 
   right click the batch file and choose to Run as Administrator. This installation takes less than 5 minutes to run on the author's test machine.
   Once you finish setting up the Python environment, you will need to restart the ArcGIS server. 
   For more details on setting up the Enterprise environment, see [Configure ArcGIS Image Server for deep learning raster analytics](https://enterprisedev.arcgis.com/en/portal/latest/administer/windows/configure-and-deploy-arcgis-enterprise-for-deep-learning-raster-analytics.htm).
    
   - If you prefer to manually setup the deep learning Python environment using command lines, see the example below which 
    installs TensorFlow on the ArcGIS raster analytics server:
       ```
       Step1: Change Directory to ArcGIS python scripts directory.
       cd C:\ArcGIS\Server\framework\runtime\ArcGIS\bin\Python\Scripts
       
       Step2: Clone a new ArcGIS python environment for deep learning.
       conda create --name deeplearning_env_name --clone arcgispro-py3
       
       Step3: Activate the new python environment.
       activate deeplearning_env_name
       
       Step4: Install tensorflow or tensorflow-gpu in the new python environment. 
       conda install tensorflow
       
       Step5: Swap it to the activate python environment of the ArcGIS servers. 
       proswap deeplearning_env_name
       ```
       
4. Get familiar with the format and requirements of the *[Esri model definition file (emd)](docs/writing_model_definition.md)*.

5. (Optional) Get familiar with *Python raster function*.
*[Anatomy of a Python Raster Function](https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#anatomy-of-a-python-raster-function)*.  
 
6. (Optional) If you are interested in writing custom deep learning Python raster function to integrate additional deep learning
models into ArcGIS, the
*[Custom Python raster function guide](docs/writing_deep_learning_python_raster_functions.md)* provides details 
on the necessary functions and how to implement model inference calls using Python raster function.    

## Examples

We provide deep learning model inference Python raster function (PRF) for a list of open source deep learning model configurations.

*Click on the model configuration name to go to the corresponding GitHub repository or project landing page for that model.*

* *[Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)*.

    [**[Sample case](examples/tensorflow/object_detection/coconut_tree_detection/README.md) | 
    [Sample emd file](examples/tensorflow/object_detection/coconut_tree_detection/tensorflow_objectdetectionapi_coconuttree.emd) |
    [Sample PRF](python_raster_functions/TensorFlow/ObjectDetectionAPI.py)**]

* *[Tensorflow DeepLab for Semantic Image Segmentation](https://github.com/tensorflow/models/tree/master/research/deeplab)*
    
    [**[Sample case](examples/tensorflow/image_classification/land_cover_classification/README.md) | 
    [Sample emd file](examples/tensorflow/image_classification/land_cover_classification/tensorflow_deeplab_landclassification.emd) |
    [Sample PRF](python_raster_functions/TensorFlow/DeepLab.py)**]
    
* [Keras Mask R-CNN for object detection and instance segmentation](https://github.com/matterport/Mask_RCNN)
  
    [**[Sample case](examples/keras/mask_rcnn/README.md) | 
    [Sample emd file](examples/keras/mask_rcnn/mask_rcnn.emd) |
    [Sample PRF](python_raster_functions/Keras/MaskRCNN.py)**]

* [Keras image classification](https://www.geeksforgeeks.org/python-image-classification-using-keras/) (For ArcGIS Pro 2.5 and Enterprise 10.8 and later)
  
    [**[Sample case](examples/keras/object_classification/README.md) | 
    [Sample emd file](examples/keras/object_classification/model/HouseDamageClassifier_ProBuiltin.emd) |
    [Sample PRF](python_raster_functions/Keras/KerasClassifier.py)**]

* [Fast.ai SSD implementation with PyTorch](https://github.com/Esri/arcgis-python-api/tree/master/talks/uc2018/Plenary/pools)

    [**[Sample case](examples/pytorch/object_detection/README.md) | 
    [Sample emd file](examples/pytorch/object_detection/pytorch_fastai_ssd.emd) |
    [Sample PRF](python_raster_functions/PyTorch/FastaiSSD.py)**]

* [Object Classification implementation with PyTorch](https://developers.arcgis.com/python/sample-notebooks/building-damage-assessment-using-feature-classifier/) (For ArcGIS Pro 2.5 and Enterprise 10.8 and later)

    [**[Sample case](examples/pytorch/object_classification/README.md) | 
    [Sample emd file](examples/pytorch/object_classification/woolseyFire_600_50.emd) |
    [Sample PRF](python_raster_functions/PyTorch/FeatureClassifier.py)**]

## Resources

* ArcGIS Learn Lesson:
  - [Use Deep Learning to Assess Palm Tree Health](https://learn.arcgis.com/en/projects/use-deep-learning-to-assess-palm-tree-health/)
* [The raster function Wiki](https://github.com/Esri/raster-functions/wiki)

## Features
Python Raster function templates are provided for object detection and pixel-level image classification
(Image Segmentation in Computer Vision). 

## [Frequently Asked Questions](docs/questions_and_answers.md)

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

A copy of the license is available in the repository's [license.txt]( license.txt) file.