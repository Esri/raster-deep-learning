# Deep Learning Python Raster Functions For ArcGIS

Deep learning model inferences in ArcGIS are implemented on top of the python raster function framework. 
This repository is served as a harbor to provide guidance on build-in deep learning python raster functions in ArcGIS,
 and most important to educate the users to create custom python raster functions to integrate deep learning 
 workflows into ArcGIS.   

## Prerequisites
1. The users need to install *either* of the following ArcGIS platforms:
    - *ArcGIS Pro 2.3 or later*.
    - *ArcGIS 10.7 or later*.
    
2. CPU or NVIDIA GPU + CUDA CuDNN

## Getting started
1. Clone raster-deep-learning repository: 
   ```
   git clone https://github.com/Esri/raster-deep-learning.git 
   ```
2. Set up python environment for deep learning in ArcGIS.
   - Install required deep learning libraries through *[ArcGIS Pro Python Package Manager](http://pro.arcgis.com/en/pro-app/arcpy/get-started/what-is-conda.htm)*.
   
   - Alternatively, the batch file *[env_setup.bat](env_setup.bat)* at root directory automatically creates a new python environment
   in ArcGIS setup folder and installs all the required deep learning libraries and dependencies within this repository. 
   Use with caution since the installation needs around 5 GB disk space on C: drive. Right click the batch file and run as administrator. 
   The batch script run takes less than 5 minutes on the author's machine. 

3. Understand *[Esri model definition file (emd)](docs/writing_model_definition.md)*.

4. (Optional) Understand *python raster functions* through reading the document
*[Anatomy of a Python Raster Function](https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#anatomy-of-a-python-raster-function)*.  
 
5. (Optional) If your interest is to write custom deep learning model inference python raster functions, 
*[Custom python raster functions guide](docs/writing_deep_learning_python_raster_functions.md)* provides details 
on each function to be called, and how to implement model inference call in python raster function.    

## Features

* Support inference well-known implementation of deep learning models in ArcGIS
* Python raster function templates for bounding box based object detection and pixel-level image classification
(Image Segmentation in Computer Vision) 

## Examples 

## Frequently Asked Questions

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