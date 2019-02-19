# Esri Model Definition file

Esri Model Definition(emd) file is a configuration file in JSON format that tells ArcGIS which Python Raster Function to 
use(indicated by *InferenceFunction*). It is also an argument (as file path) to the Python Raster Function.
Thus you can put configurable and meta data in the emd and access the data from the Python Raster Function, for example, model 
location. Take the built-in python raster functions for example, we put the height and width of the training image chips 
in the emd file. Then the raster functions access this dimension information and ask ArcGIS to hand over the 
correct size pixel blocks through the python raster function [*getConfiguration*](https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#getconfiguration)
api method. 

An emd file is a json formatted file. Here is an example.

```json
{
    "Framework": "TensorFlow",
    "ModelConfiguration": "ObjectDetectionAPI",
    "ModelFile":"\\\\pa\\raster\\DataSwap\\Han\\PRFTestData\\ObjectDetection\\Tree\\tensorflow\\exported_graphs_tree\\frozen_inference_graph.pb",
    "ModelType":"ObjectionDetection",
    "InferenceFunction":"[Functions]\\CustomObjectDetector.py",
    "ImageHeight":850,
    "ImageWidth":850,
    "ExtractBands":[0,1,2],

    "Classes" : [
      {
        "Value": 0,
        "Name": "Tree",
        "Color": [0, 255, 0]
      }
    ]
}
```

## Keywords that are used by ArcGIS Pro and mandatory:
- "ModelType"

  Type: String.

  The type of the model: "ImageClassification" for classifying pixels, and "ObjectDetection" for detecting objects.
 
- "InferenceFunction"

  Type: String.

  The path of a custom inference Python function. If not specified, will use default built-in python raster function. 
  
## Keywords that are used by built-in python raster functions and the templates:
- "Framework"

  Type: String.

  The name of a deep learning framework used to train your model, e.g. "TensorFlow", "CNTK", "PyTorch", and etc. Use 
  "Templates" if you use built-in templates and fill in your own python code.

- "ModelConfiguration"

  Type: String or JSON object.

  The name of a model configuration. A model configuration specifies a model trained by a well-known python implementation.
  There are many existing open source deep learning projects that define "standard" inputs and outputs 
  configuration, and inference logic. ArcGIS built-in python raster function support a set of predefined such well-known 
  configurations. The number of supported model configurations/architectures is increasing. 

  If you train your deep learning model by using one of existing open source code projects or model configurations, 
  you can select a model architecture/configuration. The current supported model architectures and their project pages are:

  **TensorFlow**
  - "ObjectDetectionAPI", https://github.com/tensorflow/models/tree/master/research/object_detection
  - "DeepLab", https://github.com/tensorflow/models/tree/master/research/deeplab
  
  **PyTorch**
  - 'FastaiSSD', https://github.com/Esri/arcgis-python-api/tree/master/talks/uc2018/Plenary/pools
  
  **CNTK**
  - "FasterRCNN", https://docs.microsoft.com/en-us/cognitive-toolkit/object-detection-using-faster-r-cnn
  - "AzurePixelLevelLandClassification", https://github.com/Azure/pixel_level_land_classification
    
  **Keras**
  - "MaskRCNN", https://github.com/matterport/Mask_RCNN
  
    In this MaskRCNN configuration case, you also need to specify the "Architecture" python module where a python 
    MaskRCNN object with name "model" should be declared and also the "Config" python module where a python 
    configuration object with name "conig" is declared. See [Mask RCNN example](../examples/keras/mask_rcnn/README.md)
    for more details.
    ```json
    "ModelConfiguration": {
      "Name":"MaskRCNN",
      "Architecture":".\\mrcnn\\spacenet",
      "Config":".\\mrcnn\\spacenet"
    },
    ```

  **Templates**
  - "ImageClassifierTemplate"
    If you fill in the image classifier template with your own python code.
  
  - "ObjectDetectorTemplate"
    If your fill in the object detector template with your own python code.

- "ModelFile"

  Type: String.

  The path of a trained deep learning model file.
  
- "ImageHeight"

  Type: Int.

  The image height of the deep learning model.
  
- "ImageWidth"

  Type: Int.

  The image width of the deep learning model.
  
- "ExtractBands"

  Type: Array of ints or strings.

  The array of band indexes or band names to extract from the input raster/imagery.
  
- "DataRange"

  Type: Array of two elements.
  
  This represents the minimum and maximum value the deep learning model expects. ArcGIS will 
  rescale the input raster tile to this data range if this is defined by the actual range of the input raster.
  
- "BatchSize"

  Type: Integer.
  
  Specify the batch size you want to inference the model. If missing, a parameter batch_size is added to the python
  raster function so the batch size is specified every time the tool runs.
 
- "ModelPadding"
 
  Type: Integer.
  
  This is only used in pixel-level classification. If the model has a padding itself, for example, if a model outputs
  the center 128x128 segmentation pixels given an input tile 256x256, the model has a padding of 64.

## emd for custom python raster functions
If you don't train your model using the well-known model configurations listed above thus have your own inference function. You can either
use our provided built-in templates if this could make your work easier or point "InferenceFunction" to your own python 
raster function. In any way the built-in python raster functions are always good references.
