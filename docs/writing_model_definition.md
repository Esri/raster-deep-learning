# Esri Model Definition file

The Esri Model Definition file (EMD) is a configuration file in JSON format that provides the parameters to use 
for deep learning model inference in ArcGIS. Importantly, the EMD contains a parameter 
called "InferenceFunction" which specifies the custom Python raster function you want to use. Therefore, the EMD also 
serves as an argument (as file path) to the Python raster function. The EMD includes model properties and metadata 
that can be accessed from the Python raster function (for example, the location of the trained model file).

In the built-in Python raster functions examples, we include the height and width of the training image chips 
in the EMD file. Then, the raster functions can access this information through the getConfiguration method
to ensure that ArcGIS delivers the pixel blocks of the correct size. 

```json
{
    "Framework": "TensorFlow",
    "ModelConfiguration": "ObjectDetectionAPI",
    "ModelFile":".\\frozen_inference_graph.pb",
    "ModelType":"ObjectionDetection",
    "InferenceFunction":".\\CustomObjectDetector.py",
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
## Keywords supported in built-in Python raster functions:
- "Framework"

  Type: String.

  The name of a deep learning framework used to train your model, e.g. "TensorFlow", "CNTK", "PyTorch", and etc. Use 
  "Templates" if you use built-in templates and fill in your own python code.

- "ModelConfiguration"

  Type: String or JSON object.

  The name of a model configuration. A model configuration specifies a model trained by a well-known python implementation.
  There are many existing open source deep learning projects that define "standard" inputs and outputs, 
  and inference logic. ArcGIS built-in Python raster functions support a set of predefined well-known 
  configurations, and we are continuing to expand the list of supported model configurations. 

  The current supported model configurations are listed below along
  with a link to their project pages:

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

- "ModelType"

  Type: String.

  The type of the model: "ImageClassification" for classifying pixels, and "ObjectDetection" for detecting objects.
 
- "InferenceFunction"

  Type: String.

  The path of a custom inference Python raster function. If not specified, the default built-in Python raster function
  will be used. 
  
- Templates

    --"ImageClassifierTemplate" If you fill in the image classifier template with your own python code.
    
    --"ObjectDetectorTemplate" If your fill in the object detector template with your own python code.

  Deep learning Python raster function templates are also provided to help you writing your custom deep learning 
  Python raster function.  
  
## EMD for custom Python raster functions
If you find that the sample model definition files and built-in Python raster function cannot describe your deep learning
model architecture/properties, or if you use a deep learning framework other than TensorFlow, Keras, CNTK, or PyTorch, you
can write your own deep learning Python raster function and reference the Python raster function in the EMD file next 
to the "InferenceFunction" parameter. The built-in Python raster functions are good references for formatting your custom
deep learning Python raster functions. 


