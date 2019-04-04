# Writing deep learning Python raster function
If you find that the sample model definition files and built-in Python raster functions cannot describe your deep learning
model architecture/properties, or if you use a deep learning framework other than TensorFlow, Keras, CNTK, or PyTorch, 
you can write your own deep learning Python raster function and reference the Python raster function in the .emd file next
to the "InferenceFunction" parameter.

Help documentation for working with Python raster functions in ArcGIS can be found on the 
[Python raster function wiki page](https://github.com/Esri/raster-functions/wiki). 
The [anatomy of a Python raster function](https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#anatomy-of-a-python-raster-function)
is an especially useful document if you want to get familiar with the components of a Python raster function.
In this page, we will show you what a deep learning Python raster function 
looks like and how to write your own functions more efficiently.

## `.initialize(self,**kwargs)`
This method is called at the beginning of your Python raster function.
kwargs\['model'\] is the model definition file. This method requires the following steps:
1. Load the model information your Esri model definition file (EMD).
2. Load your deep learning model and keep a handle to the model.

e.g.
```python
def initialize(self, **kwargs):
    if 'model' not in kwargs:
        return

    json_file = kwargs['model']
    with open(json_file, 'r') as f:
        self.json_info = json.load(f)
    
    # access the model path in the model definition file
    model_path = json_info['ModelFile']
    # load your model and keep and instance for the model
    # self.model = load_your_model(model_path)
```

## `.getParameterInfo(self)`
This is called after initialize(), and it is where you define your parameters. The first two parameters are
mandatory as they define the input raster and the input EMD. You can copy the following code snippet directly to your 
Python raster function:

```python
        {
            'name': 'raster',
            'dataType': 'raster',
            'required': True,
            'displayName': 'Raster',
            'description': 'Input Raster'
        },
        {
            'name': 'model',
            'dataType': 'string',
            'required': True,
            'displayName': 'Input Model Description (EMD) File',
            'description': 'Input model description (EMD) JSON file'
        }
```

## `.getConfiguration(self, **scalars)`
This method is used to set the input bands, padding and tile size. 
The *sclars* value contains all the parameter values that can be accessed by the parameter name. 
Remember to save the parameter values here if you want to use the parameter values in other methods. 

e.g.
```python
def getConfiguration(self, **scalars):
    if 'score_threshold' in scalars:
      self.score_threshold = float(scalars['score_threshold'])
    if 'padding' in scalars:
      self.padding = int(scalars['padding'])

    return {
        'extractBands': tuple(self.json_info['ExtractBands']),
        'padding': self.padding,
        'tx': self.json_info['ImageWidth'] - 2*self.padding,
        'ty': self.json_info['ImageHeight'] - 2*self.padding
    }
```

## `.getFields(self)`
Use this method to return the JSON string fields of the output feature class. 

e.g.
```python
def getFields(self):
    fields = {
        'fields': [
            {
                'name': 'OID',
                'type': 'esriFieldTypeOID',
                'alias': 'OID'
            },
            {
                'name': 'Class',
                'type': 'esriFieldTypeString',
                'alias': 'Class'
            },
            {
                'name': 'Confidence',
                'type': 'esriFieldTypeDouble',
                'alias': 'Confidence'
            },
            {
                'name': 'Shape',
                'type': 'esriFieldTypeGeometry',
                'alias': 'Shape'
            }
        ]
    }
    return json.dumps(fields)
```

## `.getGeometryType(self)`
Use this method if you use the Detect Objects Using Deep Learning tool and you want to declare the feature geometry type of 
the output detected objects. Typically, the output is a polygon feature class if the model is to draw bounding boxes around objects.

e.g.
```python
class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4
   
def getGeometryType(self):
    return GeometryType.Polygon
```

## `.vectorize(self, **pixelBlocks)`
Use this method if you use the Detect Objects Using Deep Learning tool. This method returns a dictionary in which
the "output_vectors" property is a string of features in image space in JSON format. A typical workflow is below:
1. Obtain the input image from *pixelBlocks* and transform to the shape of the model's input.
2. Run the deep learning model on the input image tile.
3. Post-process the model's output as necessary.
4. Generate a feature JSON object, wrap it as a string in a dictionary and return the dictionary.

e.g.
```python
def vectorize(self, **pixelBlocks):
    # obtain the input image
    input_image = pixelBlocks['raster_pixels']
    
    # Todo: transform the input image to the shape of the model's input
    # input_image = np.transform(input_image, ....)
    
    # Todo: run the model on the transformed input image, something like 
    # model_output = self.model.run(input_image)
    
    # Todo: create feature json object and fill out the geometry
    # features geometry and properties are filled by model_output
    
    # Todo: wrap the json object as a string in dictionary
    # return {'output_vectors': json.dumps(features)}
```


## `.updatePixels(self, tlc, shape, props, **pixelBlocks)`
Use this method if you use the Classify Pixels Using Deep Learning tool for semantic segmentation.
This method returns the classified raster wrapped in a dictionary. The typical workflow is below:

1. Obtain the input image from *pixelBlocks* and transform to the shape of the model's input.
2. Run the deep learning model on the input image tile.
3. Post-process the model's output as necessary.
4. Generate a classified raster, wrap it in a dictionary and return the dictionary.

e.g.
```python
def updatePixels(self, tlc, shape, props, **pixelBlocks):
    # obtain the input image
    input_image = pixelBlocks['raster_pixels']
    
    # Todo: transform the input image to the shape of the model's input
    # input_image = np.transform(input_image, ....)
    
    # Todo: run the model on the transformed input image, something like 
    # model_output = self.model.run(input_image)
    
    # Todo: wrap the classified raster in dictonary, something like
    # pixelBlocks['output_pixels'] =  model_output.astype(props['pixelType'], copy=False)
    
    # Return the dict
    # return pixelBlocks

    return pixelBlocks
```
