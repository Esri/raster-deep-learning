# Writing deep learning python raster functions
If you find that our current model definition and built-in python raster functions can't describe your deep learning
model architecture/properties or if you use a different deep learning framework, you can write your own deep python 
raster function yourself and just point the "Inference Function" to your own python raster function in the model 
definition file.

The help documents for general python raster function in ArcGIS can be found at the 
[python raster function wiki page](https://github.com/Esri/raster-functions/wiki). 
The [anatomy of a python raster function](https://github.com/Esri/raster-functions/wiki/PythonRasterFunction#anatomy-of-a-python-raster-function)
is especially useful to know the components of a python raster function. You can get started from there to get familiar 
with python raster functions. In this page we will cover how a deep learning based python raster function generally 
looks like to help you write your own function more efficiently.

## `.initialize(self,**kwargs)`
This method is called at the very first beginning in your python raster function.
kwargs\['model'\] is the model definition file. Here is what is suggested to be done in this method.
1. Load information from you model definition file.
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
This is called second after initialize(). You can define your parameters in this method. The first two parameters are
mandatory to be input raster and input model definition file. You can copy this code snippet to your python raster 
function.

e.g.
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
*sclars* has all the parameter values that can be accessed by the parameter name. Remember the save the parameter 
values here if you want to use the parameter values in other methods. 

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
Write this method to return the fields JSON string of the output feature class. 

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
Write this method if you use the tool Detect Objects Using Deep Learning to tell the feature geometry type of 
the output. It would be usually polygon if the model is to draw bounding boxes around objects.

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
Write this method if you use Detect Objects Using Deep Learning. This method is supposed to return a dictionary in which
the "output_vectors" property is the string of the features in JSON format in the image space. Normally what should 
to be done in this method are:
1. obtain the input image from *pixelBlocks* and transform to the shape of the model's input.
2. run the deep learning model on the input image tile.
3. post process the model's output as necessary.
4. generate a feature json object, wrap it as a string in a dictionary and return the dictionary.

An e.g. workflow would be:
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
Write this method if you use Classify Pixels Using Deep Learning for semantic segmentation.
This method is supposed to return the classified raster wrapped in a dictionary. Normally what should be done in this
method are:
1. obtain the input image from *pixelBlocks* and transform to the shape of the model's input.
2. run the deep learning model on the input image tile.
3. post process the model's output as necessary.
4. generate a classified raster, wrap it in a dictionary and return the dictionary.

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
