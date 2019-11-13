# Instructions
This examples uses the model fron the CNTK model from the [Deep Learning - CNTK for Land Cover Classification](https://www.arcgis.com/home/item.html?id=e8bc272d1ce2456fa4b87c9af749a57f) example.

The CNTK model file was converted to ONNX using the CNTK to ONNX Export](https://github.com/onnx/tutorials/blob/master/tutorials/CntkOnnxExport.ipynb) instructions. 

``` 
import cntk as C
model_path = "trained.model"
z = C.Function.load(model_path, device=C.device.cpu())
z.save("trained.onnx", format=C.ModelFormat.ONNX)
```

## Running the Model

Step 0. Download the test deep learning model and image [here](https://www.arcgis.com/home/item.html?id=e8bc272d1ce2456fa4b87c9af749a57f). You can also use your own trained model and test image.

Step 1. Open "Classify Pixels Using Deep Learning" geoprocessing tool.

Step 2. Fill in the parameters.

![](https://github.com/gbrunner/raster-deep-learning/blob/master/docs/img/onnx_landclassificationexampletool.png)

Step 3. Run the tool.

![](https://github.com/gbrunner/raster-deep-learning/blob/master/docs/img/onnx_landclassificationexample.png)
