# Instructions
The ONNX model was converted from the [XView](http://xviewdataset.org/) TensorFlow model, multires.pb. For more tools to work with the XView dataset, please see the [XView Github page](https://github.com/DIUx-xView).

# XView Model Conversion Using ```tf2onnx```
On Windows, the ONNX model was generated with

```
tensorflow=1.13.1
onnx=1.6.0
tf2onnx=1.5.3
```

and

```
tensorflow=1.14.0
onnx=1.6.0
tf2onnx=1.5.3
```

Some errors are seen when converting to ONNX that will effect it's accuracy.

Conversion to ONNX can be done through the command line using the following statement:

```
python -m tf2onnx.convert --input C:\XVIEW\multires.pb --inputs "image_tensor:0" --outputs "detection_boxes:0,detection_scores:0,detection_classes:0" --output C:\XVIEW\ONNX\saved_model.onnx --opset 10
```

# Running the Model

Step 0. Download the test deep learning model and the XView training or validation images.

Step 1. Open "Detect Object Using Deep Learning" geoprocessing tool.

Step 2. Fill in the parameters.

![](https://github.com/gbrunner/raster-deep-learning/blob/master/docs/img/onnx_objectdetectiontool.png)

Step 3. Run the Model.

![](https://github.com/gbrunner/raster-deep-learning/blob/master/docs/img/onnx_objectdetection.png)
