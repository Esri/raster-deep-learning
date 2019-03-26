# Keras Mask RCNN House Footprints Example in ArcGIS Pro
Step 0. Download the test deep learning model and image [here](https://www.arcgis.com/home/item.html?id=646dae44d4334d5ba68c7541031d9edc).
You can also use your own trained model and test image.

Step 1. Open "Detect Object Using Deep Learning"

Step 2. Fill in the parameters

| Parameter | Value |
| --------- | ----- |
| Input Raster | "images\15OCT22183656-S2AS_R1C1-056155973040_01_P001.TIF" |
| Input Model Definition File | mask_rcnn_spacenet.emd |
| Arguments | padding:0 |

Step 3. Run the tool.
<img src='../../../docs/img/keras_maskrcnn_housefootprintsexample.jpg'>
