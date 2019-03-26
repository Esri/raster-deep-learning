# PyTorch Swimming Pools Detection Example in ArcGIS Pro
Step 0. Download the test deep learning model and image [here](https://www.arcgis.com/home/item.html?id=16fa8bab78d24832b4a7c2ecac835019).
You can also use your own trained model and test image.

Step 1. Open "Detect Object Using Deep Learning" geoprocessing tool. 

Step 2. Fill in the parameters.

| Parameter | Value |
| --------- | ----- |
| Input Raster | "redlands_large_18_12_(2014).jpg" |
| Input Model Definition File | pytorch_fastai_ssd.emd |
| Arguments | padding:0, batch_size:1|
Step 3. Run the GP tool.
<img src='../../docs/img/pytorch_fastaiSSD_swimmingpooldetectionexample.jpg'>
