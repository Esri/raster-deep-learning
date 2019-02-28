# PyTorch Swimming Pools Detection Example in ArcGIS Pro
1. Open "Detect Object Using Deep Learning" geoprocessing tool. 

2. Fill in the parameters.

| Parameter | Value |
| --------- | ----- |
| Input Raster | "redlands_large_18_12_(2014).jpg" |
| Input Model Definition File | pytorch_fastai_ssd.emd |
| Arguments | padding:0, batch_size:1|
3. Run the GP tool.
<img src='../../docs/img/pytorch_fastaiSSD_swimmingpooldetectionexample.jpg'>
