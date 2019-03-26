# Azure Pixel-level Land Classification Example using CNTK in ArcGIS Pro
Step 0. Download the test deep learning model and image [here](https://www.arcgis.com/home/item.html?id=e8bc272d1ce2456fa4b87c9af749a57f).
You can also use your own trained model and test image.

Step 1. Open "Classify Pixels Using Deep Learning" geoprocessing tool.

Step 2. Fill in the parameters.

| Parameter | Value |
| --------- | ----- |
| Input Raster | "images\m_4712101_nw_10_1_20150928.tif" |
| Input Model Definition File | azure_pixel_level_land_classification.emd |

Step 3. Run the tool.

<img src='../../../../docs/img/cntk_azurepixellevellandclassification_example.jpg'>
