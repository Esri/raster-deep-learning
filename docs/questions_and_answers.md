# Frequently Asked Questions

1. What is python raster function? 

    Python raster function allows developers to implement custom image processing algorithms in ArcGIS 
    with Python. Since many deep learning developers and data scientists use Python to train, interpret and deploy 
    deep learning models, we build our deep learning platform on top of existing Python raster function framework to allow an 
    integrated deep learning workflow in ArcGIS. You can get more information on Python raster function from Esri's help
    documentation and the Esri Raster Function Github repository (Link: https://github.com/Esri/raster-functions). 

2.  Which model frameworks are supported for integration? 

    The following model frameworks have built-in Python raster function in ArcGIS Pro 2.3 and ArcGIS Enterprise 10.7:
    - TensorFlow
    - Keras
    - CNTK

    However, if you use a different deep learning framework, you can write your own deep learning Python raster function and
    point to this Python raster function in your .emd file next to the "InferenceFunction" parameter. For more information, 
    see [Writng deep learning Python raster function](writing_deep_learning_python_raster_functions.md).  

3. What license do I need? 

    Deep learning inference tools in the ArcGIS platform require the ArcGIS Image Analyst license. 

