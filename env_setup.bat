SETLOCAL
SET SCRIPTS_DIR=C:\Program Files\ArcGIS\Pro\bin\Python\Scripts\
SET ENV_DIR=C:\Program Files\ArcGIS\Pro\bin\Python\envs\
SET ENV_NAME=DeepLearningEnvTest

cd "%SCRIPTS_DIR%"
CALL "%SCRIPTS_DIR%deactivate.bat"

SET TF_PACKAGE=tensorflow-gpu
SET CTNK_PACKAGE=cntk-gpu
SET PYTORCH=https://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-win_amd64.whl

ECHO INFO: Clone a new ArcGIS python environment %ENV_NAME% ...
CALL "%SCRIPTS_DIR%conda.exe" create --name %ENV_NAME% --clone arcgispro-py3
CALL "%SCRIPTS_DIR%activate.bat" %ENV_NAME%

ECHO INFO: Install tensorflow-gpu, keras, and scikit-image ...
CALL "%SCRIPTS_DIR%conda.exe" install -c anaconda %TF_PACKAGE% keras scikit-image --yes

ECHO INFO: Install cntk-gpu, pytorch, torchvision, and Pillow ...
CALL "%ENV_DIR%%ENV_NAME%\Scripts\pip.exe" install %CTNK_PACKAGE% %PYTORCH% torchvision Pillow

ECHO INFO: Upgrading h5py library to latest version ...
CALL "%SCRIPTS_DIR%conda.exe" install -c scw h5py --yes

ECHO INFO: Run proswap to switch to the new env ...
CALL "%SCRIPTS_DIR%proswap.bat" %ENV_NAME%

EXIT