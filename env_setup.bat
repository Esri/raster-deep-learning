SETLOCAL
FOR /F "usebackq tokens=2*" %%A IN (`reg query "HKLM\Software\Microsoft\Windows\CurrentVersion\Uninstall\ArcGISPro" /v "InstallLocation"`) DO (
  SET ARCGISPRO_INSTALL_DIR=%%B
)
SET SCRIPTS_DIR=%ARCGISPRO_INSTALL_DIR%bin\Python\Scripts\
SET ENV_DIR=%ARCGISPRO_INSTALL_DIR%bin\Python\envs\
SET ENV_NAME=DeepLearningEnvTest

cd "%SCRIPTS_DIR%"
CALL "%SCRIPTS_DIR%deactivate.bat"

SET TF_PACKAGE=tensorflow-gpu
SET CTNK_PACKAGE=cntk-gpu

ECHO INFO: Clone a new ArcGIS python environment %ENV_NAME% ...
CALL "%SCRIPTS_DIR%conda.exe" create --name %ENV_NAME% --clone arcgispro-py3
CALL "%SCRIPTS_DIR%activate.bat" %ENV_NAME%

ECHO INFO: Install tensorflow-gpu, keras, and scikit-image ...
CALL "%SCRIPTS_DIR%conda.exe" install -c anaconda %TF_PACKAGE%=1.14.0 keras=2.2.4 scikit-image=0.15.0 --yes

ECHO INFO: Install cntk-gpu and Pillow ...
CALL "%ENV_DIR%%ENV_NAME%\Scripts\pip.exe" install %CTNK_PACKAGE% Pillow==6.1.0

ECHO INFO: Install Env for ArcGIS API For Python ...
CALL "%SCRIPTS_DIR%conda.exe" install -c fastai -c pytorch fastai=1.0.54 pytorch=1.1.0 torchvision=0.3.0 --yes

ECHO INFO: Run proswap to switch to the new env ...
CALL "%SCRIPTS_DIR%proswap.bat" %ENV_NAME%

EXIT