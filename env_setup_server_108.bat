SETLOCAL
FOR /F "usebackq tokens=2*" %%A IN (`reg query "HKLM\Software\Microsoft\Windows\CurrentVersion\Uninstall\ArcGIS Server 10.8" /v "InstallLocation"`) DO (
  SET ARCGISSERVER_INSTALL_DIR=%%B
)
SET SCRIPTS_DIR=%ARCGISSERVER_INSTALL_DIR%framework\runtime\ArcGIS\bin\Python\Scripts\
SET ENV_DIR=%ARCGISSERVER_INSTALL_DIR%framework\runtime\ArcGIS\bin\Python\envs\
SET ENV_NAME=ServerDLEnvTest

cd "%SCRIPTS_DIR%"
CALL "%SCRIPTS_DIR%deactivate.bat"

SET TF_PACKAGE=tensorflow-gpu
SET KERAS_PACKAGE=keras-gpu

ECHO INFO: Clone a new ArcGIS python environment %ENV_NAME% ...
CALL "%SCRIPTS_DIR%conda.exe" create --name %ENV_NAME% --clone arcgispro-py3
CALL "%SCRIPTS_DIR%activate.bat" %ENV_NAME%

ECHO INFO: Install tensorflow-gpu, keras, and scikit-image ...
CALL "%SCRIPTS_DIR%conda.exe" install -c anaconda %TF_PACKAGE%=1.14.0 %KERAS_PACKAGE%=2.2.4 scikit-image=0.15.0 --yes
CALL "%SCRIPTS_DIR%conda.exe" install -c anaconda Pillow=6.1.0 --yes

ECHO INFO: Install Env for ArcGIS API For Python ...
CALL "%SCRIPTS_DIR%conda.exe" install fastai=1.0.54 --yes
CALL "%SCRIPTS_DIR%conda.exe" install pytorch=1.1.0 --yes
CALL "%SCRIPTS_DIR%conda.exe" install libtiff=4.0.10 --no-deps --yes

ECHO INFO: Run proswap to switch to the new env ...
CALL "%SCRIPTS_DIR%proswap.bat" %ENV_NAME%

EXIT