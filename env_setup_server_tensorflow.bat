SETLOCAL
FOR /F "usebackq tokens=2*" %%A IN (`reg query "HKLM\Software\Microsoft\Windows\CurrentVersion\Uninstall\ArcGIS Server 10.7" /v "InstallLocation"`) DO (
  SET ARCGISSERVER_INSTALL_DIR=%%B
)
SET SCRIPTS_DIR=%ARCGISSERVER_INSTALL_DIR%framework\runtime\ArcGIS\bin\Python\Scripts\
SET ENV_DIR=%ARCGISSERVER_INSTALL_DIR%framework\runtime\ArcGIS\bin\Python\envs\
SET ENV_NAME=TensorFlowEnvTest

cd "%SCRIPTS_DIR%"
CALL "%SCRIPTS_DIR%deactivate.bat"

SET TF_PACKAGE=tensorflow-gpu

ECHO INFO: Clone a new ArcGIS python environment %ENV_NAME% ...
CALL "%SCRIPTS_DIR%conda.exe" create --name %ENV_NAME% --clone arcgispro-py3
CALL "%SCRIPTS_DIR%activate.bat" %ENV_NAME%

ECHO INFO: Install tensorflow-gpu ...
CALL "%SCRIPTS_DIR%conda.exe" install -c anaconda %TF_PACKAGE%=1.13.1 --yes

ECHO INFO: Run proswap to switch to the new env ...
CALL "%SCRIPTS_DIR%proswap.bat" %ENV_NAME%

EXIT