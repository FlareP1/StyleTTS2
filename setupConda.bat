@echo off

set PYTHON_VER=3.10.9

REM Check if Python version meets the recommended version
python --version 2>nul | findstr /b /c:"Python %PYTHON_VER%" >nul
if errorlevel 1 (
    echo Warning: Python version %PYTHON_VER% is recommended.
)

IF NOT EXIST venv (
    python -m venv venv
) ELSE (
    echo venv folder already exists, skipping creation...
)
call .\venv\Scripts\activate.bat

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U
pip install -r requirements.txt

pip install phonemizer
REM this is linux only, 
REM sudo apt-get install espeak-ng

REM This is the fix for espeak-ng for Windwows
REM See https://github.com/bootphon/phonemizer/issues/44
REM Install espeak-ng from https://github.com/espeak-ng/espeak-ng/releases/tag/1.51
REM edit .\StyleTTS2\venv\Lib\site-packages\phonemizer\backend\espeak\espeak.py
REM add the following lines near the top of the file before class EspeakBackend(BaseEspeakBackend):
REM from phonemizer.backend.espeak.wrapper import EspeakWrapper
REM _ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
REM EspeakWrapper.set_library(_ESPEAK_LIBRARY)

REM Test espeak is working (not sure how I got version 1.52?)
REM  (venv) C:\Users\Chris\Documents\StyleTTS2>phonemize --version
REM phonemizer-3.2.1
REM available backends: espeak-ng-1.52, segments-2.2.1
REM uninstalled backends: espeak-mbrola, festival

pip install jupyter

REM Now you can go to the styleTTS2 dir in your venv command shell and type the following
REM python -m notebook


