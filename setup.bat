@echo off
setlocal

REM --- Install pyenv-win and Python 3.10.13 if needed ---
where pyenv >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing pyenv-win...
    git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%\.pyenv
    setx PATH "%USERPROFILE%\.pyenv\pyenv-win\bin;%USERPROFILE%\.pyenv\pyenv-win\shims;%PATH%"
    set PYENV=%USERPROFILE%\.pyenv\pyenv-win
)

call pyenv install 3.10.13
call pyenv local 3.10.13

REM --- Create and activate virtualenv ---
python -m venv .venv
call .venv\Scripts\activate.bat

echo [INFO] Upgrading pip...
pip install --upgrade pip setuptools wheel

echo [INFO] Installing Python packages...
pip install torch==2.5.1 numpy soundfile mecab-python3 unidic-lite
pip install "TTS[ja]" transformers huggingface_hub

echo [INFO] Downloading MMS-TTS Vietnamese model...
python download_tts_models.py

echo [DONE] Setup complete!
echo To activate the virtual environment later, run:
echo    call .venv\Scripts\activate.bat

endlocal
