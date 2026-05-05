@echo off
echo ==========================================
echo Setting up EPR_master Python Environment...
echo ==========================================

:: 1. Check if EPR_master environment exists, if not create it
if not exist "EPR_master_env" (
    echo Creating virtual environment...
    python -m venv EPR_master_env
) else (
    echo EPR_master_env virtual environment already exists.
)

:: 2. Upgrade pip to ensure smooth installation
echo Upgrading pip...
.\EPR_master_env\Scripts\python -m pip install --upgrade pip

:: 3. Install dependencies from requirements.txt
echo Installing dependencies...
.\EPR_master_env\Scripts\pip install -r requirements.txt

:: 4. Install EPR_master in editable mode
echo Installing EPR_master in editable mode...
.\EPR_master_env\Scripts\pip install -e .

echo ==========================================
echo Setup Complete!
echo To start working, run: EPR_master_env\Scripts\activate
echo ==========================================
pause