@echo off
REM FoundPose runner script - sets environment variables and PYTHONPATH
set REPO_PATH=D:\LDH_ws\foundpose-main
set BOP_PATH=D:\LDH_ws\gigapose_custom\gigaPose_datasets\datasets
set BOP_OUTPUT_PATH=D:\LDH_ws\foundpose-main\outputs
set PYTHONPATH=%REPO_PATH%;%REPO_PATH%\external\bop_toolkit;%REPO_PATH%\external\dinov2

C:\Users\PC\miniconda3\envs\foundpose\python.exe %*
