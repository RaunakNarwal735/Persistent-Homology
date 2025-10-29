@echo off
:: using this batch file you auto commit and push changes to the repo, if any changes are detected
cd /d "C:\Users\rishu narwal\Desktop\MPI-CBG"
:: Checking for changes in the script, we are letting git handle venv via .gitignore
for /f %%i in ('git status --porcelain') do set changes=true

if not defined changes (
    echo No changes detected. Exiting.
    pause
    exit /b
)

:: 
git add .

::
set datetime=%date% %time%
git commit -m "commit on %datetime%"
git push origin main

echo loose? I dont loose, I win , that's my job that's what i do
pause