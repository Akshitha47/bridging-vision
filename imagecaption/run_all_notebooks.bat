@echo off
cd /d "c:\Users\jrahu\Downloads\Bridging Vision and Language Generative AI-Based Image Captioning for Enhanced Accessibility and Understanding\imagecaption"
echo Running all notebooks...
echo.
echo 1. Running image_captioning.ipynb...
jupyter nbconvert --to notebook --execute image_captioning.ipynb --output image_captioning_executed.ipynb
echo.
echo 2. Running train_model.ipynb...
jupyter nbconvert --to notebook --execute train_model.ipynb --output train_model_executed.ipynb
echo.
echo 3. Running temp.ipynb...
jupyter nbconvert --to notebook --execute temp.ipynb --output temp_executed.ipynb
echo.
echo All notebooks executed!
pause
