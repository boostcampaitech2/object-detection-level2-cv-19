conda create --name detection_custom --clone detection
conda activate detection_custom

git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git
cd Swin-Transformer-Object-Detection
pip install -r requirements/build.txt
pip install -v -e .
cd ..

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ..
