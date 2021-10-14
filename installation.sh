# upstage level2 p-stage object detection 서버 사용시 기존 환경을 클론한 환경을 만든다.
conda create --name detection_custom --clone detection
conda activate detection_custom

# Swin Transformer for Object Detection 설치
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git
cd Swin-Transformer-Object-Detection
pip install -r requirements/build.txt
pip install -v -e .
cd ..
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ..
