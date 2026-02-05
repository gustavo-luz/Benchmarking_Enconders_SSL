#!/bin/bash
pip install --upgrade pip
pip install --upgrade setuptools
echo "🎈 Installing Minerva-Dev"
(cd Minerva-Dev && pip install -e .)


pip install pandasql==0.7.3
pip install graphviz==0.21
pip install ray[default]==2.46.0
pip install git+https://github.com/KaiyangZhou/Dassl.pytorch
pip install sqlitedict==2.1.0
pip3 install opencv-python-headless==4.13.0.90
# pip install jsonargparse==4.37.0
pip install --upgrade numba
pip uninstall numpy -y
pip install numpy==1.26.4
pip uninstall kaleido -y
pip install kaleido==0.2.1
pip install thop

pip uninstall pandas -y
pip install pandas==2.3.3