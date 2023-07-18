# Step-by-step installation instructions

> The installation is referred from [MMDetection3D](https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation).



**a. Create a conda virtual environment and activate it.**
```shell
conda create -n openscene python=3.8 -y
conda activate openscene
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9

```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
#  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

**f. Install nuplan-devkit.**
```shell
pip install nuplan-devkit

# Optional, you can install it locally
# git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
# pip install -r requirements.txt
# pip install -e .
# Please modify the python version manually
```

**g. Install other dependencies.**
```shell
pip install similaritymeasures==0.6.0
pip install numpy==1.22.4
pip install scipy==1.8.0
pip install ortools==9.2.9972
pip install setuptools==59.5.0
pip install networkx==2.5
```


**h. Clone OpenScene.**
```
git clone https://github.com/OpenDriveLab/OpenScene.git
```

**i. Prepare pre-trained models.**
```shell
cd OpenScene
cd DriveEngine
mkdir ckpts

cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

