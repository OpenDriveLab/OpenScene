# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation



**a. Create a conda virtual environment and activate it.**
```shell
conda create -n nuplan python=3.8 -y
conda activate nuplan
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
pip install mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**d. Install mmdet, mmseg and mmdet3d.**
```shell
pip install mmdet==2.26.0
pip install mmsegmentation==0.29.1
pip install mmdet3d==1.0.0rc6
```

**e. Install nuplan.**
```shell
pip install nuplan-devkit

# Optional, you can install it locally
# git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
# pip install -e .
```

**f. Install other dependencies.**
```shell
pip install similaritymeasures==0.6.0
pip install numpy==1.22.4
pip install scipy==1.8.0
pip install ortools==9.2.9972
pip install setuptools==59.5.0
```


**g. Clone DriveEngine.**
```
git clone https://github.com/OpenDriveLab/DriveEngine.git
```

**h. Prepare pretrained models.**
```shell
cd DriveEngine
mkdir ckpts

cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

