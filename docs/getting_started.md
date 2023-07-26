# Prerequisites

**Please ensure you have prepared the environment and the Openscene dataset.**

## Download Data


### Download-ToDo
The files mentioned below can also be downloaded via <img src="https://user-images.githubusercontent.com/29263416/222076048-21501bac-71df-40fa-8671-2b5f8013d2cd.png" alt="OpenDataLab" width="18"/>[OpenDataLab](https://opendatalab.com/CVPR2023-3D-Occupancy/download).It is recommended to use provided [command line interface](https://opendatalab.com/CVPR2023-3D-Occupancy/cli) for acceleration.

| Subset | Google Drive <img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" alt="Google Drive" width="18"/> | Baidu Cloud <img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="Baidu Yun" width="18"/> | Size |
| :---: | :---: | :---: | :---: |
| mini | [data](https://drive.google.com/drive/folders/1ksWt4WLEqOxptpWH2ZN-t1pjugBhg3ME?usp=share_link) | [data](https://pan.baidu.com/s/1IvOoJONwzKBi32Ikjf8bSA?pwd=5uv6)  | approx. 440M |
| trainval  | [data](https://drive.google.com/drive/folders/1JObO75iTA2Ge5fa8D3BWC8R7yIG8VhrP?usp=share_link) | [data](https://pan.baidu.com/s/1_4yE0__UDIJS8JtBSB0Bpg?pwd=li5h) | approx. 32G |
| test | [data](https://drive.google.com/drive/folders/1hVs2AzSlEePN7QR502d8q7FoAbdJLxx8?usp=share_link) | [data](https://pan.baidu.com/s/1ElTu7i5gjXz3TwE2L0YBQQ?pwd=jstt) | approx. 6G |

* Mini and trainval data contain three parts -- `imgs`, `gts` and `annotations`. The `imgs` datas have the same hierarchy with the image samples in the original nuScenes dataset.


## Install Devkit

### Development Kit-ToDo

We provide a baseline model based on [OccNet](https://github.com/OpenDriveLab/OccNet).

Please refer to [DriveEngine](https://github.com/OpenDriveLab/DriveEngine/tree/main) for details.


## Prepare Dataset



## Train and Test

Train model with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/bevformer/bev_tiny_occ_r50_nuplan.py 8
```

Eval model with 8 GPUs
```
./tools/dist_test.sh ./projects/configs/bevformer/bev_tiny_occ_r50_nuplan.py ./path/to/ckpts.pth 8
```
Note: using 1 GPU to eval can obtain slightly higher performance because continuous video may be truncated with multiple GPUs. By default we report the score evaled with 8 GPUs.


## Visualization 

see [visual.py](../tools/analysis_tools/visual.py)
