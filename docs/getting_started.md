# Getting Started

The OpenScene dataset is a large-scale dataset for visual pretraining in the field of autonomous driving.
Based on [nuPlan](https://www.nuscenes.org/nuplan), we provide occupancy detection and flow annotations in 3D space.
**Download now to discover our dataset!**

## OpenScene v1.1

### :fire: Change Log
- We reorganized the meta data files and divided it by nuPlan log file to improve accessibility.
- We supplemented more logs that have sensor data.
- We further uploaded the LiDAR raw sensor data.

### Download Data
We recommended to download from [**OpenDriveLab**](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene)<img src="https://github.com/OpenDriveLab/OpenLane-V2/assets/29263416/4cfa0f7f-535c-40fa-9fca-81276683931d" alt="OpenDriveLab" width="18"/> and use provided **command line interface (CLI)** for acceleration. 

Additionally, we provide download link from Amazon AWS S3.

#### Subset mini

| File Name  | Amazon AWS S3 | Approx. Size |
| :---: |  :---: | :---: |
| openscene_metadata_mini.tgz | [Asia](https://opendrivelab-openscene.s3.ap-southeast-1.amazonaws.com/openscene-v1.1/openscene_metadata_mini.tgz) | 509.6MB |
| openscene_sensor_mini_camera.tgz | [Asia](https://opendrivelab-openscene.s3.ap-southeast-1.amazonaws.com/openscene-v1.1/openscene_sensor_mini_camera.tgz) | 83.9GB |
| openscene_sensor_mini_lidar.tgz | [Asia](https://opendrivelab-openscene.s3.ap-southeast-1.amazonaws.com/openscene-v1.1/openscene_sensor_mini_lidar.tgz) | 59.1GB |

#### Subset trainval

| File Name  | Amazon AWS S3 | Approx. Size |
| :---: |  :---: | :---: |
| openscene_metadata_trainval.tgz | [Asia](https://opendrivelab-openscene.s3.ap-southeast-1.amazonaws.com/openscene-v1.1/openscene_metadata_trainval.tgz) | 6.6GB |

The rest files are uploading.

#### Subset test

| File Name  | Amazon AWS S3 | Approx. Size |
| :---: |  :---: | :---: |
| openscene_metadata_test.tgz | [Asia](https://opendrivelab-openscene.s3.ap-southeast-1.amazonaws.com/openscene-v1.1/openscene_metadata_test.tgz) | 31.3MB |
The rest files are uploading.

## OpenScene v1.0

### Download Data

We recommended to download from [**OpenDriveLab**](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene)<img src="https://github.com/OpenDriveLab/OpenLane-V2/assets/29263416/4cfa0f7f-535c-40fa-9fca-81276683931d" alt="OpenDriveLab" width="18"/> and use provided **command line interface (CLI)** for acceleration. In addition, Google Drive<img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" alt="Google Drive" width="18"/> and Baidu Cloud<img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="Baidu Yun" width="18"/> are also available. If you already have the nuPlan dataset, you only need to download the `label` and `meta data`.

| Subset  | Google Drive<img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" alt="Google Drive" width="18"/> | Baidu Cloud<img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="Baidu Yun" width="18"/> | Approx. Size |
| :---: |  :---: | :---: | :---: |
| mini |  [image](https://drive.google.com/drive/folders/1sWCpfQiAjOf2e9D3O3R2gJz5e9MXNArI?usp=drive_link) / [label](https://drive.google.com/drive/folders/16xjIgfaIiUq34aU3Qev9pVCDCk90HoEu?usp=drive_link) |  [image](https://pan.baidu.com/s/15nF043xirjZnrOm9qgLh8w?pwd=hksj) / [label](https://pan.baidu.com/s/1HKeDR-qRKpxOAhjesSMjyQ?pwd=ek5j) |  81.2G / 6.7G |
| trainval  |  [image](https://drive.google.com/drive/folders/1kwPMvECZbyWx1AsVLNLP9sLNYQIvScki?usp=drive_link) / [label](https://drive.google.com/drive/folders/1rtMG5gfqL7T7aV06awEa0oyjgGd5q2bI?usp=drive_link) |  [image](https://pan.baidu.com/s/1ZW5oV4JmKFwtO9ciTC-sBA?pwd=qx9x) / [label](https://pan.baidu.com/s/1_WW6spKo_Ru_0ge9SCuOQg?pwd=j6qn) |  1.1T / 95.4G |
| test |  [image](https://drive.google.com/drive/folders/1VUapdlwKRRVl7rh6XLUC9ekLja-EHC4R?usp=drive_link)   |  [image](https://pan.baidu.com/s/1kUKzYszyoRyZj-2m4uCiTw?pwd=8hxs) | 118.5G |
| meta data | [meta file](https://drive.google.com/drive/folders/1kf_qkvXQ2gT4o8JBd5fQL4_2CtApnbOF?usp=drive_link) |  [meta file](https://pan.baidu.com/s/1MxtbNvzZO_NsuYvPGPwwAA?pwd=kwbz) | 6.4G |

> - Mini and trainval data contain three parts -- `sensor_blobs (images)`, `meta_data`, and `occupancy (label)`.

To ensure the integrity of the downloaded data, we recommend verifying the file using its [MD5 checksum](https://drive.google.com/file/d/1B9E43icOc16AhzHU33_RBsgVIfhoNgRI/view?usp=drive_link) after the download is complete.

### Prepare Dataset

Please follow the steps below to get familiar with the OpenScene dataset.

1. Following [step-by-step instruction](install.md) to install the environment for setting up the dataset:

2. Download data manually from OpenDriveLab<img src="https://github.com/OpenDriveLab/OpenLane-V2/assets/29263416/4cfa0f7f-535c-40fa-9fca-81276683931d" alt="OpenDriveLab" width="18"/>, Google Drive<img src="https://user-images.githubusercontent.com/29263416/236970575-125919cc-1a36-4968-95e7-f5a17f896f9f.png" alt="Google Drive" width="18"/>, or Baidu Yun<img src="https://user-images.githubusercontent.com/29263416/236970717-fe619dd6-7e36-446b-88bf-30ff91028d87.png" alt="Baidu Yun" width="18"/>. Then link them into the `data/OpenScene/` folder and unzip them. 
3. Make sure the filesystem hierarchy is the same as the [dataset stats](dataset_stats.md#filesystem-hierarchy).

4. Process the data:
```sh
cd DriveEngine
python ./process_data/prepare_data.py
# Please be careful of the path recorded in the .pkl file, edit it manually if needed.
```


## Train a Model

### Baseline

We provide a baseline model based on [OccNet](https://github.com/OpenDriveLab/OccNet).
<!---Please refer to [DriveEngine](https://github.com/OpenDriveLab/DriveEngine/) (TBA) for details.--->



### Train and Test

**Train model with 4 GTX3090 GPUs** 
```
./tools/dist_train.sh ./projects/configs/bevformer/bev_tiny_occ_r50_nuplan.py 4
```

**Eval model with 4 GTX3090 GPUs**
```
./tools/dist_test.sh ./projects/configs/bevformer/bev_tiny_occ_r50_nuplan.py ./path/to/ckpts.pth 4
```
<!---Note: using 1 GPU to eval can obtain slightly higher performance because continuous video may be truncated with multiple GPUs. By default, we report the score evaluated with 8 GPUs.--->



### Visualization 

See [openscene_scenario_visualization.py](/DriveEngine/process_data//openscene_scenario_visualization.py)
