# Getting Started

The OpenScene dataset is a large-scale dataset for visual pretraining in the field of autonomous driving.
Based on [nuPlan](https://www.nuscenes.org/nuplan), we provide occupancy detection and flow annotations in 3D space.
**Download now to discover our dataset!**

## OpenScene v1.1

### :fire: Change Log
- We reorganized the meta data files and divided it by nuPlan log file to improve accessibility.
- We supplemented more logs that have sensor data.
- We further uploaded the LiDAR raw sensor data.

### :exclamation: Must Read for the CVPR 2024 Challenge

- For Track **End-to-End Driving at Scale**, please download the `meta_data` and the `camera` or/and `LiDAR` sensor data, depend on modalities you intend to use.
- For Track **Predictive World Model**, please download the `meta_data`, the `camera` and `LiDAR` sensor data.

- The `private test` set utilized in this challenge is exclusively provided by Motional and should not be confused with the `test` set.
- It's important to note that the private test sets for the two tracks are distinct and do not share any overlapped data.
- The input data (metadata, sensors) for the private test set will be accessible upon the test server open. The ground truth data will be only available on the test server operated by Motional.


### Download Data
<!-- We recommended to download from [**OpenDriveLab**](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene)<img src="https://github.com/OpenDriveLab/OpenLane-V2/assets/29263416/4cfa0f7f-535c-40fa-9fca-81276683931d" alt="OpenDriveLab" width="18"/> and use provided **command line interface (CLI)** for acceleration.  -->

- ~~We provide download link from Amazon **AWS S3**. You can use `wget` to download files. Currently, our download server is located in Asia. We plan to expand our server locations to include the US and Europe soon.~~
- :grey_exclamation: We meet some technical issues and we have to disable the AWS S3 download link. We are really sorry for the unconvenience caused.
- Please first download the `image` data from [OpenScene v1.0](#openscene-v10). Or directly download all the sensor data from [nuPlan](https://www.nuscenes.org/nuplan).

- The sensor data for both the trainval and test subsets amount to approximately 2TB. We recommend initially training and validating your model on the mini set.

- :bell: For those who already possess the [nuPlan](https://www.nuscenes.org/nuplan) sensor data (over 20TB) locally, you have the option to directly link it to the OpenScene folder to avoid redundant downloads. We carefully make the folder structure aligned with nuPlan and just downsample the nuPlan sensor data to improve the accessibility.

- If you already have the OpenScene v1.0 `image` data, you can use it for OpenScene v1.1 as well, since almost (>98%) all the data are present. If you want to use the occupancy label, please also download it from OpenScene v1.0. There are only a few instances of additional data in v1.1 that are missing. You can temporarily ignore those frames during training.


#### mini set

| File Name  | Google Drive | Size |
| :--------: | :----------: | :--: |
| openscene_metadata_mini.tgz | [link](https://drive.google.com/file/d/1vGaTaXUQWEo9oZgJe_pUmKXNeCVAT8ME/view?usp=drive_link)| 509.6 MB |

#### trainval set

| File Name  | Google Drive | Size |
| :--------: | :----------: | :--: |
| openscene_metadata_trainval.tgz | [link](https://drive.google.com/file/d/1ce3LLQDpST-QzpV1ZVZcaMnjVkZnHXUq/view?usp=drive_link) | 6.6 GB |

#### test set

| File Name  | Google Drive | Size |
| :--------: | :----------: | :--: |
| openscene_metadata_test.tgz | [link](https://drive.google.com/file/d/1hTQ56OqaNgljE3zD5qtte91uNE9qgSMk/view?usp=drive_link) | 31.3 MB |

<!-- 
#### mini set

| File Name  | Amazon AWS S3 | Size |
| :---: |  :---: | :---: |
| openscene_metadata_mini.tgz | [Asia](https://opendrivelab-openscene.s3.ap-southeast-1.amazonaws.com/openscene-v1.1/openscene_metadata_mini.tgz)| 509.6 MB |
| openscene_sensor_mini_camera.tgz | [Asia](https://opendrivelab-openscene.s3.ap-southeast-1.amazonaws.com/openscene-v1.1/openscene_sensor_mini_camera.tgz) | 83.9 GB |
| openscene_sensor_mini_lidar.tgz | [Asia](https://opendrivelab-openscene.s3.ap-southeast-1.amazonaws.com/openscene-v1.1/openscene_sensor_mini_lidar.tgz) | 59.1 GB |

#### trainval set

| File Name  | Amazon AWS S3 | Size |
| :---: |  :---: | :---: |
| openscene_metadata_trainval.tgz | [Asia](https://opendrivelab-openscene.s3.ap-southeast-1.amazonaws.com/openscene-v1.1/openscene_metadata_trainval.tgz) | 6.6 GB |
| openscene_sensor_trainval_camera_{0-24}.tgz | [Asia](download_links/openscene_sensor_trainval_camera.txt) | 1.1 TB |
| openscene_sensor_trainval_lidar_{0-24}.tgz | [Asia](download_links/openscene_sensor_trainval_lidar.txt) | 821.6 GB |

#### test set

| File Name  | Amazon AWS S3 | Size |
| :---: |  :---: | :---: |
| openscene_metadata_test.tgz | [Asia](https://opendrivelab-openscene.s3.ap-southeast-1.amazonaws.com/openscene-v1.1/openscene_metadata_test.tgz) | 31.3 MB |
| openscene_sensor_test_camera_{0-2}.tgz | [Asia](download_links/openscene_sensor_test_camera.txt) | 119.1 GB |
| openscene_sensor_test_lidar_{0-2}.tgz | [Asia](download_links/openscene_sensor_test_lidar.txt) | 86.3 GB |
-->
#### private test set

The input data (metadata, sensors) for the private test set will be accessible upon the test server open.

### Prepare Dataset

Please follow the steps below to get familiar with the OpenScene v1.1 dataset.

1. Download all the data manually and unzip them.
2. Make sure the filesystem hierarchy is the same as the [dataset stats](dataset_stats.md#filesystem-hierarchy).
3. Modify and run `python DriveEngine/process_data/collect_data.py` to collect the meta_data in any custom split.


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

## Train a Occupancy Prediction Model

### Baseline

We provide a baseline model based on [OccNet](https://github.com/OpenDriveLab/OccNet). The baseline is currently compatible with OpenScene dataset `v1.0`.
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

See [openscene_scenario_visualization.py](/DriveEngine/process_data/openscene_scenario_visualization.py)
