# Getting Started

The OpenScene dataset is a large-scale dataset for end-to-end planning, visual pretraining, and occupancy prediction in the field of autonomous driving.

Based on [nuPlan](https://www.nuscenes.org/nuplan), we provide bounding box, occupancy, and flow annotations in 3D space.

## OpenScene v1.1

### :fire: Change Log
- We reorganized the meta data files and organized them by their nuPlan log files to improve usability.
- We added more logs that have sensor data and uploaded the LiDAR raw sensor data.

### :exclamation: Must Read for the CVPR 2024 Challenge

- For Track **End-to-End Driving at Scale**, please download the `meta_data` and the `camera` or/and `LiDAR` sensor data, depend on modalities you intend to use. Note that there is no separate competition track for camera-only planners.
- For Track **Predictive World Model**, please download the `meta_data`, the `camera` and `LiDAR` sensor data.

- The `private test` set utilized in the challenge leaderboards is exclusively provided by Motional and should not be confused with the `test` set.
- It is important to note that the private test sets for the two tracks are distinct and do not share any data.
- The input data (metadata, sensors) for the private test set will be accessible upon the opening of test server. The ground truth data will be only available on the test server operated by Motional.


### Download Data

- We recommended to download all data from [**OpenDriveLab**](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1)<img src="https://github.com/OpenDriveLab/OpenLane-V2/assets/29263416/4cfa0f7f-535c-40fa-9fca-81276683931d" alt="OpenDriveLab" width="18"/> and use provided **command line interface (CLI)** for acceleration.

- The sensor data for both the trainval and test subsets amount to approximately 2TB. We recommend initially training and validating your models on the **`mini` set**.

- :bell: For those who already possess the [nuPlan](https://www.nuscenes.org/nuplan) sensor data (over 20TB) locally, you have the option to directly link it to the OpenScene folder to avoid redundant downloads. We carefully aligned the folder structure with nuPlan and just downsampled the nuPlan sensor data to improve the accessibility.

- :bell: If you already have the OpenScene v1.0 `image` data, you can use it for OpenScene v1.1 as well, since almost (>98%) of all the data is present. If you want to use the occupancy label, please also download it from OpenScene v1.0. There are only a few instances of additional data in v1.1 that are missing. You can temporarily ignore those frames during training.

- If you can't access [**OpenDriveLab**](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1)<img src="https://github.com/OpenDriveLab/OpenLane-V2/assets/29263416/4cfa0f7f-535c-40fa-9fca-81276683931d" alt="OpenDriveLab" width="18"/>, please try the HuggingFace download links below. Alternatively, you may first download the `image` data from [OpenScene v1.0](#openscene-v10). Or directly download all the sensor data from [nuPlan](https://www.nuscenes.org/nuplan).

#### mini set

| File Name  | Download Link | Size |
| :--------: | :----------: | :--: |
| openscene_metadata_mini.tgz  | [Google Drive](https://drive.google.com/drive/folders/1MnRwhnEBsgZxbaleHxc3Gw7Ovc4I9az1?usp=sharing) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_mini.tgz?download=true)  | 509.6 MB |
| openscene_sensor_mini_camera | [OpenDriveLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_mini_camera) | 84 GB |
| openscene_sensor_mini_lidar  | [OpenDriveLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_mini_lidar) | 60 GB |

#### trainval set

| File Name  | Download Link | Size |
| :--------: | :----------: | :--: |
| openscene_metadata_trainval.tgz  | [Google Drive](https://drive.google.com/drive/folders/1MnRwhnEBsgZxbaleHxc3Gw7Ovc4I9az1?usp=sharing) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz?download=true) | 6.6 GB |
| openscene_sensor_trainval_camera | [OpenDriveLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_trainval_camera) | 1.1 TB |
| openscene_sensor_trainval_lidar  | [OpenDriveLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_trainval_lidar) | 822 GB |

#### test set

| File Name  | Download Link | Size |
| :--------: | :----------: | :--: |
| openscene_metadata_test.tgz  | [Google Drive](https://drive.google.com/drive/folders/1MnRwhnEBsgZxbaleHxc3Gw7Ovc4I9az1?usp=sharing) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz?download=true) | 454 MB |
| openscene_sensor_test_camera | [OpenDriveLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_test_camera) | 120 GB |
| openscene_sensor_test_lidar  | [OpenDriveLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_test_lidar) | 87 GB |

#### private test set

| File Name  | Download Link | Size |
| :--------: | :----------: | :--: |
| openscene_metadata_private_test_wm.tgz | [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_private_test_wm.tgz?download=true) | 7.3 MB |
| openscene_sensor_private_test_wm.tgz | [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_private_test_wm.tgz?download=true) | 15 GB |
| openscene_metadata_private_test_e2e.tgz | [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_private_test_e2e.tgz?download=true) | 4 MB |
| openscene_sensor_private_test_e2e.tgz | [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_private_test_e2e.tgz?download=true) | 23.6 GB |

- `private_test_wm` is the private test set for `Predictive World Model` track.
- `private_test_e2e` is the private test set for `End-to-End Driving at Scale` track.
- **[2024-04-09]** We fix some bugs and update the metadata of `private_test_wm`, please replace it!

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
