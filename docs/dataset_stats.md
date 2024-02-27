# Dataset Stats

<p align="center">
  <img src="../assets/OpenScene_data_stats.gif" width="996px" >
</p>





## The Largest Up-to-Date Dataset in Autonomous Driving
Comparison to prevailing benchmarks in the wild: 


|  Dataset  |      Sensor Data (hr)     | Scan | Annotated Fame |  Sensor Setup | Annotation | Ecosystem |
|:---------:|:--------------------:|:---------:|:-------------:|:------:|:--------------------------------------------:|:----------------:|
| [KITTI](https://www.cvlibs.net/datasets/kitti/index.php)  |           1.5  |  15K | 15K         | 1L 2C    | 3D box, segmentation, depth, flow | [Leaderboard](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) |
| [Waymo](https://waymo.com/open/)   |             6.4  |  230K | 230K   | 5L 5C    | 3D box, flow  | [Challenge](https://waymo.com/open/challenges/) |
| [nuScenes](https://www.nuscenes.org/)   |             5.5  |  390K | 40K  | 1L 6C  | 3D box, segmentation  | [Leaderboard](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) |
| [Lyft](https://self-driving.lyft.com/level5/data/) | 2.5|   323K | 46K | 3L 7C | 3D box | - |
| [ONCE](https://once-for-auto-driving.github.io/)   |            144  |  1M | 15K | 1L 7C  | 3D box, 3D lane  | - |
| [BDD100k](https://www.vis.xyz/bdd100k/)   |            1000  |  100K | 100K| 1C  | 2D box, 2D lane :cry:  | [Workshop](https://www.vis.xyz/bdd100k/challenges/cvpr2023/) |
| **OpenScene** |          **:boom: 120**  |  **:boom: 40M** |  **:boom: 4M** | 5L 8C  | Occupancy :smile: | [Leaderboard](https://opendrivelab.com/AD23Challenge.html#Track3) <br> [Challenge](https://opendrivelab.com/AD24Challenge.html) <br> [Workshop](https://opendrivelab.com/e2ead/cvpr23.html) |

> L: LiDAR, C: Camera


## Fact Sheet

<center>

|  Type  | Info | 
|:---------:|:-----------------|
| Location | Las Vegas (64%), Singapore (15%), Pittsburgh (12%), Boston (9%) |
| Duration | 15910 logs, 120+ hours |
| Scenario category | Dynamics: 5 types (e.g. high lateral acceleration) <br>  Interaction: 18 types (e.g. waiting for pedestrians to cross) <br> Zone: 8 types (e.g. on pickup-dropoff area) <br> Maneuver: 22 types (e.g. unprotected cross turn) <br>  Behavior: 22 types (e.g. stopping at a traffic light with a lead vehicle ahead) |
| Track| Frequency of tracks/ego: 2hz <br> Average length of scenes: 20s |
| Class| Vehicle, Bicycle, Pedestrian, Traffic cone, Barrier, Construction zone sign, Generic object, Background |
| Split | Trainval (1310 logs), Test (147 logs), Mini (64 logs) |
| Voxel | Range: [-50m, -50m, -4m, 50m, 50m, 4m]; Size: 0.5m |
<!---| Scenarios |  Total unique scenario types |--->

</center>

## Download (need to be modified)

The files mentioned below can also be downloaded via [OpenDataLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenLane-V2).
It is recommended to use provided command line interface (CLI) for acceleration.

| Subset | Split | Google Drive <img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" alt="Google Drive" width="18"/> | Baidu Yun <img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="Baidu Yun" width="18"/> | md5 | Size |
| --- | --- | --- | --- | --- | --- |
| sample | OpenLane-V2 |[sample](https://drive.google.com/file/d/1Ni-L6u1MGKJRAfUXm39PdBIxdk_ntdc6/view?usp=share_link) | [sample](https://pan.baidu.com/s/1ncqwDtuihKTBZROL5vdCAQ?pwd=psev) | 21c607fa5a1930275b7f1409b25042a0 | ~300M |
| subset_A | OpenLane-V2 | [info](https://drive.google.com/file/d/1t47lNF4H3WhSsAqgsl9lSLIeO0p6n8p4/view?usp=share_link) | [info](https://pan.baidu.com/s/1uXpX4hqlMJLm0W6l12dJ-A?pwd=6rzj) |95bf28ccf22583d20434d75800be065d | ~8.8G |
|  | Map Element Bucket | [info](https://drive.google.com/file/d/14Wr2Gv2kyogY7_ZLEClqY0-Uhz4S109f/view?usp=share_link) | [info](https://pan.baidu.com/s/110kXzVro4z1Ysz-POVRTbA?pwd=g6y7) | 1c1f9d49ecd47d6bc5bf093f38fb68c9 | ~240M |
|  | Image (train) | [image_0](https://drive.google.com/file/d/1jio4Gj3dNlXmSzebO6D7Uy5oz4EaTNTq/view?usp=share_link) | [image_0](https://pan.baidu.com/s/12aV4CoT8znEY12q4M8XFiw?pwd=m204) | 8ade7daeec1b64f8ab91a50c81d812f6 | ~14.0G |
|  |  | [image_1](https://drive.google.com/file/d/1IgnvZ2UljL49AzNV6CGNGFLQo6tjNFJq/view?usp=share_link) | [image_1](https://pan.baidu.com/s/1SArnlA2_Om9o0xcGd6-EwA?pwd=khx8) | c78e776f79e2394d2d5d95b7b5985e0f | ~14.3G |
|  |  | [image_2](https://drive.google.com/file/d/1ViEsK5hukjMGfOm_HrCiQPkGArWrT91o/view?usp=share_link) | [image_2](https://pan.baidu.com/s/1ZghG7gwJqFrGxCEcUffp8A?pwd=0xgm) | 4bf09079144aa54cb4dcd5ff6e00cf79 | ~14.2G |
|  |  | [image_3](https://drive.google.com/file/d/1r3NYauV0JIghSmEihTxto0MMoyoh4waK/view?usp=share_link) | [image_3](https://pan.baidu.com/s/1ogwmXwS9u-B9nhtHlBTz5g?pwd=sqeg) | fd9e64345445975f462213b209632aee | ~14.4G |
|  |  | [image_4](https://drive.google.com/file/d/1aBe5yxNBew11YRRu-srQNwc5OloyKP4r/view?usp=share_link) | [image_4](https://pan.baidu.com/s/1tMAmUcZH2SzCiJoxwgk87w?pwd=i1au) | ae07e48c88ea2c3f6afbdf5ff71e9821 | ~14.5G |
|  |  | [image_5](https://drive.google.com/file/d/1Or-Nmsq4SU24KNe-cn9twVYVprYPUd_y/view?usp=share_link) | [image_5](https://pan.baidu.com/s/1sRyrhcSz-izW2U5x3UACSA?pwd=nzxx) | df62c1f6e6b3fb2a2a0868c78ab19c92 | ~14.2G |
|  |  | [image_6](https://drive.google.com/file/d/1mSWU-2nMzCO5PGF7yF9scoPntWl7ItfZ/view?usp=share_link) | [image_6](https://pan.baidu.com/s/1P3zn_L6EIGUHb43qWOJYWg?pwd=4wei) | 7bff1ce30329235f8e0f25f6f6653b8f | ~14.4G |
|  | Image (val) | [image_7](https://drive.google.com/file/d/19N5q-zbjE2QWngAT9xfqgOR3DROTAln0/view?usp=share_link) | [image_7](https://pan.baidu.com/s/1rRkPWg-zG2ygsbMhwXjPKg?pwd=qsvb) | c73af4a7aef2692b96e4e00795120504 | ~21.0G |
|  | Image (test) | [image_8](https://drive.google.com/file/d/1CvT9w0q8vPldfaajI5YsAqM0ZINT1vJv/view?usp=share_link) | [image_8](https://pan.baidu.com/s/10zjKeuAw350fwTYAeuSLxg?pwd=99ch) | fb2f61e7309e0b48e2697e085a66a259 | ~21.2G |
|  | SD Map | [sdmap](https://drive.google.com/file/d/1nTsdxRZy_6N-itYndJujb-Ipwom6A5fh/view?usp=sharing) | [sdmap]( https://pan.baidu.com/s/1BwtWcIZ4cZE-yqcUZAcPrg?pwd=56q4) | de22c7be880b667f1b3373ff665aac2e | ~7M |
| subset_B | OpenLane-V2 | [info](https://drive.google.com/file/d/1Kn1tTwh9VrVa8nKwipL0bs9J5G9YLtIR/view?usp=drive_link) | [info](https://pan.baidu.com/s/16PGcEwD4sUuoxwpnwLR2ug?pwd=2o1b) | 27696b1ed1d99b1f70fdb68f439dc87d | ~7.7G |
|  | Image (train) | [image_0](https://drive.google.com/file/d/1L7Uy6g3brT3kXj25RLqxAIBhxWcFkeDE/view?usp=drive_link) | [image_0](https://pan.baidu.com/s/1roo9vwYtgoHI-Uvh-yVVoQ?pwd=k2re) | 0876c6b2381bacedeb3be16e57c7d59b | ~3.4G |
|  |  | [image_1](https://drive.google.com/file/d/1csxAVAXkfyACmcRMUyjtKdXQubFc7l62/view?usp=drive_link) | [image_1](https://pan.baidu.com/s/1uiCPh2XGb6Fe8JyGrQRVUw?pwd=tq88) | ecdec8ff8c72525af322032a312aad10 | ~3.3G |
|  |  | [image_2](https://drive.google.com/file/d/1JyRI4xvuugarYyR8NAm7hhoroXHTcWcY/view?usp=drive_link) | [image_2](https://pan.baidu.com/s/1XrQyeaKGbRw0OqcX0kGvCg?pwd=77up) | b720bf7fdf0ebd44b71beffc84722359 | ~3.3G |
|  |  | [image_3](https://drive.google.com/file/d/12dRnHAPWxUPYY_H3oZPb0oivqYjTtNQQ/view?usp=drive_link) | [image_3](https://pan.baidu.com/s/1Kjn9RWY1NewPPep6v7hUiQ?pwd=5fib) | ac3bc9400ade6c47c396af4b12bbd0e0 | ~3.4G |
|  |  | [image_4](https://drive.google.com/file/d/1K8eTn9lI_WDNaifLnhnnhL0AzZVZyAYu/view?usp=drive_link) | [image_4](https://pan.baidu.com/s/1fxFcnCi-B_4MW_UWf4wSCA?pwd=1675) | fa4c4a04b5ad3eac817e6368047d0d89 | ~3.5G |
|  |  | [image_5](https://drive.google.com/file/d/1_jveO4-Nn2uju_vfGW5qN4kohp1TsUc8/view?usp=drive_link) | [image_5](https://pan.baidu.com/s/12WBEYbIcNbEQjiFSZDtzcg?pwd=3tvn) | 19d2cc92514e65270779e405d3a93c61 | ~3.6G |
|  |  | [image_6](https://drive.google.com/file/d/1EDBRXvkb6Z32ydcKrAsvgr-dmejiJb5K/view?usp=drive_link) | [image_6](https://pan.baidu.com/s/1YEjg0vLQXEnSX5GLqt4iiA?pwd=5z7j) | d4f56c562f11a6bcc918f2d20441c42c | ~3.3G |
|  | Image (val) | [image_7](https://drive.google.com/file/d/1xSiBF3FMccvH2LPoPl3pgq2Ojv-crFaJ/view?usp=drive_link) | [image_7](https://pan.baidu.com/s/1I3Yh00jJ7DbFGQdns5tuOg?pwd=3pny) | 443045d7a3faf5998af27e2302d3503e | ~5.0G |
|  | Image (test) | [image_8](https://drive.google.com/file/d/1NS-CuOLYq1l0-AYwSeT13L1yjk-ojrSF/view?usp=drive_link) | [image_8](https://pan.baidu.com/s/11yKRL3yonLaG3X4fg0BrZg?pwd=qtt3) | 6ecb7a9e866e29ed73d335c2d897f50e | ~5.4G |

> - `OpenLane-V2` contains annotations for the initial task of OpenLane Topology.
> - `Map Element Bucket` contains annotations for the task of Driving Scene Topology.
> - `Image` and `SD Map` serves as sensor inputs.

For files in Google Drive, you can use the following command by replacing `[FILE_ID]` and `[FILE_NAME]` accordingly:
```sh
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=[FILE_ID]' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=[FILE_ID]" -O [FILE_NAME]
```



## Filesystem Hierarchy
The final hierarchy should look as follows (depending on the splits downloaded above):
```angular2html
~/OpenScene
├── assets
├── docs
├── DriveEngine
│   └── ${USER CODE}
│       ├── project
│       │   └── my_project
│       └── exp
│           └── my_openscene_experiment
└── dataset
    └── openscene-v1.1
        ├── meta_datas
        |     ├── mini
        │     │     ├── 2021.05.12.22.00.38_veh-35_01008_01518.pkl
        │     │     ├── 2021.05.12.22.28.35_veh-35_00620_01164.pkl
        │     │     ├── ...
        │     │     └── 2021.10.11.08.31.07_veh-50_01750_01948.pkl
        |     ├── test
        |     ├── test_with_anno
        |     └── trainval
        |     
        └── sensor_blobs
              ├── mini
              │    ├── 2021.05.12.22.00.38_veh-35_01008_01518                                           
              │    │    ├── CAM_F0
              │    │    │     ├── c082c104b7ac5a71.jpg
              │    │    │     ├── af380db4b4ca5d63.jpg
              │    │    │     ├── ...
              │    │    │     └── 2270fccfb44858b3.jpg
              │    │    ├── CAM_B0
              │    │    ├── CAM_L0
              │    │    ├── CAM_L1
              │    │    ├── CAM_L2
              │    │    ├── CAM_R0
              │    │    ├── CAM_R1
              │    │    ├── CAM_R2
              │    │    └── MergedPointCloud
              │    │            ├── 0079e06969ed5625.pcd
              │    │            ├── 01817973fa0957d5.pcd
              │    │            ├── ...
              │    │            └── fffb7c8e89cd54a5.pcd       
              │    ├── 2021.06.09.17.23.18_veh-38_00773_01140 
              │    ├── ...                                                                            
              │    └── 2021.10.11.08.31.07_veh-50_01750_01948
              ├── test
              └── trainval

```


## Meta Data
Each pkl file is stored in the following format：

```
{
    'token':                                <str> -- Unique record identifier.
    'frame_idx':                            <int> -- Indicates the idx of the current frame.
    'timestamp':                            <int> -- Unix time stamp.
    'log_name':                             <str> -- Short string identifier.
    'log_token':                            <str> -- Foreign key. Points to log from where the data was extracted.
    'scene_name':                           <str> -- Short string identifier.
    'scene_token':                          <str> -- Foreign key pointing to the scene.
    'map_location':                         <str> -- Relative path to the file with the map mask.
    'roadblock_ids':                        <list> -- XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    'vehicle_name':                         <str> -- String identifier for the current car.
    'can_bus':                              <list> -- Used for vehicle communications, including low-level information about position, speed, acceleration, steering, lights, batteries, etc.
    'lidar_path':                           <str> -- The relative address to store the lidar data.
    'lidar2ego_translation':                <list> -- Translation matrix from the lidar coordinate system to the ego coordinate system.
    'lidar2ego_rotation':                   <list> -- Rotation matrix from the lidar coordinate system to the ego coordinate system.
    'ego2global_translation':               <list> -- Translation matrix from ego coordinate system to global coordinate system
    'ego2global_rotation':                  <list> -- Rotation matrix from ego coordinate system to global coordinate system
    'ego_dynamic_state':                    <list> -- XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    'traffic_lights':                       <list> -- XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    'driving_command':                      <list> -- XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    'cams': {
        'CAM_F0': {
            'data_path':                    <str> -- The relative address to store the camera_front_0 data.
            'sensor2lidar_rotation':        <list> -- Rotation matrix from camera_front_0 sensor to lidar coordinate system.
            'sensor2lidar_translation':     <list> -- Translation matrix from camera_front_0 sensor to lidar coordinate system.
            'cam_intrinsic':                <list> -- Intrinsic matrix of the camera 
            'distortion':                   <list> -- XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        }
        'CAM_L0':                           <dict> -- Camera configuration.
        'CAM_R0':                           <dict> -- Camera configuration.
        'CAM_L1':                           <dict> -- Camera configuration.
        'CAM_R1':                           <dict> -- Camera configuration.
        'CAM_L2':                           <dict> -- Camera configuration.
        'CAM_R2':                           <dict> -- Camera configuration.
        'CAM_B0':                           <dict> -- Camera configuration.
    }
    'sample_prev':                          <str> -- Foreign key. Sample that precedes this in time. Empty if start of scene.
    'sample_next':                          <str> -- Foreign key. Sample that follows this in time. Empty if end of scene.
    'ego2global':                           <list> -- Ego to the global coordinate system transformation matrix.
    'lidar2ego':                            <list> -- Lidar to the ego coordinate system transformation matrix.
    'lidar2global':                         <list> -- Lidar to the global coordinate system transformation matrix.
    'anns': {
        'gt_boxes':                         <list> -- Ground truth boxes. (x,y,z, XXXXXXXXXX)
        'gt_names':                         <list> -- Class names.
        'gt_velocity_3d':                   <list> -- Relative velocity.
        'instance_tokens':                  <list> -- Unique record identifier.
        'track_tokens':                     <list> -- Unique record identifier.

    }
    'occ_gt_final_path':                    <str> -- The relative address to store the occupancy gt data.
    'flow_gt_final_path':                   <str> -- The relative address to store the flow gt data.     
}
```

## Sensor Blobs
