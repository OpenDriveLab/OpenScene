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
    └── openscene-v1.0
        ├── meta_datas
        |     ├── meta_data_trainval/train/val.pkl
        |     ├── meta_data_test.pkl
        |     └── meta_data_mini.pkl
        ├── occupancy
        |     ├── mini
        |     |    ├── log-0001-scene-0001
        |     |    |     └── occ_gt
        |     |    |           ├── 0000_occ_final.npy
        |     |    |           ├── 0001_occ_final.npy
        |     |    |           ├── ...
        |     |    |           ├── 0023_occ_final.npy
        |     |    |           ├── 0000_flow_final.npy
        |     |    |           ├── 0001_flow_final.npy
        |     |    |           ├── ....
        |     |    |           └── 0023_flow_final.npy
        |     |    ├── log-0001-scene-0002
        |     |    ├── ...
        |     |    └── log-0064-scene-0011
        |     ├── trainval
        |     └── test
        └── sensor_blobs   
              ├── 2021.05.12.22.00.38_veh-35_01008_01518                                           
              │    ├── CAM_F0
              │    │     ├── c082c104b7ac5a71.jpg
              │    │     ├── af380db4b4ca5d63.jpg
              │    │     ├── ...
              │    │     └── 2270fccfb44858b3.jpg
              │    ├── CAM_B0
              │    ├── CAM_L0
              │    ├── CAM_L1
              │    ├── CAM_L2
              │    ├── CAM_R0
              │    ├── CAM_R1
              │    └── CAM_R2
              │
              ├── 2021.06.09.17.23.18_veh-38_00773_01140 
              ├── ...                                                                            
              └── 2021.10.11.08.31.07_veh-50_01750_01948
```

