
<div id="top" align="center">

# OpenScene

**The Largest 3D Occupancy Prediction Benchmark in Autonomous Driving**

<a href="/docs/dataset_stats.md">
  <img alt="OpenScene: v1.0" src="https://img.shields.io/badge/OpenScene-v1.0-blueviolet"/>
</a>
<a href="#license-and-citation">
  <img alt="License: Apache2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"/>
</a>



<p align="center">
  <img src="assets/OpenScene.gif" width="900px" >
</p>

</div>


> - [Medium Blog](https://medium.com/@opendrivelab/introducing-openscene-the-largest-benchmark-for-occupancy-prediction-in-autonomous-driving-74cfc5bbe7b6) | [Zhihu.com]() (in Chinese)
> - [CVPR 2023 Autonomous Driving Challenge - Occupancy Track](https://opendrivelab.com/AD23Challenge.html#3d_occupancy_prediction)
> - Point of contact: [contact@opendrivelab.com](mailto:contact@opendrivelab.com)

## Grad-and-Go

- **`[2023/08/04]`** OpenScene `v1.0` released

## Table of Contents
- [Highlights](#highlights)
- [Task and Evaluation Metric](#task-and-evaluation-metric)
- [Ecosystem and Leaderboard](#ecosystem-and-leaderboard)
- [TODO](#todo)
- [Getting Started](#getting-started)
- [License and Citation](#license-and-citation)
- [Related Resources](#related-resources)

## Highlights


### :oncoming_automobile: Representing 3D Scene as Occupancy



As we quote from [OccNet](https://arxiv.org/abs/2306.02851):

>  **Occupancy** serves as a `general` representation of the scene and could facilitate perception and planning in the full-stack of autonomous driving. 3D Occupancy is a geometry-aware representation of the scene.

Compared to the formulation of `3D bounding box` and `BEV segmentation`,  3D occupancy could capture the fine-grained details of critical obstacles in the driving scene.


### :fire: OpenScene: The Largest Benchmark for 3D Occupancy Prediction


Driving behavior on a sunny day does not apply to that in dancing snowflakes. For machine learning, data is the `must-have` food. 
To highlight, we build OpenScene on top of [nuPlan](https://www.nuscenes.org/nuplan#challenge), covering a wide span of over **120 hours** of occupancy labels collected in various cities, from `Boston`, `Pittsburgh`, `Las Vegas` to `Singapore`.
The stats of the dataset is summarized [here](docs/dataset_stats.md).



<center>
  
|  Dataset  | Original Database |      Sensor Data (hr)    |   Flow | Semantic Category                               |
|:---------:|:-----------------:|:--------------------:|:------:|:--------------------------------------------:|
| [MonoScene](https://github.com/astra-vision/MonoScene)  |      NYUv2 / SemanticKITTI     | 5 / 6  |  :x:     | 10 / 19   |
| [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)   |      nuScenes / Waymo    | 5.5 / 5.7 |  :x:    | 16 / 14 |
| [Occupancy-for-nuScenes](https://github.com/FANG-MING/occupancy-for-nuscenes)   |      nuScenes     | 5.5  |  :x:     | 16  |
| [SurroundOcc](https://github.com/weiyithu/SurroundOcc)   |      nuScenes     | 5.5  |   :x:    | 16  |
| [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy)   |      nuScenes     | 5.5  |  :x:     | 16  |
| [SSCBench](https://github.com/ai4ce/SSCBench)   |      KITTI-360 / nuScenes / Waymo     | 1.8 / 4.7 / 5.6  |   :x:     | 19 / 16 / 14  |
| [OccNet](https://github.com/OpenDriveLab/OccNet)   |      nuScenes     | 5.5  |   :x:     | 16   |
| **OpenScene** |       nuPlan      | **:boom: 120**  |   **:heavy_check_mark:**    | **`TODO`** |

</center>

> - The time span of LiDAR frames accumulated for each occupancy annotation is **20** seconds.
> - Flow: the annotation of motion direction and velocity for each occupancy grid.
> - `TODO`: Full semantic labels of grids would be released in future version






### :fire: OpenScene: Empowering [DriveAGI](https://github.com/OpenDriveLab/DriveAGI) in the era of Foundation Model


> Which formulation is good for modeling the autonomous driving scenarios?

We posit that incorporating the motion information of **occupancy flow** can help bridge the gap between `decision-making` and `scene representation`.
Besides, the OpenScene dataset provides a semantic label for each foreground grid, serving as a crucial initial step toward achieving DriveAGI. 





<p align="right">(<a href="#top">back to top</a>)</p>




## Task and Evaluation Metric

> Disclaimer: The following task (or title) is **_prone_** to change as we are shaping the 2024 edition of the Autonomous Driving Challenge.


### Large-Scale Occupancy Prediction


Given massive images from multiple cameras in OpenScene, the goal is to predict the current occupancy state and semantics of each voxel grid in the scene.
In this task, we use the **[intersection-over-union (mIoU)](docs/metrics.md#miou)** over all classes to evaluate model performance.



Here we provide a naive baseline for the **Large-Scale Occupancy Prediction** on OpenScene `mini` set, trained with 8 Tesla A100 GPUs.

<center>

| Backbone | mIoU |    IoU@Car   |  Precision  | Recall  | Memory  |  Time  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  ResNet-50   |   7.5 (not fully trained)  |   21.4  |  24.4   |   65.3  |  9260   |   43   |
|  VoVNet-99   |  14.4  (not fully trained)   |   35.9  |  46.7   |   76.1  |  14537  |   81   |
> - `mIoU` (%), `IoU@Car` (%), `Precision` (%), and `Recall` (%) are evaluated on 20% OpenScene `mini` set.
> - `Memory` (MB/GPU) and `Time` (hr) are recorded as the reference of resource consumption during training. 

</center>



### Foundation Model Challenge







In this task, given arbitrary data and architecture, we aim to have
a unified backbone (aka, `foundation model`) to effectively address multifaceted downstream tasks.
The **[OpenScene metric (OSM)](docs/metrics.md#osm)** is adopted to evaluate the effectiveness of such a foundation model in all aspects.
In order to train the large model, you can use `OpenScene` or whatever means of solution at your discretion.


<center>

| Downstream Task | KITTI | nuScenes| Waymo | Scene Diversity| OSM |
|:---------:|:---------:|:---------:|:---------------:|:---------:|:---:|
| 3D Detection |  | :heavy_check_mark:  | |   `downtown`  `crowded` | [NDS](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) |
| Semantic Segmentation |  | :heavy_check_mark: |  |  `downtown`  `crowded`  | mIoU |   
| Scene Completion |  | :heavy_check_mark: |   |   `downtown`  `crowded` | mIoU |
| Map Construction |  | :heavy_check_mark: |   |   `downtown`  `crowded`   | mAP  |
| Object Tracking | |  |  :heavy_check_mark: |  `suburb` `nighttime` `rainy`   | [MOTA](https://waymo.com/open/challenges/2020/3d-tracking/)  |  
| Depth Estimation |  :heavy_check_mark: |  |  |  `countryside` `highway`  | [SILog](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)  |
| Visual Odometry |  :heavy_check_mark: |  |   |   `countryside` `highway` |  [Translation](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)  |
| Flow Estimation |  :heavy_check_mark:  |   |  |  `countryside` `highway`  | [Fl-all](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)   |
| 3D Lane Detection |  |  | :heavy_check_mark:  |  `suburb` `nighttime` `rainy` | [F1-Score](https://github.com/OpenDriveLab/OpenLane) |

</center>

> - We consolidate the above metrics to OSM by computing a weighted sum.
> - The listed datasets and tasks are tentative. Please refer to the AD24 challenge (TBA) for details.


<p align="right">(<a href="#top">back to top</a>)</p>




## Ecosystem and Leaderboard

### Upcoming Challenge in 2024

We plan to release a trailer version of the upcoming challenge. Please stay tuned for more details in `Late August`.
- Challenge website: [AD24Challenge](https://opendrivelab.com/AD24Challenge.html) 



### CVPR 2023 3D Occupancy Prediction Challenge (Server Remains `Active`)


- Please submit your great work as we would **`regularly`** maintain this leaderboard!
- Challenge website: [AD23Challenge](https://opendrivelab.com/AD23Challenge.html#3d_occupancy_prediction)

![Leaderboard](https://github.com/OpenDriveLab/OpenScene/assets/29263416/b5407380-be98-42f6-a7a8-447e84676121)

<p align="right">(<a href="#top">back to top</a>)</p>




## TODO 
- [x] OpenScene `v1.0`
- [ ] Full-stack annotation update: background label and camera-view mask
- [ ] Official Announcement for Autonomous Driving Challenge 2024

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started
- [Download Data](/docs/getting_started.md#download-data)
- [Prepare Dataset](/docs/getting_started.md#prepare-dataset)
- [Train a Model](/docs/getting_started.md#train-a-model)


<p align="right">(<a href="#top">back to top</a>)</p>


## License and Citation
All assets (including figures and data) and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.
Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@misc{openscene2023,
      author = {OpenScene Contributors},
      title = {OpenScene: The Largest Up-to-Date 3D Occupancy Prediction Benchmark in Autonomous Driving},
      url = {https://github.com/OpenDriveLab/OpenScene},
      year = {2023}
}

@article{sima2023_occnet,
      title={Scene as Occupancy}, 
      author={Chonghao Sima and Wenwen Tong and Tai Wang and Li Chen and Silei Wu and Hanming Deng  and Yi Gu and Lewei Lu and Ping Luo and Dahua Lin and Hongyang Li},
      year={2023},
      eprint={2306.02851},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Related Resources
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
- [DriveAGI](https://github.com/OpenDriveLab/DriveAGI)  | [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2) | [DriveLM](https://github.com/OpenDriveLab/DriveLM) (TBA)
- [Survey on Bird's-eye-view Perception](https://github.com/OpenDriveLab/BEVPerception-Survey-Recipe) | [BEVFormer](https://github.com/fundamentalvision/BEVFormer) |  [OccNet](https://github.com/OpenDriveLab/OccNet)


<p align="right">(<a href="#top">back to top</a>)</p>










