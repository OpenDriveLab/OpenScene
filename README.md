
<div id="top" align="center">

# OpenScene

**The Largest Up-to-Date 3D Occupancy Forecasting Benchmark in Autonomous Driving**

<a href="#数据">
  <img alt="OpenScene-v1: v1.0" src="https://img.shields.io/badge/OpenScene--V1-v1.0-blueviolet"/>
</a>
<a href="#开发工具">
  <img alt="devkit: v0.1.0" src="https://img.shields.io/badge/devkit-v0.1.0-blueviolet"/>
</a>
<a href="#许可说明">
  <img alt="License: Apache2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"/>
</a>
  


<p align="center">
  <img src="assets/videos/OpenScene_vis.gif" width="996px" >
</p>
  
</div>



> - Medium Blog | Zhihu.com (in Chinese)
> - [CVPR 2023 Autonomous Driving Challenge - Occupancy Track](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction)
> - Point of contact: [contact@opendrivelab.com](mailto:contact@opendrivelab.com)


## Grad-and-Go

- **`[07/21]`** OpenScene `v1.0` released


## Table of Contents
- [Highlights](#Highlights)
- [Task and Evaluation](#tasks)
- [Getting Started](#getting-started)
- [ToDo](#todo)
- [License](#license)
- [Related resources](#related-resources)

## Highlights


### :oncoming_automobile: Representing 3D Scene as Occupancy

<!---
![teaser](assets/figs/pipeline.PNG)
--->

As we quote from [OccNet]():

>  We believe **Occupancy** serves as a `general` representation of the scene and could facilitate perception and planning in the full-stack of autonomous driving. 3D Occupancy is a geometry-aware representation of the scene.

Compared to the formulation of `3D bounding box` and `BEV segmentation`,  3D occupancy could capture the fine-grained details of critical obstacles in the driving scene.


### :fire: Scale Up Data: A Massive Dataset for Visual Pre-Training and [DriveAGI](https://github.com/OpenDriveLab/DriveAGI)


Comparison to prevailing benchmarks in the wild: 


<!---
|  Dataset  |      Raw Data      |   Annotation Duration  |Sensor Configuration| Annotation Label | 
|:---------:|:--------------------:|:-------------:|:------:|:--------------------------------------------:|
| KITTI  |           1.5h  |  1.5h | 1 LiDAR, 2 cameras    | 3D box, segmentation, depth, flow |
| Waymo   |             6.4h  |  6.4h | 5 LiDARs, 5 cameras    | 3D box  |
| nuScenes   |             5.5h  |  5.5h | 1 LiDARs, 6 cameras  | 3D box, LiDAR segmentation  |
| ONCE   |            144h  |  2.2h | 1 LiDARs, 7 cameras  | 3D box  |
| ~~BDD100k~~   |            1000h  |  1000h | 1 camera  | 2D lane :cry:  |
| **OpenScene** |          **:boom: 1200h**  |  **:boom: 120h** | 5 LiDARs, 8 cameras  | Occupancy |
--->

|  Dataset  |      Sensor Data (hr)     | Scans | Annotation Fames |  Sensor Configuration | Annotation Label | Ecosystem |
|:---------:|:--------------------:|:---------:|:-------------:|:------:|:--------------------------------------------:|:----------------:|
| KITTI  |           1.5  |  15k | 15k         | 1L 2C    | 3D box, segmentation, depth, flow | Leaderboard |
| Waymo   |             6.4  |  230k | 230k   | 5L 5C    | 3D box  | Challenge |
| nuScenes   |             5.5  |  390k | 40k  | 1L 6C  | 3D box, LiDAR segmentation  | Leaderboard |
| ONCE   |            144  |  1M | 15k | 1L 7C  | 3D box  | - |
| BDD100k   |            1000  |  100k | 100k| 1C  | 2D lane :cry:  | - |
| **OpenScene** |          **:boom: 120**  |  **:boom: 40M** |  **:boom: 4M** | 5L 8C  | Occupancy | Leaderboard Challenge Workshop |

> L: LiDAR, C: Camera

**OpenScene: The Largest Dataset for Occupancy**

Driving behavior in the sunny day does not apply to that in the dancing snowflakes. For machine learning, data is the `must-have` food. 
To highlight, we build OpenScene on top of [nuPlan](), covering a wide span of over **120 hours** of occupancy labels collected in various cities, from Austin, Boston, Miami to Singapore. The diversity of data enables models to generalize in different atmospheres and landscapes.


<center>
  
|  Dataset  | Original Database |      Sensor Data (hr)    |   Sweeps  | Flow | Semantic Classes                               |
|:---------:|:-----------------:|:--------------------:|:-------------:|:------:|:--------------------------------------------:|
| MonoScene  |      NYUv2/SemanticKITTI     | 4.1  |  1 | ×    | 19   |
| [Occ3D](https://github.com/FANG-MING/occupancy-for-nuscenes/tree/main)   |      nuScenes     | 5.5  |  10 | ×    | 16  |
| Occupancy-for-nuScenes   |      nuScenes     | 5.5  |  20 | ×    | 16  |
| SurroundOcc   |      nuScenes     | 5.5  |  10 | ×    | 16  |
| OpenOccupancy   |      nuScenes     | 5.5  |  10 | ×    | 16  |
| SSCBench   |      KITTI-360/nuScenes/Waymo     | 13.4  |  10 | ×    | 16  |
| [OccNet](https://github.com/OpenDriveLab/OccNet)   |      nuScenes     | 5.5  |  **40** | ×    | 16   |
| **OpenScene** |       nuPlan      | **:boom: 120**  |  **40** | **√**    | **7+X** |

</center>


We consider occupancy as a unified representation for various sub-tasks within autonomous driving perception, general scene understanding, and robotics navigation.
The released OpenScene is the largest dataset with occupancy representation.
The pre-training of occupancy detection tasks on massive datasets is expected to benefit various downstream perception-related tasks, such as 3D object detection, semantic segmentation, depth estimation, scene completion, and so on.



### Beyond Perception: Empowering DriveAGI with Occupancy Flow

What kind of modeling is needed for autonomous driving scenarios to meet the demands of planning-oriented perception?
Previous occupancy datasets were annotated in a static scene. 
However, in practical applications, static occupancy data cannot support vehicle route planning due to the lack of instance motion information. Therefore, **occupancy flow** data is indispensable.
We posit that incorporating the motion information of occupancy flow can help bridge the gap between decision-making and scene representation.



### Explore the World Fully: Recognizing Scenarios with Semantic Labels



Recognizing whether the geometric space is occupied is only the first step. 
To drive a car, it is also necessary to identify various **traffic elements** (such as vehicles and obstacles) in the environment. 
The richness of the scene's semantics can greatly ensure driving safety.
We hope that the establishment of this dataset can promote the development of driving scene semantic-level perception tasks. 
The OpenScene dataset provides semantic labels for each foreground grid. 
We will add semantic labels for background grids in future updates.


<p align="right">(<a href="#top">back to top</a>)</p>


## Fact Sheet

<center>

|  Type  | Info | 
|:---------:|:-----------------|
| Location | Las Vegas (64%), Singapore (15%), Pittsburgh (12%), Boston (9%) |
| Duration | 15910 logs, 120+ hours |
| Scenarios |  Total unique scenario types |
| Scenario categories | Dynamics: 5 types (e.g. high lateral acceleration) <br>  Interaction: 18 types (e.g. waiting for pedestrians to cross) <br> Zone: 8 types (e.g. on pickup-dropoff area) <br> Maneuver: 22 types (e.g. unprotected cross turn) <br>  Behavior: 22 types (e.g. stopping at a traffic light with a lead vehicle ahead) |
| Tracks| Frequency of tracks/ego: 20hz <br> Average length of tracks: 9.64s |
|Object classes| Vehicle, Bicycle, Pedestrian, Traffic cone, Barrier, Construction zone sign, Generic object |
| Split | Trainval (1310 logs), Test (147 logs), Mini (64 logs) |
| Voxel | Range: [-50m, -50m, -4m, 50m, 50m, 4m]; Size: 0.5m |


</center>

<p align="right">(<a href="#top">back to top</a>)</p>

## Task and Evaluation Metric

TODO


## Ecosystem and Leaderboard


### CVPR 2023 3D Occupancy Prediction Challenge

Given images from multiple cameras, the goal is to predict the current occupancy state and semantics of each voxel grid in the scene. The voxel state is predicted to be either free or occupied. If a voxel is occupied, its semantic class needs to be predicted, as well. Besides, we also provide a binary observed/unobserved mask for each frame. An observed voxel is defined as an invisible grid in the current camera observation, which is ignored in the evaluation stage.


### Task 1 - Domain-Specific Fine-Tunning

Given an occupancy detection network (OccNet) pre-trained on nuPlan, the goal is to fine-tune the network while freezing the backbone parameters, so that the model can adapt to domain shifts and perform as many downstream tasks as possible on the nuScenes dataset. The fine-tuning stage uses the nuScenes Trainval set as data and the downstream tasks include `occupancy detection, 3D detection, map segmentation, and object tracking`.

**Rules:** 
- The occupancy model and its parameters are not allowed to be modified.
- Participants are required to submit labels for at least one category.
- The use of additional data is not permitted.




### Task 2 - Unified Large-Scale Pre-Training

We provide the full occupancy data of nuPlan, and participants are required to explore a large-scale pre-training method that enables pre-trained models to perform well on various datasets and downstream tasks after fine-tuning on small-batch data. The sub-datasets include `nuScenes, Waymo, and KITTI`, and the sub-tasks include `occupancy detection, 3D detection, map segmentation, and object tracking`. We will make the required fine-tuning datasets publicly available, but the test datasets will not be disclosed.

**Rules:** 
- Participants are not allowed to modify the models specified for each downstream task.
- Participants are required to submit the pre-trained backbone (ResNet-101 / VoVNet-99).
- The use of additional data is not permitted.


<p align="right">(<a href="#top">back to top</a>)</p>


## TODO 
- [x] 3D Occupancy and flow dataset `v1.0`
- [ ] 3D Occupancy prediction code `v1.0`
- [ ] Pre-trained models
- [ ] Semantic labels

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

<p align="right">(<a href="#top">back to top</a>)</p>

Download Data

Tutorial



## License and Citation
All assets (including figures and data) and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.
Please consider citing our paper if the project helps your research with the following BibTex:
```bibtex
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

## Related resources
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [Bird's-eye-view Perception](https://github.com/OpenDriveLab/BEVPerception-Survey-Recipe)
- [OccNet](https://github.com/OpenDriveLab/OccNet)


<p align="right">(<a href="#top">back to top</a>)</p>










