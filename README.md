
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



## Highlights


### :oncoming_automobile: Representing 3D Scene as Occupancy

<!---
![teaser](assets/figs/pipeline.PNG)
--->

As we quote from [OccNet](https://arxiv.org/abs/2306.02851):

>  **Occupancy** serves as a `general` representation of the scene and could facilitate perception and planning in the full-stack of autonomous driving. 3D Occupancy is a geometry-aware representation of the scene.

Compared to the formulation of `3D bounding box` and `BEV segmentation`,  3D occupancy could capture the fine-grained details of critical obstacles in the driving scene.


### :fire: Scale Up Data: A Massive Dataset for Visual Pre-Training and [DriveAGI](https://github.com/OpenDriveLab/DriveAGI)


Comparison to prevailing benchmarks in the wild: 


<!---
|  Dataset  |      Raw Data      |   Annotation Duration  |Sensor Configuration| Annotation Label | 
|:---------:|:--------------------:|:-------------:|:------:|:--------------------------------------------:|
| [KITTI](https://www.cvlibs.net/datasets/kitti/index.php)  |           1.5h  |  1.5h | 1 LiDAR, 2 cameras    | 3D box, segmentation, depth, flow |
| [Waymo](https://waymo.com/open/)   |             6.4h  |  6.4h | 5 LiDARs, 5 cameras    | 3D box  |
| [nuScenes](https://www.nuscenes.org/)   |             5.5h  |  5.5h | 1 LiDARs, 6 cameras  | 3D box, LiDAR segmentation  |
| [ONCE](https://once-for-auto-driving.github.io/)   |            144h  |  2.2h | 1 LiDARs, 7 cameras  | 3D box  |
| ~~[BDD100k](https://www.vis.xyz/bdd100k/)~~   |            1000h  |  1000h | 1 camera  | 2D lane :cry:  |
| **OpenScene** |          **:boom: 1200h**  |  **:boom: 120h** | 5 LiDARs, 8 cameras  | Occupancy |
--->

|  Dataset  |      Sensor Data (hr)     | Scans | Annotation Fames |  Sensor Configuration | Annotation Label | Ecosystem |
|:---------:|:--------------------:|:---------:|:-------------:|:------:|:--------------------------------------------:|:----------------:|
| [KITTI](https://www.cvlibs.net/datasets/kitti/index.php)  |           1.5  |  15k | 15k         | 1L 2C    | 3D box, segmentation, depth, flow | Leaderboard |
| [Waymo](https://waymo.com/open/)   |             6.4  |  230k | 230k   | 5L 5C    | 3D box  | Challenge |
| [nuScenes](https://www.nuscenes.org/)   |             5.5  |  390k | 40k  | 1L 6C  | 3D box, LiDAR segmentation  | Leaderboard |
| [ONCE](https://once-for-auto-driving.github.io/)   |            144  |  1M | 15k | 1L 7C  | 3D box  | - |
| [BDD100k](https://www.vis.xyz/bdd100k/)   |            1000  |  100k | 100k| 1C  | 2D lane :cry:  | - |
| **OpenScene** |          **:boom: 120**  |  **:boom: 40M** |  **:boom: 4M** | 5L 8C  | Occupancy | Leaderboard Challenge Workshop |

> L: LiDAR, C: Camera

**OpenScene: The Largest Dataset for Occupancy**

Driving behavior on a sunny day does not apply to that in dancing snowflakes. For machine learning, data is the `must-have` food. 
To highlight, we build OpenScene on top of [nuPlan](https://www.nuscenes.org/nuplan#challenge), covering a wide span of over **120 hours** of occupancy labels collected in various cities, from `Austin`, `Boston`, `Miami` to `Singapore`.
<!---The diversity of data enables models to generalize in different atmospheres and landscapes.--->


<center>
  
|  Dataset  | Original Database |      Sensor Data (hr)    |   Sweeps  | Flow | Semantic Classes                               |
|:---------:|:-----------------:|:--------------------:|:-------------:|:------:|:--------------------------------------------:|
| [MonoScene](https://github.com/astra-vision/MonoScene)  |      NYUv2/SemanticKITTI     | 4.1  |  1 | :x:     | 19   |
| [Occ3D](https://github.com/FANG-MING/occupancy-for-nuscenes/tree/main)   |      nuScenes     | 5.5  |  10 | :x:    | 16  |
| [Occupancy-for-nuScenes](https://github.com/FANG-MING/occupancy-for-nuscenes)   |      nuScenes     | 5.5  |  20 | :x:     | 16  |
| [SurroundOcc](https://github.com/weiyithu/SurroundOcc)   |      nuScenes     | 5.5  |  10 | :x:    | 16  |
| [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy)   |      nuScenes     | 5.5  |  10 | :x:     | 16  |
| [SSCBench](https://github.com/ai4ce/SSCBench)   |      KITTI-360/nuScenes/Waymo     | 13.4  |  10 | :x:     | 16  |
| [OccNet](https://github.com/OpenDriveLab/OccNet)   |      nuScenes     | 5.5  |  **40** | :x:     | 16   |
| **OpenScene** |       nuPlan      | **:boom: 120**  |  **40** | **:heavy_check_mark:**    | **7+X** |

</center>

<!---
We consider occupancy as a unified representation for various sub-tasks within autonomous driving perception, general scene understanding, and embodied robotics.
The released OpenScene is the largest dataset with occupancy representation.
The pre-training of occupancy detection tasks on massive datasets is expected to benefit various downstream perception-related tasks.
--->


### Beyond Perception: Empowering DriveAGI with Motional Occupancy


What kind of modeling is needed for autonomous driving scenarios to meet the demands of planning-oriented perception?
<!---
Previous occupancy datasets were annotated in a static scene. 
However, in practical applications, static occupancy data cannot support vehicle route planning due to the lack of instance motion information. Therefore, **occupancy flow** data is indispensable.
--->
We posit that incorporating the motion information of **occupancy flow** can help bridge the gap between `decision-making` and `scene representation`.
Besides, the OpenScene dataset provides semantic labels for each foreground grid. 
We will add semantic labels of background grids in future updates for boosting open-vocabulary detection.

<!---
### Explore the World Fully: Recognizing Scenarios with Semantic Labels



Recognizing whether the geometric space is occupied is only the first step. 
To drive a car, it is also necessary to identify various **traffic elements** (such as vehicles and obstacles) in the environment. 
The richness of the scene's semantics can greatly ensure driving safety.
We hope that the establishment of this dataset can promote the development of driving scene semantic-level perception tasks. 
The OpenScene dataset provides semantic labels for each foreground grid. 
We will add semantic labels for background grids in future updates.
--->

<p align="right">(<a href="#top">back to top</a>)</p>


## Table of Contents
- [Highlights](#Highlights)
- [Task and Evaluation](#tasks)
- [Getting Started](#getting-started)
- [ToDo](#todo)
- [License](#license)
- [Related resources](#related-resources)

## Grad-and-Go

- **`[07/21]`** OpenScene `v1.0` released


<p align="right">(<a href="#top">back to top</a>)</p>

## Fact Sheet

<center>

|  Type  | Info | 
|:---------:|:-----------------|
| Location | Las Vegas (64%), Singapore (15%), Pittsburgh (12%), Boston (9%) |
| Duration | 15910 logs, 120+ hours |
| Scenario categories | Dynamics: 5 types (e.g. high lateral acceleration) <br>  Interaction: 18 types (e.g. waiting for pedestrians to cross) <br> Zone: 8 types (e.g. on pickup-dropoff area) <br> Maneuver: 22 types (e.g. unprotected cross turn) <br>  Behavior: 22 types (e.g. stopping at a traffic light with a lead vehicle ahead) |
| Tracks| Frequency of tracks/ego: 20hz <br> Average length of tracks: 9.64s |
|Object classes| Vehicle, Bicycle, Pedestrian, Traffic cone, Barrier, Construction zone sign, Generic object |
| Split | Trainval (1310 logs), Test (147 logs), Mini (64 logs) |
| Voxel | Range: [-50m, -50m, -4m, 50m, 50m, 4m]; Size: 0.5m |
<!---| Scenarios |  Total unique scenario types |--->

</center>

<p align="right">(<a href="#top">back to top</a>)</p>

## Task and Evaluation Metric

We consider occupancy as a unified representation for various sub-tasks within autonomous driving perception, general scene understanding, and embodied robotics.
The pre-training of occupancy detection tasks on massive data is expected to benefit various downstream perception-related tasks.

### Pre-Training by Occupancy Forecasting

The pre-training task is defined as occupancy forecasting by the intersection-over-union (**mIoU**) over all classes. 

Let $C$ be the number of classes. 

$$
    \text{mIoU}=\frac{1}{C}\displaystyle \sum_{c=1}^{C}\frac{TP_c}{TP_c+FP_c+FN_c},
$$

where $TP_c$, $FP_c$, and $FN_c$ correspond to the number of true positive, false positive, and false negative predictions for class $c_i$.


### Fine-Tunning with Diverse Downstream Tasks
<!---
<center>

| Doenstream Tasks | KITTI | nuScenes | Waymo | Metrics |
|:---------:|:---------:|:---------:|:---------------:|:-------------:|
| 3D Detection |  :heavy_check_mark:| :heavy_check_mark:| :heavy_check_mark:  |   mAP  |
| Semantic Segmantation |  :heavy_check_mark:| :heavy_check_mark:|   |   mIoU  |
| Scene Completion |  :heavy_check_mark:| :heavy_check_mark:|   |   mIoU  |
| Map Generation |  | :heavy_check_mark:|   |   mIoU  |
| Object Tracking | :heavy_check_mark: | :heavy_check_mark:|   |   [HOTA](https://link.springer.com/article/10.1007/s11263-020-01375-2)  |
| Depth Estimation |  :heavy_check_mark:| :heavy_check_mark:| :heavy_check_mark:  |   [SILog](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)  |
| Visual Odometry |  :heavy_check_mark:| |   |   [Translation](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)  |
| Flow Estimation |  :heavy_check_mark:| |  |   [Fl-all](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)  |
| 3D Lane Detection |  | :heavy_check_mark: |  |   mAP  |

</center>
--->

After pre-training, the fine-tuning stage allows for diverse downstream tasks to be defined on various datasets.


<center>
  
| Downstream Tasks | KITTI Metrics | nuScenes Metrics| Waymo Metrics | 
|:---------:|:---------:|:---------:|:---------------:|
| 3D Detection |  :heavy_check_mark: mAP| :heavy_check_mark: [mAP & NDS](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) | :heavy_check_mark:  [AP & APH](https://waymo.com/open/challenges/2020/3d-detection/)|   
| Semantic Segmentation |  :heavy_check_mark: mIoU| :heavy_check_mark: mIoU|  :heavy_check_mark: mIoU |   
| Scene Completion |  :heavy_check_mark: mIoU| :heavy_check_mark: mIoU| :heavy_check_mark: mIoU  |   
| Map Generation | - | :heavy_check_mark: mIoU| :heavy_check_mark: mIoU  |    
| Object Tracking | :heavy_check_mark: [HOTA](https://link.springer.com/article/10.1007/s11263-020-01375-2)| :heavy_check_mark: [AMOTA & AMOTP](https://www.nuscenes.org/tracking?externalData=all&mapData=all&modalities=Any)|  :heavy_check_mark:[MOTA & MOTP](https://waymo.com/open/challenges/2020/3d-tracking/) |  
| Depth Estimation |  :heavy_check_mark: [SILog](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)| :heavy_check_mark: [Abs Rel](https://arxiv.org/abs/2204.03636)| - |
| Visual Odometry |  :heavy_check_mark: [Translation](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)| - |  - | 
| Flow Estimation |  :heavy_check_mark: [Fl-all](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) |  - | :heavy_check_mark: [EPE](https://waymo.com/open/challenges/2022/occupancy-flow-prediction-challenge/) |
| 3D Lane Detection | - |:heavy_check_mark: [mAP](https://github.com/OpenDriveLab/OpenLane-V2/) | :heavy_check_mark: [F1-Score](https://github.com/OpenDriveLab/OpenLane) |  
</center>


## Ecosystem and Leaderboard

### Upcoming Challenge in 2024

We plan to release a trailer version of the upcoming challenge. Please stay tuned for more details in `Late August`.


### CVPR 2023 3D Occupancy Prediction Challenge (Server remains `active`)

Given images from multiple cameras, the goal is to predict the current occupancy state and semantics of each voxel grid in the scene. The voxel state is predicted to be either free or occupied. If a voxel is occupied, its semantic class needs to be predicted, as well. Besides, we also provide a binary observed/unobserved mask for each frame. An observed voxel is defined as an invisible grid in the current camera observation, which is ignored in the evaluation stage.


- Challenge website: [AD23Challenge](https://opendrivelab.com/AD23Challenge.html#Track3)

![Leaderboard](assets/figs/leaderboard-06-10-2023.png)

<p align="right">(<a href="#top">back to top</a>)</p>




### Task 1： Multifaceted-Task Fine-Tunning

Given an occupancy detection network (OccNet) pre-trained on nuPlan, the goal is to fine-tune the network while freezing the backbone parameters, so that the model can adapt to domain shifts and perform as many downstream tasks as possible on the nuScenes dataset. The fine-tuning stage uses the nuScenes Trainval set as data and the downstream tasks include `occupancy detection, 3D detection, map segmentation, and object tracking`.

**Rules:** 
- The occupancy model and its parameters are not allowed to be modified.
- Participants are required to submit labels for at least one category.
- The use of additional data is not permitted.




### Task 2： Unified Large-Scale Pre-Training

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
- [Download Data](/docs/getting_started.md#download-data)
- [Install Devkit](/docs/getting_started.md#install-devkit)
- [Prepare Dataset](/docs/getting_started.md#prepare-dataset)
- [Train a Model](/docs/getting_started.md#train-a-model)



### Download-ToDo
The files mentioned below can also be downloaded via <img src="https://user-images.githubusercontent.com/29263416/222076048-21501bac-71df-40fa-8671-2b5f8013d2cd.png" alt="OpenDataLab" width="18"/>[OpenDataLab](https://opendatalab.com/CVPR2023-3D-Occupancy/download).It is recommended to use provided [command line interface](https://opendatalab.com/CVPR2023-3D-Occupancy/cli) for acceleration.

| Subset | Google Drive <img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" alt="Google Drive" width="18"/> | Baidu Cloud <img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="Baidu Yun" width="18"/> | Size |
| :---: | :---: | :---: | :---: |
| mini | [data](https://drive.google.com/drive/folders/1ksWt4WLEqOxptpWH2ZN-t1pjugBhg3ME?usp=share_link) | [data](https://pan.baidu.com/s/1IvOoJONwzKBi32Ikjf8bSA?pwd=5uv6)  | approx. 440M |
| trainval  | [data](https://drive.google.com/drive/folders/1JObO75iTA2Ge5fa8D3BWC8R7yIG8VhrP?usp=share_link) | [data](https://pan.baidu.com/s/1_4yE0__UDIJS8JtBSB0Bpg?pwd=li5h) | approx. 32G |
| test | [data](https://drive.google.com/drive/folders/1hVs2AzSlEePN7QR502d8q7FoAbdJLxx8?usp=share_link) | [data](https://pan.baidu.com/s/1ElTu7i5gjXz3TwE2L0YBQQ?pwd=jstt) | approx. 6G |

* Mini and trainval data contain three parts -- `imgs`, `gts` and `annotations`. The `imgs` datas have the same hierarchy with the image samples in the original nuScenes dataset.


### Development Kit-ToDo

We provide a baseline model based on [OccNet](https://github.com/OpenDriveLab/OccNet).

Please refer to [getting_started](docs/getting_started.md) for details.




<p align="right">(<a href="#top">back to top</a>)</p>







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
- [DriveAGI](https://github.com/OpenDriveLab/DriveAGI) | [DriveLM](https://github.com/OpenDriveLab/DriveLM) (TBA) | [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2)
- [OccNet](https://github.com/OpenDriveLab/OccNet)
- [Bird's-eye-view Perception](https://github.com/OpenDriveLab/BEVPerception-Survey-Recipe) | [BEVFormer](https://github.com/fundamentalvision/BEVFormer)


<p align="right">(<a href="#top">back to top</a>)</p>










