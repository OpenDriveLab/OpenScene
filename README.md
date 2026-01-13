> [!IMPORTANT]
> 🌟 Stay up to date at [opendrivelab.com](https://opendrivelab.com/#news)!

<div id="top" align="center">

# OpenScene: Autonomous Grand Challenge Toolkits

<p align="center">
  <img src="assets/OpenScene.gif" width="900px" >
</p>
</div>

> - [Medium Blog](https://medium.com/@opendrivelab/introducing-openscene-the-largest-benchmark-for-occupancy-prediction-in-autonomous-driving-74cfc5bbe7b6) | [Zhihu](https://zhuanlan.zhihu.com/p/647953862) (in Chinese)
> - Point of contact: [contact@opendrivelab.com](mailto:contact@opendrivelab.com)

### Description
OpenScene is a compact redistribution of the large-scale [nuPlan](https://www.nuscenes.org/nuplan#challenge) dataset, retaining only relevant annotations and sensor data at 2Hz. This reduces the dataset size by a factor of >10. We cover a wide span of over **120 hours**, and provide additional **occupancy labels** collected in various cities, from `Boston`, `Pittsburgh`, `Las Vegas` to `Singapore`.

OpenScene is also **the large-scale dataset used for the `End-to-End Driving` and `Predictive World Model` tracks for the [CVPR 2024 Autonomous Grand Challenge](https://opendrivelab.com/challenge2024), and the `NAVSIM-v2 End-to-End Driving` track at [CVPR 2025 Autonomous Grand Challenge](https://opendrivelab.com/challenge2025/#navsim-e2e-driving).** Please check the [challenge docs](/docs/challenge_2024.md) for more details.

The stats of the dataset are summarized [here](/docs/dataset_stats.md).

<center>

|  Dataset  | Original Database |      Sensor Data (hr)    |   Flow | Semantic Categories                               |
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

### Getting Started
- [Download Data](/docs/getting_started.md#download-data)
- [Prepare Dataset](/docs/getting_started.md#prepare-dataset)

## License and Citation <a name="license-and-citation"></a>
> Our dataset is based on the [nuPlan Dataset](https://www.nuscenes.org/nuplan) and therefore we distribute the data under [Creative Commons Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license and [nuPlan Dataset License Agreement for Non-Commercial Use](https://www.nuscenes.org/terms-of-use). You are free to share and adapt the data, but have to give appropriate credit and may not use the work for commercial purposes.
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@inproceedings{yang2024vidar,
  title={Visual Point Cloud Forecasting enables Scalable Autonomous Driving},
  author={Yang, Zetong and Chen, Li and Sun, Yanan and Li, Hongyang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

@misc{openscene2023,
  title={OpenScene: The Largest Up-to-Date 3D Occupancy Prediction Benchmark in Autonomous Driving},
  author={OpenScene Contributors},
  howpublished={\url{https://github.com/OpenDriveLab/OpenScene}},
  year={2023}
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

## Related Resources  <a name="resources"></a>
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
- [DriveAGI](https://github.com/OpenDriveLab/DriveAGI)  | [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2) | [DriveLM](https://github.com/OpenDriveLab/DriveLM)
- [Survey on Bird's-eye-view Perception](https://github.com/OpenDriveLab/BEVPerception-Survey-Recipe) | [BEVFormer](https://github.com/fundamentalvision/BEVFormer) |  [OccNet](https://github.com/OpenDriveLab/OccNet)


<p align="right">(<a href="#top">back to top</a>)</p>

