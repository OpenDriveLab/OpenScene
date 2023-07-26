### ~~Task 1： Multifaceted-Task Fine-Tunning~~

Given an occupancy detection network (OccNet) pre-trained on nuPlan, the goal is to fine-tune the network while freezing the backbone parameters, so that the model can adapt to domain shifts and perform as many downstream tasks as possible on the nuScenes dataset. The fine-tuning stage uses the nuScenes Trainval set as data and the downstream tasks include `occupancy detection, 3D detection, map segmentation, and object tracking`.

**Rules:** 
- The occupancy model and its parameters are not allowed to be modified.
- Participants are required to submit labels for at least one category.
- The use of additional data is not permitted.




### :fire:Task 2： Unified Large-Scale Pre-Training

We provide the full occupancy data of nuPlan, and participants are required to explore a large-scale pre-training method that enables pre-trained models to perform well on various datasets and downstream tasks after fine-tuning on small-batch data. The sub-datasets include `nuScenes, Waymo, and KITTI`, and the sub-tasks include `occupancy detection, 3D detection, map segmentation, and object tracking`. We will make the required fine-tuning datasets publicly available, but the test datasets will not be disclosed.

**Rules:** 
- Participants are not allowed to modify the models specified for each downstream task.
- Participants are required to submit the pre-trained backbone (ResNet-101 / VoVNet-99).
- The use of additional data is not permitted.

### Evaluation Metric

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

<p align="right">(<a href="#top">back to top</a>)</p>
