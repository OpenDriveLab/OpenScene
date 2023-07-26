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


<p align="right">(<a href="#top">back to top</a>)</p>
