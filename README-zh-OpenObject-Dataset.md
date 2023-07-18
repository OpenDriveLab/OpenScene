<div id="top" align="center">

# OpenObject-nuPlan
<!-- **全球最大的自动驾驶占用栅格感知和预测基准数据集** -->
**The World's Largest Up-to-Date 3D Occupancy Forecasting Dataset in Autonomous Driving.**

<a href="#数据">
  <img alt="OpenObject-v1: v1.0" src="https://img.shields.io/badge/OpenObject--V1-v1.0-blueviolet"/>
</a>
<a href="#开发工具">
  <img alt="devkit: v0.1.0" src="https://img.shields.io/badge/devkit-v0.1.0-blueviolet"/>
</a>
<a href="#许可说明">
  <img alt="License: Apache2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"/>
</a>
  
**中文 | [English](./README-en-OpenObject-Dataset.md)**
  
  
<img src="./imgs/poster.gif" width="696px">

</div>


## 目录

- [关键](#特色---我们为什么是独一无二的)
- [任务](#任务)
  - [3D车道线检测🛣️](#3d车道线检测-%EF%B8%8F)
  - [交通标志检测🚥](#交通标志检测-)
  - [拓扑认知🕸️](#拓扑认知-%EF%B8%8F)
- [信息发布](#信息发布)
- [数据](#数据)
- [开发工具](#开发工具)
- [入门指南](#入门指南)
- [训练模型](#训练模型)
- [基准和排行榜](#基准和排行榜)
- [引用](#引用)
- [许可说明](#许可说明)

## Highlight - Why Are We Exclusive?


### Represent 3D Scenes as Occupancy: A Generic Approach


![teaser](assets/figs/pipeline.PNG)
:oncoming_automobile: We believe **Occupancy** serves as a `general` representation of the scene and could facilitate perception and planning in the full-stack of autonomous driving. 
3D Occupancy is a geometry-aware representation of the scene. Compared to the form of 3D bounding box & BEV segmentation,  3D occupancy could capture the fine-grained details of critical obstacles in the scene.


### Scale Up Your Data: A Massive Dataset for Visual Pre-Training

Experience from the sunny day does not apply to the dancing snowflakes. For machine learning, data is the must-have food. 
We provide over **170 hours** of occupancy labels collected in various cities, from Austin to Singapore and from Boston to Miami. 
The diversity of data enables models to generalize in different atmospheres and landscapes.



### Beyond Perception: Empowering DriveAGI with Occupancy Forecasting


### Explore the World Fully: Recognizing Scenarios with Semantic Labels



道路环境需要怎样的建模才能满足车辆自动驾驶功能的需求？

以往的数据集一般都在2D图像上标注车道，但在实际应用中，由于缺乏高度信息，2D车道数据无法支持车辆在上下坡、路面不平整等场景行驶，因此**3D车道**数据是不可或缺的。

同我们之前的工作 [OpenLane](https://github.com/OpenDriveLab/OpenLane) 一致，OpenLane-Huawei 提供三维车道标注。

<!--
### 关注交通要素，确保行车规范
在道路上平稳的行驶只是第一步，驾驶汽车还需要遵循环境中的**交通要素**（交通灯，路标）信息指示，确保不违反当地交通法规，因此 OpenLane-Huawei 也提供了2D交通要素标注。

交通要素在2D图像上尺寸大多都较小，检测领域中小目标检测一直是个热点，这也是该基准的挑战之一。

### 拓扑关系支撑场景结构化 
环境中交通要素应仅对其特定的车道有指示作用，遵循错误的信号将造成灾难性的后果，因此需要明确车道与交通要素的关系。环境中的车道也不应是割裂存在的，车道与车道的关系也同样重要。

OpenLane-Huawei 提供车道与车道、车道与交通要素之间的拓扑关系，打造结构化场景。

自动驾驶汽车需要通过**推理**捕捉正确的**拓扑关系**以正确的方式行驶。
我们希望这个数据集的建立可以促进**道路场景感知和推理**任务的发展。
-->

### 多元大规模数据，搭建海量数据集
对于机器学习任务，数据是必备的基石。
我们提供从奥斯汀到新加坡，从波士顿到迈阿密等各个城市收集的数据的标注，
支撑数据的**多元性**。



### 助力构建端到端决策



<p align="right">(<a href="#top">回到顶部</a>)</p>




## OpenObject
The Largest up-to-date **3D Occupancy Forecasting** dataset for visual pre-training.

**Quick facts:**
- Task: given the large amount of data, predict the 3D occupancy in the environment. 
- Origin dataset: `nuPlan`
- Estimated date of release: `Early August, 2023`
- Repo: https://github.com/OpenDriveLab/OpenObject (TBA)
- Related work: [OccNet](https://github.com/OpenDriveLab/OccNet), [3D Occupancy Prediction Challenge 2023](https://opendrivelab.com/AD23Challenge.html#Track3)




## 任务

数据集的首要任务是**场景结构感知和推理**，这需要模型能够识别周围环境中车道的可行驶状态。
该数据集的任务不仅包括车道中心线和交通要素检测，还包括检测到的对象的拓扑关系识别。
我们定义了[**OpenLane-V2 Score (OLS)**](./docs/metrics.md#openlane-v2-score)，该指标为各个子任务指标的平均值：

$$
\text{OLS} = \frac{1}{4} \bigg[ \text{DET}_{l} + \text{DET}_{t} + f(\text{TOP}_{ll}) + f(\text{TOP}_{lt}) \bigg].
$$

子任务的指标如下所述：

### 3D车道线检测 🛣️

[OpenLane](https://github.com/OpenDriveLab/OpenLane) 数据集是迄今为止第一个真实世界和规模最大的 3D 车道数据集，提供 3D 空间下的车道线标注。
在OpenLane基础上，我们将 3D 车道检测的任务定义如下：从覆盖整个水平 FOV(视场角-Field Of View) 的多视图中检测带方向的 3D 车道中心线。
用平均精度 $mAP_{LC}$ 指标评估车道中心线的检测性能。

<p align="center">
  <img src="./imgs/lane.gif" width="696px" >
</p>

### 交通标志检测 🚥

现有的数据集很少关注交通标志的检测及其语义，但是交通标志是自动驾驶汽车中关键信息。
该属性表示交通要素的语义，例如交通灯的红色。
在这个子任务中，在给定的前视图图像上，要求同时感知交通要素（交通灯和路标）的位置及其属性。
与典型的 2D 检测数据集相比，挑战在于由于室外环境的大规模，交通要素的尺寸很小。
与典型的多分类 2D 检测任务类似， $mAP_{TE}$ 用于衡量交通要素 (TE)综合的检测性能。


<p align="center">
  <img src="./imgs/traffic_element.gif" width="696px" >
</p>


### 拓扑认知 🕸️
我们首先定义在自动驾驶领域识别拓扑关系的任务。
给定多视图图像，该模型学习识别车道中心线之间以及车道中心线与交通要素之间的拓扑关系。
最相似的任务是图领域的连通性预测，其中顶点是给定的，模型只预测边。
在我们的例子中，模型的顶点和边都是未知的。
因此，首先需要检测车道中心线和交通要素，然后建立拓扑关系。
参照连通性预测任务，
我们用 $mAP_{LCLC}$ 评估车道中心线（LCLC）之间的拓扑表现，
用 $mAP_{LCTE}$ 评估车道中心线和交通要素（LCTE）之间的拓扑表现。

<p align="center">
  <img src="./imgs/topology.gif" width="696px" >
</p>

<p align="right">(<a href="#top">回到顶部</a>)</p>




## 信息发布
- [2023/02]
  * 数据集 `v1.0`: `subset_A` 数据发布。
  * 基模型发布。
- [2023/01]
  * 数据集 `v0.1`： OpenLane-Huawei 数据集样本发布。
  * 开发工具 `v0.1.0`： OpenLane-Huawei 开发工具发布。

<p align="right">(<a href="#top">回到顶部</a>)</p>


## 数据

OpenLane-Huawei 数据集是自动驾驶领域用于道路结构感知和推理的大规模数据集。
与 [OpenLane](https://github.com/OpenDriveLab/OpenLane) 数据集一致，我们提供三维空间中的车道真值。与之有区别的是，OpenLane-Huawei 提供的是车道中心线的3D标注，而OpenLane提供的是车道分割线3D标注。此外，我们还提供了交通标志(交通灯和路标)及其属性的2D框标注，和车道中心线之间以及车道中心线与交通要素之间的拓扑关系标注。

数据集分为两个子集。
**`subset_A`作为主要子集，服务于即将到来的比赛和排行榜，比赛中不允许任何外部数据，包括本数据集其他子集**。
`subset_B`可以用来测试模型的泛化能力。
更多信息请参考对应的页面：[使用数据](./data/README.md)、[标注文档](./docs/annotation.md)与[数据统计](./docs/statistics.md)。

现在就[下载](./data/README.md#download)我们的数据集来发现更多!

<p align="right">(<a href="#top">回到顶部</a>)</p>


## 开发工具

我们提供了一个开发工具来方便社区熟悉并使用 OpenLane-Huawei 数据集。
可以通过 `openlanv2` 的API实现数据集的使用，例如加载图像、加载元数据和评估结果。
更多开发工具信息请参考[开发工具](./docs/devkit.md)。


<p align="right">(<a href="#top">回到顶部</a>)</p>

## 入门指南

按照以下步骤熟悉 OpenLane-Huawei 数据集：

1. 运行以下命令安装必要的工具包，完成研究环境准备：

    ```sh
    git clone https://github.com/OpenDriveLab/OpenLane-V2.git
    cd OpenLane-V2
    conda create -n openlanev2 python=3.8 -y
    conda activate openlanev2
    pip install -r requirements.txt
    python setup.py develop
    ```

2. 点击[链接](./data/README.md#download)从合适的渠道下载数据：

    - <img src="https://user-images.githubusercontent.com/29263416/222076048-21501bac-71df-40fa-8671-2b5f8013d2cd.png" alt="OpenDataLab" width="18"/> OpenDataLab，
    - <img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" alt="Google Drive" width="18"/> Google Drive，
    - <img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="百度云" width="18"/> 百度云。

    并将它们移动至 `data/OpenLane-V2/` 文件夹下解压。
    生成的目录结构应[如下](./data/README.md#hierarchy)所示。
    或者用这些命令来下载数据集样本:

    ```sh
    cd data/OpenLane-V2
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Ni-L6u1MGKJRAfUXm39PdBIxdk_ntdc6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Ni-L6u1MGKJRAfUXm39PdBIxdk_ntdc6" -O OpenLane-V2_sample.tar
    md5sum -c openlanev2.md5
    tar -xvf *.tar
    cd ../..
    ```

3. 在 jupyter notebook 上运行 [tutorial](./tutorial.ipynb) 来熟悉数据集与对应的开发工具。


<p align="right">(<a href="#top">回到顶部</a>)</p>

## 训练模型
我们提供不同神经网络训练框架的插件来支持在我们的数据集上训练模型。
如果缺少你常用的训练框架，我们欢迎你的提议或对插件的共同维护。

### mmdet3d

这个[插件](./plugin/mmdet3d/)基于 [mmdet3d v1.0.0rc6](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6)，并且在以下的环境中进行过测试：
- Python 3.8.15
- PyTorch 1.9.1
- CUDA 11.1
- GCC 5.4.0
- mmcv-full==1.5.2
- mmdet==2.26.0
- mmsegmentation==0.29.1

请按照 mmdet3d 的[指引](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc6/docs/en/getting_started.md)来安装这个训练框架。
假设这个数据集安装在 `OpenLane-V2/` 目录下，并且 mmdet3d 安装在 `mmdetection3d/` 目录下，你可以通过软连接的方式将该插件引入到训练框架中：
```
└── mmdetection3d
    └── projects
        ├── example_project
        └── openlanev2 -> OpenLane-V2/plugin/mmdet3d
```
在将数据路径换成你的本地路径之后，你可以使用我们提供的 config 文件 `mmdetection3d/projects/openlanev2/configs/baseline.py` 来进行模型训练和各种 mmdet3d 中支持的操作。
并且可以通过在对模型进行推理时输入不同的选项来获取不同的功能，已经实现的功能有：`--eval-options dump=True dump_dir=/PATH/TO/DUMP` 来存储用于上传测试集结果的文件；`--eval-options visualization=True visualization_dir=/PATH/TO/VIS` 来对模型输出进行可视化。

<p align="right">(<a href="#top">回到顶部</a>)</p>

## 基准和排行榜
我们将提供 OpenLane-Huawei 数据集的初始基准测试，欢迎您在这里添加您的工作!
基准和排行榜将在不久后发布，请持续关注。

| Method | OLS (main metric) (%) | $mAP_{LC}$ (%) | $mAP_{TE}$ (%) | $mAP_{LCLC}$ (%) | $mAP_{LCTE}$ (%) | F-Score* (%) |
| - | - | - | - | - | - | - |
| Baseline | 0.29 | 0.08 | 0.31 | 0.00 | 0.01 | 8.56 |

<sub>*在比赛和排行榜中不考虑车道中心线检测的 F-Score。
  
<p align="right">(<a href="#top">回到顶部</a>)</p>


## 引用

使用 OpenLane-Huawei 时请使用如下引用：
  
```bibtex
@misc{ openlanev2_dataset,
  author = {{OpenLane-V2 Dataset Contributors}},
  title = {{OpenLane-V2: The World's First Perception and Reasoning Benchmark for Scene Structure in Autonomous Driving}},
  url = {https://github.com/OpenDriveLab/OpenLane-V2},
  license = {Apache-2.0},
  year = {2023}
}
```

我们的数据集是基于[NuScenes](https://www.nuscenes.org/nuscenes) 和[Argoverse](https://www.argoverse.org/av2.html)数据集工作拓展而来。如果引用本作，也请使用如下引用：
  
```bibtex
@article{ nuscenes2019,
  author = {Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and Giancarlo Baldan and Oscar Beijbom},
  title = {nuScenes: A multimodal dataset for autonomous driving},
  journal = {arXiv preprint arXiv:1903.11027},
  year = {2019}
}

@INPROCEEDINGS { Argoverse2,
  author = {Benjamin Wilson and William Qi and Tanmay Agarwal and John Lambert and Jagjeet Singh and Siddhesh Khandelwal and Bowen Pan and Ratnesh Kumar and Andrew Hartnett and Jhony Kaesemodel Pontes and Deva Ramanan and Peter Carr and James Hays},
  title = {Argoverse 2: Next Generation Datasets for Self-driving Perception and Forecasting},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks 2021)},
  year = {2021}
}
```

<p align="right">(<a href="#top">回到顶部</a>)</p>

## 许可说明
使用 OpenLane-Huawei 数据集时，您需要在网站上注册并同意 [nuScenes](https://www.nuscenes.org/nuscenes) 和 [Argoverse 2](https://www.argoverse.org/av2.html) 数据集的使用条款。

本项目的发布受 [Apache License 2.0](./LICENSE)许可认证。


<p align="right">(<a href="#top">回到顶部</a>)</p>

