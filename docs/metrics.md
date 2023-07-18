# Metrics

## Large-Scale Occupancy Prediction


The metric for occupancy prediction is defined by the **intersection-over-union (mIoU)** over all classes.
Let $C$ be the number of classes. 

$$
    \text{mIoU}=\frac{1}{C}\displaystyle \sum_{c=1}^{C}\frac{TP_c}{TP_c+FP_c+FN_c},
$$

where $TP_c$, $FP_c$, and $FN_c$ correspond to the number of true positive, false positive, and false negative predictions for class $c_i$.

## All-in-One Model Verification with Diverse Downstream Tasks


We define the **OpenScene metric (OSM)**, which is consolidated by computing a weighted sum of various metrics covering different aspects of the primary task:


```math
\text{OSM} = \frac{1}{9} \bigg[ \text{DET}_{3d} + \text{DET}_{seg} + \text{DET}_{com} + \text{DET}_{map} + \text{DET}_{d} + \text{DET}_{l} + \text{PRE}_{t} + \text{PRE}_{o} + \text{PRE}_{f}   \bigg].
```

To evaluate performances on different aspects of the task, several metrics are adopted:

- $\text{DET}_{3d}$ for NDS on 3D object detection on `nuScenes`,
- $\text{DET}_{seg}$ for mIoU on semantic segmentation on `nuScenes`,
- $\text{DET}_{com}$ for mIoU on scene completion on `nuScenes`,
- $\text{DET}_{map}$ for mAP on map construction on `nuScenes`,
- $\text{DET}_{d}$ for $\text{DET}_d = \text{max}(1 - \text{SILog}/100, 0)$ on depth estimation on `KITTI`,
- $\text{DET}_{l}$ for F1-score on 3D lane detection on `Waymo`,
- $\text{PRE}_{t}$ for MOTA on object tracking on `Waymo`,
- $\text{PRE}_{o}$ for $\text{PRE}_o = \text{max}(1 - \text{Translation}, 0)$  on visual odometry on `KITTI`,
- $\text{PRE}_{f}$ for $\text{PRE}_f = \text{max}(1 - \text{Fl-all}, 0)$  on flow estimation on `KITTI`.

