
<div id="top" align="center">

# OpenScene: Autonomous Grand Challenge Toolkits

**The large-scale dataset used for the `End-to-End Driving` and `Predictive World Model` tracks for the [CVPR 2024 Autonomous Grand Challenge](https://opendrivelab.com/challenge2024).**

<p align="center">
  <img src="../assets/challenge.jpeg" width="900px" >
</p>
</div>

## News

- **`2024/03/18`** We updated the [test metadata](/docs/getting_started.md#test-set) with box annotations, please re-download it.
- **`2024/03/01`** OpenScene `v1.1` released, [change log](/docs/getting_started.md#openscene-v11).
- **`2024/03/01`** We are hosting **CVPR 2024 Autonomous Grand Challenge**.

## Table of Contents

1. [Track: End-to-End Driving at Scale](#navsim)
2. [Track: Predictive World Model](#worldmodel)
3. [Dataset: OpenScene](#dataset)
4. [License and Citation](#license-and-citation)
5. [Related Resources](#resources)

## Track: End-to-End Driving at Scale <a name="navsim"></a>
<div id="top" align="center">
<p align="center">
  <img src="../assets/e2e_banner.png" width="900px" >
</p>
</div>

> - Official website: :globe_with_meridians: [AGC2024](https://opendrivelab.com/challenge2024/#end_to_end_driving_at_scale)
> - Evaluation server: :hugs: [Hugging Face](https://huggingface.co/spaces/AGC2024-P/e2e-driving-2024)
> - Development Kit: :ringed_planet: [NAVSIM](https://github.com/autonomousvision/navsim)

- [Problem Formulation](#navsim-baseline)
- [Evaluation: PDM Score](#navsim-eval)

NAVSIM gathers simulation-based metrics (such as progress and time to collision) for end-to-end driving by unrolling simplified bird's eye view abstractions of scenes for a short simulation horizon. It operates under the condition that the policy has no influence on the environment, which enables efficient, open-loop metric computation while being better aligned with closed-loop evaluations than traditional displacement errors.

### Problem Formulation <a name="navsim-baseline"></a>
Given sensor inputs (multi-view images from 8 cameras, LiDAR, ego states, and discrete navigation commands) for a 2-second history, the end-to-end planner must output a safe trajectory for the ego vehicle to navigate along for the next 4 seconds. More information is available in the [NAVSIM docs](https://github.com/kashyap7x/navsim/blob/internal_main/docs/agents.md).

### Evaluation: PDM Score <a name="navsim-eval"></a>
Fair comparisons are challenging in the open-loop planning literature, due to metrics of narrow scope or inconsistent definitions between different projects. The PDM Score is a combination of six sub-metrics, which provides a comprehensive analysis of different aspects of driving performance. Five of these sub-metrics are discrete-valued, and one is continuous. All metrics are computed after a 4-second non-reactive simulation of the planner output: background actors follow their recorded future trajectories, and the ego vehicle moves based on an LQR controller. More information is available in the [NAVSIM docs](https://github.com/kashyap7x/navsim/blob/internal_main/docs/metrics.md).

## Track: Predictive World Model <a name="worldmodel"></a>
<div id="top" align="center">
<p align="center">
  <img src="assets/pred_banner.png" width="900px" >
</p>
</div>

> - Official website: :globe_with_meridians: [AGC2024](https://opendrivelab.com/challenge2024/#predictive_world_model)
> - Evaluation server: :hugs: [Hugging Face](https://huggingface.co/spaces/AGC2024-P/predictive-world-model-2024)

- [Problem Formulation](#worldmodel-baseline)
- [Evaluation: Chamfer Distance](#worldmodel-eval)

Serving as an abstract spatio-temporal representation of reality, the world model can predict future states based on the current state. The learning process of world models has the potential to provide a pre-trained foundation model for autonomous driving. Given vision-only inputs, the neural network outputs point clouds in the future to testify its predictive capability of the world.

### Problem Formulation <a name="worldmodel-baseline"></a>
Given an visual observation of the world for the past 3 seconds, predict the point clouds in the future 3 seconds based on the designated
future ego-vehicle pose. In other words,
given historical images in 3 seconds and corresponding history ego-vehicle pose information (from -2.5s to 0s, 6 frames under 2 Hz),
the participants are required to forecast future point clouds
in 3 seconds (from 0.5s to 3s, 6 frames under 2Hz) with specified future ego-poses.

All output point clouds should be aligned to the LiDAR coordinates of the ego-vehicle in the `n` timestamp, which spans a
range of 1 to 6 given predicting 6 future frames.

We then evaluate the predicted future point clouds by querying rays. We will provide a set of query rays for testing propose,
and `the participants are required to estimate depth along each ray for rendering point clouds. An example of submission 
will be provided soon.` Our evaluation toolkit will render
point clouds according to ray directions and provided depths by participants, and compute chamfer distance for points within
the range from -51.2m to 51.2m on the X- and Y-axis as the criterion.

For more details, please refer to [ViDAR](https://github.com/OpenDriveLab/ViDAR).

### Evaluation: Chamfer Distance <a name="worldmodel-eval"></a>
Chamfer Distance is used for measuring the similarity of two point sets, which represent shapes or outlines of two scenens.
It compares the similarity between predicted and ground-truth shapes by calculating the average nearest-neighbor distance between
points in one set to points in the other set, and vice versa.

For this challenge, we will compare chamfer distance between predicted point clouds and ground-truth point clouds for points
within the range of -51.2m to 51.2m. Participants are required to provide depths of specified ray directions. Our evaluation
system will render point clouds by ray directions and provided depth for chamfer distance evaluation.

