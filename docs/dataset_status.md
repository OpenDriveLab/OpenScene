## Largest Up-to-Date Dataset in Autonomous Driving
Comparison to prevailing benchmarks in the wild: 



|  Dataset  |      Sensor Data (hr)     | Scans | Annotation Fames |  Sensor Configuration | Annotation Label | Ecosystem |
|:---------:|:--------------------:|:---------:|:-------------:|:------:|:--------------------------------------------:|:----------------:|
| [KITTI](https://www.cvlibs.net/datasets/kitti/index.php)  |           1.5  |  15k | 15k         | 1L 2C    | 3D box, segmentation, depth, flow | Leaderboard |
| [Waymo](https://waymo.com/open/)   |             6.4  |  230k | 230k   | 5L 5C    | 3D box  | Challenge |
| [nuScenes](https://www.nuscenes.org/)   |             5.5  |  390k | 40k  | 1L 6C  | 3D box, LiDAR segmentation  | Leaderboard |
| [ONCE](https://once-for-auto-driving.github.io/)   |            144  |  1M | 15k | 1L 7C  | 3D box  | - |
| [BDD100k](https://www.vis.xyz/bdd100k/)   |            1000  |  100k | 100k| 1C  | 2D lane :cry:  | - |
| **OpenScene** |          **:boom: 120**  |  **:boom: 40M** |  **:boom: 4M** | 5L 8C  | Occupancy | Leaderboard Challenge Workshop |

> L: LiDAR, C: Camera


## Fact Sheet

<center>

|  Type  | Info | 
|:---------:|:-----------------|
| Location | Las Vegas (64%), Singapore (15%), Pittsburgh (12%), Boston (9%) |
| Duration | 15910 logs, 120+ hours |
| Scenario categories | Dynamics: 5 types (e.g. high lateral acceleration) <br>  Interaction: 18 types (e.g. waiting for pedestrians to cross) <br> Zone: 8 types (e.g. on pickup-dropoff area) <br> Maneuver: 22 types (e.g. unprotected cross turn) <br>  Behavior: 22 types (e.g. stopping at a traffic light with a lead vehicle ahead) |
| Tracks| Frequency of tracks/ego: 20hz <br> Average length of tracks: 9.64s |
|Foreground classes| Vehicle, Bicycle, Pedestrian, Traffic cone, Barrier, Construction zone sign, Generic object |
| Split | Trainval (1310 logs), Test (147 logs), Mini (64 logs) |
| Voxel | Range: [-50m, -50m, -4m, 50m, 50m, 4m]; Size: 0.5m |
<!---| Scenarios |  Total unique scenario types |--->

</center>

<p align="right">(<a href="#top">back to top</a>)</p>
