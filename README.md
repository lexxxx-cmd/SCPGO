# SCPGO — Scan Context Pose Graph Optimization

基于 Scan Context 描述子的激光 SLAM 位姿图优化系统，集成 GTSAM iSAM2 增量优化器，支持多线程回环检测、ICP 多层验证、空间近邻回环和优雅退出。

## 系统架构

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 里程计/点云  │ →  │ 关键帧抽取   │ →  │ 位姿图构建   │
│ GPS 缓存     │    │ (process_pg) │    │ (GTSAM iSAM2)│
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
┌──────────────┐    ┌──────────────┐           │
│ Scan Context │ →  │ ICP 多层验证 │ ← 空间近邻│
│ 回环检测     │    │ (3 层校验)   │   回环检测 │
│ (process_lcd)│    │ (process_icp)│           │
└──────────────┘    └──────────────┘           │
                                               ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ RViz 可视化  │ ←  │ 轨迹/地图    │ ←  │ 结果自动保存 │
│ (10/0.1 Hz)  │    │ 发布         │    │ (优雅退出)   │
└──────────────┘    └──────────────┘    └──────────────┘
```

**6 个工作线程：**
- `process_pg` — 关键帧抽取与初始位姿图构建
- `process_lcd` — Scan Context + 空间近邻回环检测 (1 Hz)
- `process_icp` — ICP 回环约束计算与多层验证
- `process_isam` — GTSAM iSAM2 周期优化 (1 Hz)
- `process_viz_path` — 轨迹可视化发布 (10 Hz)
- `process_viz_map` — 全局地图拼接发布 (0.1 Hz)

## 核心特性

| 特性 | 说明 |
|------|------|
| **Scan Context 回环** | 基于极坐标描述子的位置识别，支持旋转不变性 |
| **ICP 三层验证** | fitness score → 几何合理性 → 里程计一致性 |
| **空间近邻回环** | 绕过 SC 直接基于欧氏距离发现回环 |
| **多核函数支持** | Cauchy / DCS / Geman-McClure 可配置 |
| **优雅退出** | Ctrl+C 后先保存所有结果再关闭 |
| **输入静默退出** | rosbag 播完自动触发保存退出 |
| **关键帧保存** | 输出 interactive_slam 兼容格式 |
| **RANSAC 去地面** | 可选开关，提升回环检测鲁棒性 |
| **CUDA 加速** | CMake 集成 CUDA 依赖 |
| **g2o 信息矩阵** | 导出完整 6×6 信息矩阵供离线优化 |

## 依赖

- ROS (Melodic/Noetic)
- **GTSAM** — 因子图优化 (iSAM2)
- **PCL** — 点云处理、ICP 配准
- **OpenCV** — Scan Context 图像运算
- **Ceres** — 非线性优化（备用）
- **Eigen3** — 矩阵运算
- **CUDA** — GPU 加速（可选）
- **Boost** — 文件系统与格式化

## 编译

```bash
mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src
git clone <this-repo> SCPGO
cd ~/catkin_ws
catkin_make -j$(nproc)
source devel/setup.bash
```

## 运行

```bash
# 播放 rosbag + 在线优化
roslaunch SCPGO laser_pgo.launch

# 自定义参数
roslaunch SCPGO laser_pgo.launch \
  sc_dist_thres:=0.3 \
  use_spatial_loop_closure:=true \
  input_silence_timeout:=60.0

# 仅播放数据，结束后自动保存退出
rosbag play your_data.bag --clock
```

## 关键参数

### 关键帧
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `keyframe_meter_gap` | 2.0 | 平移间隔 (m) |
| `keyframe_deg_gap` | 30.0 | 旋转间隔 (°) |

### Scan Context
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sc_dist_thres` | 0.4 | 回环距离阈值 |
| `sc_max_radius` | 80.0 | 最大扫描半径 (m) |
| `sc_lidar_height` | 0.3 | LiDAR 安装高度 (m) |

### ICP 验证
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `icp_fitness_score_threshold` | 0.2 | ICP 匹配分数阈值 |
| `icp_max_correspondence_distance` | 150.0 | 最大对应点距离 (m) |
| `icp_max_translation` | 50.0 | 修正量平移上限 (m) |
| `icp_max_rotation_deg` | 30.0 | 修正量旋转上限 (°) |
| `icp_min_source_points` | 50 | 最少源点数 |
| `icp_min_target_points` | 50 | 最少目标点数 |

### 回环核函数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `loop_kernel_type` | geman_mcclure | cauchy / dcs / geman_mcclure |
| `loop_kernel_param` | 1.0 | 鲁棒核参数 k |
| `loop_noise_score` | 0.5 | 回环噪声方差 |

### 空间近邻回环
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_spatial_loop_closure` | true | 启用开关 |
| `spatial_loop_radius` | 5.0 | 欧氏距离阈值 (m) |
| `spatial_loop_fitness_thres` | 0.2 | 更严格的 ICP fitness |
| `spatial_loop_min_separation` | 50 | 最少间隔帧数 |

### 功能开关
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_ground_removal` | false | RANSAC 去地面 |
| `use_icp_submap_enhancement` | false | 滑动窗口 + 空间近邻子地图增强 |
| `save_keyframes` | true | 保存关键帧到磁盘 |
| `input_silence_timeout` | 50.0 | 输入静默超时 (s)，0 禁用 |

## 输出文件

```
output/
├── optimized_poses.txt      # TUM 格式优化位姿
├── odom_poses.txt            # KITTI 格式里程计位姿
├── all_optimized_poses.txt   # 全帧优化位姿（插值）
├── gragh.g2o                 # g2o 格式位姿图（含信息矩阵）
├── global_map.pcd            # 全局拼接地图
├── times.txt                 # 时间戳
└── keyframes/                # 关键帧目录
    ├── 000000/
    │   ├── cloud.pcd         # 降采样点云
    │   ├── raw.pcd           # 全分辨率点云
    │   └── data              # 元数据
    ├── 000001/
    └── ...
```

## 话题

### 订阅
| 话题 | 类型 | 说明 |
|------|------|------|
| `/velodyne_cloud_registered_local` | PointCloud2 | 去畸变点云 |
| `/aft_mapped_to_init` | Odometry | 里程计位姿 |
| `/gps/fix` | NavSatFix | GPS (可选) |

### 发布
| 话题 | 类型 | 说明 |
|------|------|------|
| `/aft_pgo_odom` | Odometry | 优化后里程计 |
| `/aft_pgo_path` | Path | 优化后轨迹 |
| `/aft_pgo_map` | PointCloud2 | 优化后全局地图 |
| `/loop_scan_local` | PointCloud2 | 回环当前帧点云 |
| `/loop_submap_local` | PointCloud2 | 回环历史子地图 |
| `/loop_scan_icp` | PointCloud2 | ICP 配准后当前帧 |
| `/loop_submap_icp` | PointCloud2 | ICP 配准后子地图 |

## 参考

- [Scan Context: Egocentric Spatial Descriptor for Place Recognition within 3D Point Cloud Map (IROS 2018)](https://github.com/gisbi-kim/SC-A-LOAM)
- [GTSAM: Georgia Tech Smoothing and Mapping](https://github.com/borglab/gtsam)

## License

MIT
