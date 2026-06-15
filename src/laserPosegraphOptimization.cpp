#include <fstream>
#include <math.h>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <queue>
#include <set>
#include <thread>
#include <iostream>
#include <string>
#include <sstream>
#include <optional>
#include <iomanip>
#include <csignal>
#include <chrono>
#include <map>
#include <algorithm>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include <yaml-cpp/yaml.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h>

#include <eigen3/Eigen/Dense>

// Workaround for Ceres < 2.1 + Eigen >= 3.4 incompatibility.
// Eigen 3.4 removed the ScalarBinaryOpTraits class template that older Ceres
// tries to specialize in jet.h. Provide a minimal primary template so the
// Ceres partial specializations compile.

#include <ceres/ceres.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "SCPGO/aloam_velodyne/common.h"
#include "SCPGO/aloam_velodyne/tic_toc.h"

#include "SCPGO/scancontext/Scancontext.h"
#include "SCPGO/websocket_publisher.h"

// ------------------------------------------------------------
// 替代 ROS 消息类型的纯 C++ 数据结构
// ------------------------------------------------------------

// TUM 格式位姿数据（替代 nav_msgs::Odometry）
// NOTE: OdomData is now defined in websocket_publisher.h so it is
//       visible to both the main TU and the publisher implementation.

// GPS 数据（替代 sensor_msgs::NavSatFix）
struct GpsData {
    double timestamp;
    double latitude, longitude, altitude;
};

// 从 YAML 节点读取参数，带默认值回退
// NOTE: 全程使用 const 引用/指针 + const operator[]，
//       避免 yaml-cpp 0.9.0 非 const operator[] 的 bug（会向内部树插入空条目）。
template<typename T>
T getParamOrDefault(const YAML::Node& node, const std::string& key, const T& default_val)
{
    if (!node.IsMap() || !node[key]) {
        if (!node[key])
            std::cerr << "[CONFIG] key='" << key << "' NOT_FOUND, using default" << std::endl;
        return default_val;
    }
    try {
        return node[key].as<T>();
    } catch (const YAML::Exception& e) {
        std::cerr << "[CONFIG] key='" << key << "' CONVERSION_FAILED: " << e.what()
                  << ", using default" << std::endl;
        return default_val;
    }
}

// 辅助：从 "a.b.c" 格式的 key 中逐级查找 YAML 节点
template<typename T>
T getParamOrDefaultDeep(const YAML::Node& root, const std::string& dotkey, const T& default_val)
{
    const YAML::Node* node = &root;
    std::istringstream ss(dotkey);
    std::string segment;
    while (std::getline(ss, segment, '.')) {
        if (!node->IsMap()) {
            std::cerr << "[CONFIG] key='" << dotkey << "' NOT_A_MAP at segment='" << segment
                      << "', using default" << std::endl;
            return default_val;
        }
        const YAML::Node& child = (*node)[segment];  // const operator[] — 不会触发 0.9.0 bug
        if (!child) {
            std::cerr << "[CONFIG] key='" << dotkey << "' KEY_MISSING at segment='" << segment
                      << "', using default" << std::endl;
            return default_val;
        }
        node = &child;
    }
    try {
        return node->as<T>();
    } catch (const YAML::Exception& e) {
        std::cerr << "[CONFIG] key='" << dotkey << "' CONVERSION_FAILED: " << e.what()
                  << ", using default" << std::endl;
        return default_val;
    }
}

// ------------------------------------------------------------
// 优雅退出标志与信号处理
// Ctrl+C/SIGTERM 只设置标志，不做强制退出；等所有保存完成后才返回
// ------------------------------------------------------------
static std::atomic<bool> g_shutdown_requested{false};

// 输入静默超时自动退出的心跳检测
static std::atomic<double> g_last_input_time{0.0};   // 最后收到消息的 wall time (seconds)
static std::atomic<bool>  g_has_received_input{false}; // 是否已收到过消息（防启动误判）

static void gracefulShutdownHandler(int /*sig*/)
{
    g_shutdown_requested.store(true);
}

using namespace gtsam;

using std::cout;
using std::endl;

// 这个节点的主流程可以按四条线程来理解：
// 1) process_pg 负责消费里程计/点云/GPS，抽关键帧并搭建位姿图；
// 2) process_lcd 负责用 Scan Context 找回环候选；
// 3) process_icp 负责用 ICP 把回环候选变成精确约束；
// 4) process_isam 负责周期性优化并导出结果。

// ------------------------- 关键参数与运行状态 -------------------------
double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame.
double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.
double simulatedSensorHz = 10.0;       // 批量模式传感器模拟频率 (Hz), 0=全速

bool isNowKeyFrame = false;

Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero

// ------------------------- 滑动窗口：管理最近关键帧的降采样点云 -------------------------
const int SLIDING_WINDOW_SIZE = 7; // [-6, 0] 共 7 帧
std::deque<std::pair<int, pcl::PointCloud<PointType>::Ptr>> slidingWindowDeque;
std::mutex mDeque;

// ------------------------- 地面高度经验值与 RANSAC 参数 -------------------------
double estimatedGroundZ = -2.0;   // 地面 Z 经验值（中心帧坐标系，EMA 维护，初始 -2m 典型 LiDAR 高度）
bool groundZInitialized = false;
const double GROUND_Z_EMA_ALPHA = 0.3;   // EMA 平滑系数（新值权重）
const double GROUND_Z_MARGIN = 0.15;     // 裁剪容差（m），低于 estimatedZ + margin 的点被丢弃
bool useGroundRemoval = true;            // nh.param 开关：是否启用 RANSAC 去地面
bool useICPSubmapEnhancement = true;     // nh.param 开关：是否启用 ICP 子地图增强（滑动窗口+空间近邻）

// 回环鲁棒核函数参数（从 launch 文件读取）
double loopNoiseScore = 0.5;       // 回环噪声方差（越小越信任回环）
double loopKernelParam = 1.0;      // 鲁棒核参数 k（越小越激进拒绝异常值）
std::string loopKernelType = "geman_mcclure"; // cauchy / dcs / geman_mcclure

// 空间近邻回环检测参数（从 launch 文件读取）
bool useSpatialLoopClosure = true;        // 是否启用空间近邻回环
double spatialLoopRadius = 10.0;          // 欧氏距离阈值 (m)
double spatialLoopFitnessThres = 0.1;     // 比 SC 回环更严格的 ICP fitness 阈值
int spatialLoopMinSeparation = 50;        // 最少间隔关键帧数（排除近邻自匹配）

// SC 回环世界系距离预检（从配置文件读取）
// 只允许世界坐标系下距离 < 该阈值的关键帧对进入 ICP 验证
double scLoopMaxWorldDistance = 30.0;     // 世界系最大距离 (m)

// ------------------------- 输入缓存：回调只负责入队 -------------------------
std::queue<OdomData> odometryBuf;
std::queue<pcl::PointCloud<PointType>::Ptr> fullResBuf;
std::queue<GpsData> gpsBuf;
std::queue<std::pair<int, int> > scLoopICPBuf;
std::set<std::pair<int, int>> processedLoopPairs; // 已处理的回环对，防止重复添加

std::mutex mBuf;
std::mutex mKF;

// Foxglove WebSocket publisher for real-time streaming
static WebSocketPublisher g_wsPublisher;

double timeLaserOdometry = 0.0;
double timeLaser = 0.0;

// ------------------------- 当前点云与地图缓存 -------------------------
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO(new pcl::PointCloud<PointType>());

// 每个关键帧保存一份局部点云、完整点云、位姿和时间戳。
std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds;
std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserCloudsFull;
std::vector<Pose6D> keyframePoses;
std::vector<Pose6D> keyframePosesUpdated;
std::vector<double> keyframeTimes;
std::atomic<int> recentIdxUpdated{0};

// 所有原始帧的里程计位姿与时间，用于恢复完整轨迹。
std::vector<Pose6D> allFrameOdomPoses;
std::vector<double> allFrameTimestamps;

// ------------------------- GTSAM 位姿图与优化器状态 -------------------------
gtsam::NonlinearFactorGraph gtSAMgraph;
bool gtSAMgraphMade = false;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;

noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
noiseModel::Base::shared_ptr robustGPSNoise;

// ------------------------- Scan Context / ICP 相关对象 -------------------------
pcl::VoxelGrid<PointType> downSizeFilterScancontext;
SCManager scManager;
double scDistThres, scMaximumRadius;

pcl::VoxelGrid<PointType> downSizeFilterICP;
std::mutex mICPFilter;  // 保护 downSizeFilterICP 的多线程访问

// ICP 回环验证参数（从 launch 文件读取）
double icpMaxCorrespondenceDistance;
double icpFitnessScoreThreshold;
int icpMinSourcePoints;
int icpMinTargetPoints;
double icpMaxTranslation;
double icpMaxRotationRad; // 内部使用弧度

std::mutex mtxPosegraph;
std::mutex mtxRecentPose;

// ------------------------- 可视化地图与 GPS 相关状态 -------------------------
pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType>());
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;
bool laserCloudMapPGORedraw = true;

bool useGPS = true;
// bool useGPS = false;
GpsData currGPS;
bool hasGPSforThisKF = false;
bool gpsOffsetInitialized = false; 
double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX = 0.0;
double recentOptimizedY = 0.0;

// ------------------------- 导出文件 -------------------------

std::string save_directory;
std::string pgTUMformat;
std::string odomKITTIformat;
std::fstream pgG2oSaveStream, pgTimeSaveStream;

std::vector<std::string> edges_str; // used in writeEdge

// ------------------------- 关键帧保存开关与配置 -------------------------
bool saveKeyframesEnabled = false;       // nh.param 开关
std::string saveKeyframesDirectory = ""; // 输出根目录
bool saveKeyframesFullCloud = true;      // 是否保存全分辨率点云 (raw.pcd)

// 把 gtsam::Pose3 转成 g2o 需要的 VERTEX_SE3:QUAT 文本行。
std::string getVertexStr(const int _node_idx, const gtsam::Pose3& _Pose)
{
    gtsam::Point3 t = _Pose.translation();
    gtsam::Rot3 R = _Pose.rotation();

    std::string curVertexInfo {
        "VERTEX_SE3:QUAT " + std::to_string(_node_idx) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    // pgVertexSaveStream << curVertexInfo << std::endl;
    // vertices_str.emplace_back(curVertexInfo);
    return curVertexInfo;
}

// 把两帧之间的相对位姿写成 g2o 需要的 EDGE_SE3:QUAT 文本行。
void writeEdge(const std::pair<int, int> _node_idx_pair, const gtsam::Pose3& _relPose,
               const gtsam::noiseModel::Base::shared_ptr& _noise,
               std::vector<std::string>& edges_str)
{
    gtsam::Point3 t = _relPose.translation();
    gtsam::Rot3 R = _relPose.rotation();

    std::string curEdgeInfo {
        "EDGE_SE3:QUAT " + std::to_string(_node_idx_pair.first) + " " + std::to_string(_node_idx_pair.second) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " "
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " "
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    // 提取底层 Diagonal 噪声模型，写入 6×6 上三角信息矩阵（21 个值）
    const gtsam::noiseModel::Diagonal* diagPtr = nullptr;
    if (_noise) {
        auto* robust = dynamic_cast<gtsam::noiseModel::Robust*>(_noise.get());
        if (robust) {
            diagPtr = dynamic_cast<gtsam::noiseModel::Diagonal*>(robust->noise().get());
        } else {
            diagPtr = dynamic_cast<gtsam::noiseModel::Diagonal*>(_noise.get());
        }
    }

    if (diagPtr) {
        Eigen::VectorXd sigmas = diagPtr->sigmas();
        for (int i = 0; i < 6; i++) {
            curEdgeInfo += " " + std::to_string(1.0 / (sigmas(i) * sigmas(i)));
            for (int j = i + 1; j < 6; j++) {
                curEdgeInfo += " 0";
            }
        }
    }

    // pgEdgeSaveStream << curEdgeInfo << std::endl;
    edges_str.emplace_back(curEdgeInfo);
}

// 统一把 Pose6D 转成 GTSAM 的 Pose3，后面构图和导出都走这个表示。
gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3

// 保存最终位姿图为 g2o 格式，便于离线检查或再次优化。
void saveGTSAMgraphG2oFormat(const gtsam::Values& _estimates)
{
    // save pose graph (runs when programe is closing)
    // cout << "****************************************************" << endl; 
    cout << "Saving the posegraph ..." << endl; // giseop

    pgG2oSaveStream = std::fstream(save_directory + "graph.g2o", std::fstream::out);

    int pose_idx = 0;
    for(const auto& _pose6d: keyframePoses) {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);    
        pgG2oSaveStream << getVertexStr(pose_idx, pose) << endl;
        pose_idx++;
    }
    for(auto& _line: edges_str)
        pgG2oSaveStream << _line << std::endl;

    pgG2oSaveStream.close();
}

// 导出关键帧轨迹为 KITTI 格式，方便和常见 odometry 评测工具对接。
void saveOdometryVerticesKITTIformat(std::string _filename)
{
    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);
    for(const auto& _pose6d: keyframePoses) {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);
        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

// 导出优化后的关键帧轨迹为 TUM 格式，包含时间戳和四元数。
void saveOptimizedVerticesTUMformat(gtsam::Values _estimates, const std::vector<double>& _keyframeTimes, std::string _filename)
{
    using namespace gtsam;

    std::fstream stream(_filename.c_str(), std::fstream::out);
    stream.precision(std::numeric_limits<double>::max_digits10);

    for(const auto& key_value: _estimates) {
        auto p = dynamic_cast<const GenericValue<Pose3>*>(&key_value.value);
        if (!p) continue;

        size_t node_idx = key_value.key;
        double timestamp = (node_idx < _keyframeTimes.size()) ? _keyframeTimes[node_idx] : 0.0;

        const Pose3& pose = p->value();
        Point3 t = pose.translation();
        auto q = pose.rotation().toQuaternion();

        stream << std::fixed << std::setprecision(6)
               << timestamp << " "
               << t.x() << " " << t.y() << " " << t.z() << " "
               << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
}

// 读取 TUM 格式位姿文件：timestamp tx ty tz qx qy qz qw
// PCD 文件按时间戳命名且已排序，这里按行序读入 deque，与 scanPcdDirectory 顺序对齐
std::deque<OdomData> loadTumPoses(const std::string& path)
{
    std::deque<OdomData> result;
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "[ERROR] Cannot open TUM poses file: " << path << std::endl;
        return result;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        OdomData odom;
        if (iss >> odom.timestamp >> odom.x >> odom.y >> odom.z
                >> odom.qx >> odom.qy >> odom.qz >> odom.qw) {
            result.push_back(odom);
        }
    }
    std::cout << "[INFO] Loaded " << result.size() << " TUM poses from " << path << std::endl;
    return result;
}

// 扫描 PCD 目录，匹配位姿，构建 FrameData deque（按文件名中的时间戳排序）
std::deque<std::pair<double, std::string>> scanPcdDirectory(const std::string& pcdDir)
{
    std::deque<std::pair<double, std::string>> result;  // (timestamp, pcd_path)
    boost::filesystem::path dir(pcdDir);
    if (!boost::filesystem::exists(dir) || !boost::filesystem::is_directory(dir)) {
        std::cerr << "[ERROR] PCD directory not found: " << pcdDir << std::endl;
        return result;
    }
    for (const auto& entry : boost::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".pcd") {
            std::string stem = entry.path().stem().string();
            // 尝试将 stem 解析为 double timestamp
            double ts = 0.0;
            try { ts = std::stod(stem); }
            catch (...) { /* 非数字文件名，放在最后 */ }
            result.push_back({ts, entry.path().string()});
        }
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::cout << "[INFO] Found " << result.size() << " PCD files in " << pcdDir << std::endl;
    return result;
}

void initNoises( void )
// 初始化位姿图的噪声模型：先验、里程计、回环和 GPS 四类。
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    gtsam::Vector odomNoiseVector6(6);
    // odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    // 回环噪声：方差 + 鲁棒核函数（可从 launch 文件调节）
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore,
                          loopNoiseScore, loopNoiseScore, loopNoiseScore;

    auto baseNoise = gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6);

    if (loopKernelType == "geman_mcclure") {
        robustLoopNoise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::GemanMcClure::Create(loopKernelParam), baseNoise);
    } else if (loopKernelType == "dcs") {
        robustLoopNoise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::DCS::Create(loopKernelParam), baseNoise);
    } else {
        // 默认 fallback 到 Cauchy（与原行为一致）
        robustLoopNoise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(loopKernelParam), baseNoise);
    }

    double bigNoiseTolerentToXY = 1000000000.0; // 1e9
    double gpsAltitudeNoiseScore = 250.0; // if height is misaligned after loop clsosing, use this value bigger
    gtsam::Vector robustNoiseVector3(3); // gps factor has 3 elements (xyz)
    // 这里只重点约束 GPS 的高度分量，X/Y 给足够大的噪声，避免平面位置把轨迹拉偏。
    robustNoiseVector3 << bigNoiseTolerentToXY, bigNoiseTolerentToXY, gpsAltitudeNoiseScore; // means only caring altitude here. (because LOAM-like-methods tends to be asymptotically flyging)
    robustGPSNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3) );

} // initNoises
// 从里程计数据构造 Pose6D。
Pose6D getOdom(const OdomData& _odom)
{
    // 四元数 → RPY
    double sinr_cosp = 2.0 * (_odom.qw * _odom.qx + _odom.qy * _odom.qz);
    double cosr_cosp = 1.0 - 2.0 * (_odom.qx * _odom.qx + _odom.qy * _odom.qy);
    double roll = std::atan2(sinr_cosp, cosr_cosp);

    double sinp = 2.0 * (_odom.qw * _odom.qy - _odom.qz * _odom.qx);
    double pitch;
    if (std::abs(sinp) >= 1.0)
        pitch = std::copysign(M_PI / 2.0, sinp);
    else
        pitch = std::asin(sinp);

    double siny_cosp = 2.0 * (_odom.qw * _odom.qz + _odom.qx * _odom.qy);
    double cosy_cosp = 1.0 - 2.0 * (_odom.qy * _odom.qy + _odom.qz * _odom.qz);
    double yaw = std::atan2(siny_cosp, cosy_cosp);

    return Pose6D{_odom.x, _odom.y, _odom.z, roll, pitch, yaw};
} // getOdom

// 计算两个位姿之间的变化量，用于关键帧判定。
Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
{
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    // 这里取绝对值，是为了把位移/转角累计成“是否该切关键帧”的触发量。
    // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll << ", " << dpitch << ", " << dyaw << std::endl;

    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} // SE3Diff

// RANSAC 地面去除：拟合平面 + 法向检查 + EMA 维护地面 Z 经验值 + Z 裁剪残点。
// 如果平面不满足地面条件（法向不接近Z轴、内点比例异常），仅用 EMA 维护值做 Z 裁剪，不做 RANSAC 提取。
pcl::PointCloud<PointType>::Ptr removeGroundRANSAC(
    const pcl::PointCloud<PointType>::Ptr &cloudIn,
    float distanceThreshold = 0.2)
{
    if (cloudIn->size() < 100)
        return cloudIn;  // 点数过少不做处理

    bool ransacOK = false;
    double ransacGroundZ = estimatedGroundZ;

    // ---------- Step 1: RANSAC 平面拟合 ----------
    pcl::SACSegmentation<PointType> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distanceThreshold);
    seg.setMaxIterations(100);
    seg.setInputCloud(cloudIn);
    seg.segment(*inliers, *coeff);

    if (!inliers->indices.empty()) {
        // 法向检查：地面法向量应接近 (0, 0, 1)
        Eigen::Vector3f groundNormal(coeff->values[0], coeff->values[1], coeff->values[2]);
        float zAlignment = std::abs(groundNormal.dot(Eigen::Vector3f::UnitZ()));
        float inlierRatio = float(inliers->indices.size()) / float(cloudIn->size());

        if (zAlignment >= 0.7 && inlierRatio >= 0.1 && inlierRatio <= 0.8) {
            // 平面方程 ax+by+cz+d=0，原点 (0,0) 处 z = -d/c
            ransacGroundZ = -coeff->values[3] / coeff->values[2];
            ransacOK = true;

            std::cout << "[RANSAC] Ground plane found: Z=" << ransacGroundZ
                      << " (inliers=" << int(inlierRatio * 100)
                      << "%, normal_z=" << zAlignment << ")" << std::endl;
        }
    }

    // ---------- Step 2: EMA 更新地面经验高度 ----------
    if (ransacOK) {
        if (!groundZInitialized) {
            estimatedGroundZ = ransacGroundZ;
            groundZInitialized = true;
        } else {
            estimatedGroundZ = (1.0 - GROUND_Z_EMA_ALPHA) * estimatedGroundZ
                             + GROUND_Z_EMA_ALPHA * ransacGroundZ;
        }
    }
    // 即使 RANSAC 本次失败，仍然沿用上次的 estimatedGroundZ 做 Z 裁剪

    // ---------- Step 3: RANSAC 提取非地面（如果成功） ----------
    pcl::PointCloud<PointType>::Ptr workingCloud(new pcl::PointCloud<PointType>());
    if (ransacOK) {
        pcl::ExtractIndices<PointType> extract;
        extract.setInputCloud(cloudIn);
        extract.setIndices(inliers);
        extract.setNegative(true);  // 去掉地面内点
        extract.filter(*workingCloud);
    } else {
        *workingCloud = *cloudIn;  // RANSAC 不可靠，保留全云，仅靠 Z 裁剪
    }

    // ---------- Step 4: Z 裁剪，清除 RANSAC 漏掉的地面残点 ----------
    pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
    float cutoffZ = float(estimatedGroundZ + GROUND_Z_MARGIN);
    for (const auto& pt : workingCloud->points) {
        if (pt.z > cutoffZ)
            filtered->push_back(pt);
    }

    std::cout << "[RANSAC] " << cloudIn->size() << " → " << filtered->size()
              << " pts (ransac=" << (ransacOK ? "ok" : "skip")
              << ", groundZ_ema=" << estimatedGroundZ
              << ", cutoffZ=" << cutoffZ << ")" << std::endl;

    return filtered;
}

// 把局部点云变换到全局坐标系。
pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
    
    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].r = pointFrom.r;
        cloudOut->points[i].g = pointFrom.g;
        cloudOut->points[i].b = pointFrom.b;
    }

    return cloudOut;
}

// 将 cloudIn（传感器坐标系）先变换到全局坐标系，再通过中心帧位姿逆变换到中心帧坐标系。
// 等价于 T_center^{-1} * T_cloud * cloudIn，但分两步写清变换链。
pcl::PointCloud<PointType>::Ptr local2center(
    const pcl::PointCloud<PointType>::Ptr &cloudIn,
    const Pose6D& cloudPose,    // cloudIn 在全局坐标系下的位姿
    const Pose6D& centerPose)   // 中心帧在全局坐标系下的位姿
{
    // Step 1: cloud 传感器坐标系 → 全局坐标系
    pcl::PointCloud<PointType>::Ptr cloudGlobal = local2global(cloudIn, cloudPose);

    // Step 2: 全局坐标系 → 中心帧传感器坐标系 (T_center^{-1})
    Eigen::Affine3f T_center = pcl::getTransformation(
        centerPose.x, centerPose.y, centerPose.z,
        centerPose.roll, centerPose.pitch, centerPose.yaw);
    Eigen::Affine3f T_center_inv = T_center.inverse();

    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
    int cloudSize = cloudGlobal->size();
    cloudOut->resize(cloudSize);

    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudGlobal->points[i];
        cloudOut->points[i].x = T_center_inv(0,0) * pointFrom.x + T_center_inv(0,1) * pointFrom.y + T_center_inv(0,2) * pointFrom.z + T_center_inv(0,3);
        cloudOut->points[i].y = T_center_inv(1,0) * pointFrom.x + T_center_inv(1,1) * pointFrom.y + T_center_inv(1,2) * pointFrom.z + T_center_inv(1,3);
        cloudOut->points[i].z = T_center_inv(2,0) * pointFrom.x + T_center_inv(2,1) * pointFrom.y + T_center_inv(2,2) * pointFrom.z + T_center_inv(2,3);
        cloudOut->points[i].r = pointFrom.r;
        cloudOut->points[i].g = pointFrom.g;
        cloudOut->points[i].b = pointFrom.b;
    }

    return cloudOut;
}

// pubPath 已移除 —— 去 ROS 化后不再需要 ROS TF 广播和路径可视化发布。
// 优化后的轨迹通过文件导出（TUM / KITTI / g2o 格式），无需 ROS publisher。

// 用 iSAM2 的最新估计回填关键帧位姿，并更新最近优化位姿缓存。
void updatePoses(void)
{
    mKF.lock(); 
    for (int node_idx=0; node_idx < int(isamCurrentEstimate.size()); node_idx++)
    {
        Pose6D& p =keyframePosesUpdated[node_idx];
        p.x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
        p.y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
        p.z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
        p.roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
        p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
        p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
    }
    int kfUpdatedSize = (int)keyframePosesUpdated.size(); // 在 mKF 锁内捕获，避免数据竞争
    mKF.unlock();

    mtxRecentPose.lock();
    const gtsam::Pose3& lastOptimizedPose = isamCurrentEstimate.at<gtsam::Pose3>(int(isamCurrentEstimate.size())-1);
    recentOptimizedX = lastOptimizedPose.translation().x();
    recentOptimizedY = lastOptimizedPose.translation().y();

    recentIdxUpdated.store(kfUpdatedSize - 1);

    mtxRecentPose.unlock();
} // updatePoses

// 触发一次 iSAM2 优化，把新增因子和初始值一起送进优化器。
void runISAM2opt(void)
{
    // called when a variable added 
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    updatePoses();
}

// 按位姿把点云做刚体变换，供子地图拼接和回环验证使用。
pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, gtsam::Pose3 transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(
                                    transformIn.translation().x(), transformIn.translation().y(), transformIn.translation().z(), 
                                    transformIn.rotation().roll(), transformIn.rotation().pitch(), transformIn.rotation().yaw() );
    
    int numberOfCores = 8; // TODO move to yaml 
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        pointFrom = &cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
        cloudOut->points[i].r = pointFrom->r;
        cloudOut->points[i].g = pointFrom->g;
        cloudOut->points[i].b = pointFrom->b;
    }
    return cloudOut;
} // transformPointCloud

// 以给定关键帧为中心收集邻域关键帧，并在全局系下拼成局部子地图。
void loopFindNearKeyframesCloud( pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& submap_size, const int& root_idx)
{
    // extract and stacking near keyframes (in global coord)
    nearKeyframes->clear();
    for (int i = -submap_size; i <= submap_size; ++i) {
        int keyNear = key + i; // see https://github.com/gisbi-kim/SC-A-LOAM/issues/7 ack. @QiMingZhenFan found the error and modified as below.

        mKF.lock();
        if (keyNear < 0 || keyNear >= int(keyframeLaserClouds.size()) ) {
            mKF.unlock();
            continue;
        }
        *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[keyNear]);
        mKF.unlock();
    }

    if (nearKeyframes->empty())
        return;

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    {
        std::lock_guard<std::mutex> lock(mICPFilter);
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
    }
    *nearKeyframes = *cloud_temp;
} // loopFindNearKeyframesCloud


// 用 ICP 估计回环两端的精确相对位姿；失败则返回空值。
std::optional<gtsam::Pose3> doICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx, double _fitness_threshold = -1.0 )
{
    double effectiveFitnessThres = (_fitness_threshold >= 0.0) ? _fitness_threshold : icpFitnessScoreThreshold;
    // parse pointclouds
    int historyKeyframeSearchNum = 25; // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every kf gap is 1m
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());

    if (useICPSubmapEnhancement) {
        // ---- 当前帧子地图：使用滑动窗口 [curr-6, curr]（仅向前追溯历史帧） ----
        cureKeyframeCloud->clear();
        int currWindowStart = std::max(0, _curr_kf_idx - (SLIDING_WINDOW_SIZE - 1));
        for (int i = currWindowStart; i <= _curr_kf_idx; ++i) {
            mKF.lock();
            *cureKeyframeCloud += *local2global(keyframeLaserClouds[i], keyframePosesUpdated[i]);
            mKF.unlock();
        }
        // 降采样
        {
            pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
            {
                std::lock_guard<std::mutex> lock(mICPFilter);
                downSizeFilterICP.setInputCloud(cureKeyframeCloud);
                downSizeFilterICP.filter(*cloud_temp);
            }
            *cureKeyframeCloud = *cloud_temp;
        }

        // ---- 历史帧子地图：时间窗口 ±25 + 空间近邻（15m 半径） ----
        loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx);

        // 空间近邻：暴力搜索 5m 半径内的历史关键帧，合并到 target submap
        {
            const double spatialRadius = 5.0;
            const double spatialRadiusSq = spatialRadius * spatialRadius;
            // 值拷贝位姿（加锁避免 push_back 导致的引用悬空）
            mKF.lock();
            Pose6D loopPose = keyframePosesUpdated[_loop_kf_idx];
            int numClouds = (int)keyframeLaserClouds.size();
            mKF.unlock();

            for (int i = 0; i < numClouds; ++i) {
                // 跳过已在时间窗口内的帧，避免重复合并
                if (std::abs(i - _loop_kf_idx) <= historyKeyframeSearchNum)
                    continue;

                mKF.lock();
                if (i >= (int)keyframeLaserClouds.size()) { mKF.unlock(); continue; }
                Pose6D pose_i = keyframePosesUpdated[i];
                mKF.unlock();

                double dx = pose_i.x - loopPose.x;
                double dy = pose_i.y - loopPose.y;
                double dz = pose_i.z - loopPose.z;
                double distSq = dx*dx + dy*dy + dz*dz;
                if (distSq < spatialRadiusSq) {
                    mKF.lock();
                    // 二次 bounds check：防止 unlock 期间 vector 被 reallocated
                    if (i < (int)keyframeLaserClouds.size()) {
                        *targetKeyframeCloud += *local2global(keyframeLaserClouds[i], keyframePosesUpdated[i]);
                    }
                    mKF.unlock();
                }
            }
        }
        // 对合并空间近邻后的 target 再次降采样
        {
            pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
            {
                std::lock_guard<std::mutex> lock(mICPFilter);
                downSizeFilterICP.setInputCloud(targetKeyframeCloud);
                downSizeFilterICP.filter(*cloud_temp);
            }
            *targetKeyframeCloud = *cloud_temp;
        }
    } else {
        // 原始逻辑：当前帧仅自己，历史帧仅时间窗口
        loopFindNearKeyframesCloud(cureKeyframeCloud, _curr_kf_idx, 0, _loop_kf_idx);
        loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx);
    }

    // ---- 点数预检：source 和 target 必须有足够的点 ----
    int sourcePts = (int)cureKeyframeCloud->size();
    int targetPts = (int)targetKeyframeCloud->size();
    if (sourcePts < icpMinSourcePoints || targetPts < icpMinTargetPoints) {
        cout << "[ICP] Reject: too few points (source=" << sourcePts
             << " < " << icpMinSourcePoints << " or target=" << targetPts
             << " < " << icpMinTargetPoints << ")" << endl;
        return std::nullopt;
    }

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(icpMaxCorrespondenceDistance);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Publish ICP source/target for visualization (before alignment)
    if (g_wsPublisher.enabled) {
        double ts = (_curr_kf_idx < (int)keyframeTimes.size())
                    ? keyframeTimes[_curr_kf_idx] : timeLaserOdometry;
        g_wsPublisher.publishICP(cureKeyframeCloud, targetKeyframeCloud, ts);
    }

    // Align pointclouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    // ---- 校验 1: ICP 收敛 + fitness score ----
    if (icp.hasConverged() == false || icp.getFitnessScore() > effectiveFitnessThres) {
        cout << "[ICP] Reject: fitness " << icp.getFitnessScore()
             << " > " << effectiveFitnessThres << endl;
        return std::nullopt;
    }

    // ---- 校验 2: ICP 修正量几何合理性 ----
    {
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame(icp.getFinalTransformation());
        pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch, yaw);

        double transMag = sqrt(x*x + y*y + z*z);
        double rotMag = sqrt(roll*roll + pitch*pitch + yaw*yaw); // 欧拉角范数近似旋转量

        if (transMag > icpMaxTranslation || rotMag > icpMaxRotationRad) {
            cout << "[ICP] Reject: correction too large (trans=" << transMag
                 << "m max=" << icpMaxTranslation
                 << "m, rot=" << rad2deg(rotMag)
                 << "° max=" << rad2deg(icpMaxRotationRad) << "°)" << endl;
            return std::nullopt;
        }

        // ---- 校验 3: ICP 修正后与里程计预测的一致性 ----
        // 链式里程计预测相对位姿 (loop→curr)
        gtsam::Pose3 odomPredicted = Pose6DtoGTSAMPose3(keyframePoses[_loop_kf_idx]);
        for (int i = _loop_kf_idx + 1; i <= _curr_kf_idx; ++i) {
            gtsam::Pose3 from = Pose6DtoGTSAMPose3(keyframePoses[i-1]);
            gtsam::Pose3 to   = Pose6DtoGTSAMPose3(keyframePoses[i]);
            gtsam::Pose3 delta = from.between(to);
            odomPredicted = odomPredicted.compose(delta);
        }
        gtsam::Pose3 odomRelative = Pose6DtoGTSAMPose3(keyframePoses[_loop_kf_idx])
                                    .between(odomPredicted);

        // ICP 相对位姿
        gtsam::Pose3 poseLoop = Pose6DtoGTSAMPose3(keyframePosesUpdated[_loop_kf_idx]);
        gtsam::Pose3 poseCurr = Pose6DtoGTSAMPose3(keyframePosesUpdated[_curr_kf_idx]);
        gtsam::Pose3 T_icp = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw),
                                           gtsam::Point3(x, y, z));
        gtsam::Pose3 poseCurrCorrected = T_icp.compose(poseCurr);
        gtsam::Pose3 icpRelative = poseLoop.between(poseCurrCorrected);

        // 比较 odom 预测 vs ICP 结果
        gtsam::Pose3 consistencyCheck = odomRelative.between(icpRelative);
        double consistencyTrans = consistencyCheck.translation().norm();
        double consistencyRot = gtsam::Rot3::Logmap(consistencyCheck.rotation()).norm();

        const double MAX_CONSISTENCY_TRANS = icpMaxTranslation * 0.8;  // 一致性平移阈值
        const double MAX_CONSISTENCY_ROT   = icpMaxRotationRad * 0.8;  // 一致性旋转阈值

        if (consistencyTrans > MAX_CONSISTENCY_TRANS || consistencyRot > MAX_CONSISTENCY_ROT) {
            cout << "[ICP] Reject: inconsistent with odometry (trans_diff=" << consistencyTrans
                 << "m max=" << MAX_CONSISTENCY_TRANS
                 << "m, rot_diff=" << rad2deg(consistencyRot)
                 << "° max=" << rad2deg(MAX_CONSISTENCY_ROT) << "°)" << endl;
            return std::nullopt;
        }

        cout << "[ICP] Passed: fitness=" << icp.getFitnessScore()
             << ", corr_trans=" << transMag << "m"
             << ", corr_rot=" << rad2deg(rotMag) << "°"
             << ", consistency_trans=" << consistencyTrans << "m"
             << ", consistency_rot=" << rad2deg(consistencyRot) << "°" << endl;

        return icpRelative;
    }
} // doICPVirtualRelative

// 主线程之一：消费里程计/点云/GPS 缓存，抽关键帧并构建初始位姿图。
void process_pg()
{
    while(!g_shutdown_requested.load())
    {
		// 每次外循环只消费一帧（模拟 ROS callback 节奏），让 isam/lcd 穿插
		if ( !odometryBuf.empty() && !fullResBuf.empty() )
        {
            //
            // pop and check keyframe is or not  
            // 
			mBuf.lock();
            // 数据已按时间排序加载，直接 pop 即可（无需时间戳对齐判断）
            if (odometryBuf.empty())
            {
                mBuf.unlock();
                continue;
            }

            timeLaserOdometry = odometryBuf.front().timestamp;
            timeLaser = timeLaserOdometry;
            // TODO

            laserCloudFullRes->clear();
            pcl::PointCloud<PointType>::Ptr thisKeyFrame = fullResBuf.front();
            fullResBuf.pop();

            OdomData odom_curr = odometryBuf.front();   // save quaternion before conversion
            Pose6D pose_curr = getOdom(odom_curr);
            odometryBuf.pop();

            // find nearest gps
            double eps = 0.1; // find a gps topioc arrived within eps second
            while (!gpsBuf.empty()) {
                auto thisGPS = gpsBuf.front();
                double thisGPSTime = thisGPS.timestamp;
                if( abs(thisGPSTime - timeLaserOdometry) < eps ) {
                    currGPS = thisGPS;
                    hasGPSforThisKF = true;
                    break;
                } else {
                    hasGPSforThisKF = false;
                }
                gpsBuf.pop();
            }
            mBuf.unlock();

            // Paired publish via Foxglove WebSocket (cloud → pose, same timestamp)
            if (g_wsPublisher.enabled) {
                g_wsPublisher.publishPairedFrame(thisKeyFrame, timeLaserOdometry, odom_curr);
            }

            //
            // Early reject by counting local delta movement (for equi-spereated kf drop)
            //
            odom_pose_prev = odom_pose_curr;
            odom_pose_curr = pose_curr;
            Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr); // dtf means delta_transform

            double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z); // note: absolute value. 
            translationAccumulated += delta_translation;
            rotaionAccumulated += (dtf.roll + dtf.pitch + dtf.yaw); // sum just naive approach.  

            if( translationAccumulated > keyframeMeterGap || rotaionAccumulated > keyframeRadGap ) {
                // 累计运动超过阈值时才提取关键帧，控制图优化节点密度。
                isNowKeyFrame = true;
                translationAccumulated = 0.0; // reset 
                rotaionAccumulated = 0.0; // reset 
            } else {
                isNowKeyFrame = false;
            }

            // save every frame (not just keyframes) for later full-trajectory recovery
            allFrameTimestamps.push_back(timeLaserOdometry);
            allFrameOdomPoses.push_back(pose_curr);

            if( ! isNowKeyFrame )
                continue;

            if( !gpsOffsetInitialized ) {
                if(hasGPSforThisKF) { // if the very first frame
                    gpsAltitudeInitOffset = currGPS.altitude;
                    gpsOffsetInitialized = true;
                }
            }

            //
            // Save data and Add consecutive node 
            //
            pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
            downSizeFilterScancontext.setInputCloud(thisKeyFrame);
            downSizeFilterScancontext.filter(*thisKeyFrameDS);

            const int curr_node_idx = int(keyframePoses.size()); // 即将插入的关键帧索引
            const int prev_node_idx = curr_node_idx - 1;

            mKF.lock();
            keyframeLaserClouds.push_back(thisKeyFrameDS);
            keyframeLaserCloudsFull.push_back(thisKeyFrame);
            keyframePoses.push_back(pose_curr);
            keyframePosesUpdated.push_back(pose_curr); // init
            keyframeTimes.push_back(timeLaserOdometry);

            // --- 滑动窗口管理 ---
            pcl::PointCloud<PointType>::Ptr windowSubmap(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr windowSubmapDS(new pcl::PointCloud<PointType>());

            // 1) 将当前帧降采样点云加入滑动窗口
            mDeque.lock();
            {
                pcl::PointCloud<PointType>::Ptr cloudCopy(new pcl::PointCloud<PointType>());
                pcl::copyPointCloud(*thisKeyFrameDS, *cloudCopy);
                slidingWindowDeque.push_back({curr_node_idx, cloudCopy});

                // 保持窗口大小不超过 SLIDING_WINDOW_SIZE
                while (int(slidingWindowDeque.size()) > SLIDING_WINDOW_SIZE) {
                    slidingWindowDeque.pop_front();
                }

                // 2) 构建滑动窗口子地图：逐帧变换到中心帧坐标系后合并
                const Pose6D& centerPose = keyframePosesUpdated[curr_node_idx];
                for (auto& kv : slidingWindowDeque) {
                    int kfIdx = kv.first;
                    const auto& cloud = kv.second;
                    // 先到全局、再到中心帧坐标系（确保所有点云统一到中心帧位姿下）
                    *windowSubmap += *local2center(cloud, keyframePosesUpdated[kfIdx], centerPose);
                }

                mDeque.unlock();
            }

            if (useGroundRemoval) {
                // 1) RANSAC 去地面
                pcl::PointCloud<PointType>::Ptr windowSubmapNoGround = removeGroundRANSAC(windowSubmap);

                // 2) 对去地面后子地图降采样，控制点数
                downSizeFilterScancontext.setInputCloud(windowSubmapNoGround);
                downSizeFilterScancontext.filter(*windowSubmapDS);

                // 用去地面滑动窗口子地图生成 Scan Context 描述子
                scManager.makeAndSaveScancontextAndKeys(*windowSubmapDS);
            } else {
                // 不去地面：直接对原始子地图降采样生成 SC 描述子
                downSizeFilterScancontext.setInputCloud(windowSubmap);
                downSizeFilterScancontext.filter(*windowSubmapDS);
                scManager.makeAndSaveScancontextAndKeys(*windowSubmapDS);
            }
            // --- 滑动窗口管理结束 ---

            laserCloudMapPGORedraw = true;
            mKF.unlock();
            // curr_node_idx / prev_node_idx 已在 lock 之前计算完毕，无需重复
            if( ! gtSAMgraphMade /* prior node */) {
                const int init_node_idx = 0; 
                gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3(keyframePoses.at(init_node_idx));
                // auto poseOrigin = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

                mtxPosegraph.lock();
                {
                    // 第一帧作为先验约束，给整张位姿图一个固定参考系。
                    gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, poseOrigin, priorNoise));
                    initialEstimate.insert(init_node_idx, poseOrigin);
                    // runISAM2opt();          
                }   
                mtxPosegraph.unlock();

                gtSAMgraphMade = true; 

                cout << "posegraph prior node " << init_node_idx << " added" << endl;
            } else /* consecutive node (and odom factor) after the prior added */ { // == keyframePoses.size() > 1 
                gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3(keyframePoses.at(prev_node_idx));
                gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));

                mtxPosegraph.lock();
                {
                    // 相邻关键帧之间加入里程计约束，提供局部连续性。
                    gtsam::Pose3 relPose = poseFrom.between(poseTo);
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relPose, odomNoise));

                    // 如果当前关键帧附近有 GPS，就额外加入高度约束。
                    if(hasGPSforThisKF) {
                        double curr_altitude_offseted = currGPS.altitude - gpsAltitudeInitOffset;
                        mtxRecentPose.lock();
                        gtsam::Point3 gpsConstraint(recentOptimizedX, recentOptimizedY, curr_altitude_offseted); // in this example, only adjusting altitude (for x and y, very big noises are set) 
                        mtxRecentPose.unlock();
                        gtSAMgraph.add(gtsam::GPSFactor(curr_node_idx, gpsConstraint, robustGPSNoise));
                        cout << "GPS factor added at node " << curr_node_idx << endl;
                    }
                    initialEstimate.insert(curr_node_idx, poseTo);                
                    writeEdge({prev_node_idx, curr_node_idx}, relPose, odomNoise, edges_str); // giseop
                }
                mtxPosegraph.unlock();

                if(curr_node_idx % 100 == 0)
                    cout << "posegraph odom node " << curr_node_idx << " added." << endl;
            }
            // if want to print the current graph, use gtSAMgraph.print("\nFactor Graph:\n");

            // save utility
            pgTimeSaveStream << timeLaser << std::endl; // path

            // 模拟传感器延时：让 isam/lcd 有机会在同频率下穿插运行
            if (simulatedSensorHz > 0.0) {
                auto frameDelay = std::chrono::milliseconds(
                    static_cast<int>(1000.0 / simulatedSensorHz));
                std::this_thread::sleep_for(frameDelay);
            }
        }

        // ps.
        // scan context detector is running in another thread (in constant Hz, e.g., 1 Hz)
        // pub path and point cloud in another thread

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_pg

// Scan Context 回环检测入口：发现候选回环后把索引对送入 ICP 线程。
void performSCLoopClosure(void)
{
    // 快照关键帧数量（加锁避免与 process_pg 的 push_back 数据竞争）
    mKF.lock();
    int kf_size = (int)keyframePoses.size();
    mKF.unlock();

    if( kf_size < scManager.NUM_EXCLUDE_RECENT) // do not try too early
        return;

    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff
    int SCclosestHistoryFrameID = detectResult.first;
    if( SCclosestHistoryFrameID != -1 ) {
        const int prev_node_idx = SCclosestHistoryFrameID;
        const int curr_node_idx = kf_size - 1; // because cpp starts 0 and ends n-1

        mBuf.lock();
        // 去重：已成功处理过的回环对不再重复入队
        if (processedLoopPairs.count({prev_node_idx, curr_node_idx})) {
            mBuf.unlock();
            return;
        }
        mBuf.unlock();

        // ---- 世界系距离预检：拒绝相距过远的关键帧对 ----
        // 用 ISAM2 优化后的位姿（keyframePosesUpdated）计算世界系欧氏距离，
        // 防止 SC 描述子误匹配导致的远距离假阳性回环。
        {
            mKF.lock();
            if (prev_node_idx < (int)keyframePosesUpdated.size() &&
                curr_node_idx < (int)keyframePosesUpdated.size()) {
                double dx = keyframePosesUpdated[curr_node_idx].x - keyframePosesUpdated[prev_node_idx].x;
                double dy = keyframePosesUpdated[curr_node_idx].y - keyframePosesUpdated[prev_node_idx].y;
                double dz = keyframePosesUpdated[curr_node_idx].z - keyframePosesUpdated[prev_node_idx].z;
                double worldDist = std::sqrt(dx*dx + dy*dy + dz*dz);
                mKF.unlock();

                if (worldDist > scLoopMaxWorldDistance) {
                    cout << "[SC Loop] Reject: world distance " << worldDist
                         << "m > " << scLoopMaxWorldDistance
                         << "m (SC matched " << prev_node_idx << " ↔ " << curr_node_idx << ")"
                         << endl;
                    return;
                }
            } else {
                mKF.unlock();
            }
        }

        mBuf.lock();
        scLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
        mBuf.unlock();

        cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;
    }
} // performSCLoopClosure

// 空间近邻回环检测：利用里程计位姿的欧氏距离发现回环，绕过 SC，直接 ICP 验证。
// SC 可能因视角变化漏掉回环，但空间近邻不会——里程计精度尚可就有效。
void performSpatialLoopClosure(void)
{
    if (!useSpatialLoopClosure) return;

    // 快照关键帧数量（加锁避免与 process_pg 的 push_back 数据竞争）
    mKF.lock();
    int kf_size = (int)keyframePoses.size();
    mKF.unlock();

    int curr_node_idx = kf_size - 1;
    if (curr_node_idx < spatialLoopMinSeparation) return;

    double radiusSq = spatialLoopRadius * spatialLoopRadius;
    std::vector<std::pair<int, double>> neighbors; // (idx, distSq)

    mKF.lock();
    double cx = keyframePosesUpdated[curr_node_idx].x;
    double cy = keyframePosesUpdated[curr_node_idx].y;
    double cz = keyframePosesUpdated[curr_node_idx].z;
    int searchEnd = curr_node_idx - spatialLoopMinSeparation;
    for (int i = 0; i < searchEnd; i++) {
        double dx = keyframePosesUpdated[i].x - cx;
        double dy = keyframePosesUpdated[i].y - cy;
        double dz = keyframePosesUpdated[i].z - cz;
        double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < radiusSq)
            neighbors.push_back({i, d2});
    }
    mKF.unlock();

    // 按距离排序（近的优先处理）
    std::sort(neighbors.begin(), neighbors.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    int acceptedCount = 0;
    const int MAX_SPATIAL_LOOPS_PER_CYCLE = 3; // 每次最多接受 3 个空间回环

    for (const auto& [historyIdx, d2] : neighbors) {
        if (acceptedCount >= MAX_SPATIAL_LOOPS_PER_CYCLE) break;

        std::pair<int, int> loopPair(historyIdx, curr_node_idx);

        // 去重
        mBuf.lock();
        bool alreadyDone = processedLoopPairs.count(loopPair) > 0;
        mBuf.unlock();
        if (alreadyDone) continue;

        cout << "[Spatial Loop] Candidate: " << historyIdx << " ↔ " << curr_node_idx
             << " (dist=" << sqrt(d2) << "m)" << endl;

        // 直接跑 ICP，使用比 SC 回环更严格的 fitness 阈值
        auto relative_pose_optional = doICPVirtualRelative(historyIdx, curr_node_idx, spatialLoopFitnessThres);
        if (relative_pose_optional) {
            gtsam::Pose3 relative_pose = relative_pose_optional.value();
            mtxPosegraph.lock();
            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(historyIdx, curr_node_idx, relative_pose, robustLoopNoise));
            writeEdge({historyIdx, curr_node_idx}, relative_pose, robustLoopNoise, edges_str);
            mtxPosegraph.unlock();

            mBuf.lock();
            processedLoopPairs.insert(loopPair);
            mBuf.unlock();

            cout << "[Spatial Loop] Accepted: " << historyIdx << " ↔ " << curr_node_idx << endl;
            acceptedCount++;
        }
    }
} // performSpatialLoopClosure

// 回环检测线程：固定频率运行 Scan Context + 空间近邻回环检测。
void process_lcd(void)
{
    float loopClosureFrequency = 1.0; // can change
    auto cycleTime = std::chrono::duration<double>(1.0 / loopClosureFrequency);
    while (!g_shutdown_requested.load())
    {
        // 响应式 sleep：拆成 100ms 片段，每片段检查退出标志
        auto deadline = std::chrono::steady_clock::now() + cycleTime;
        while (!g_shutdown_requested.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (g_shutdown_requested.load()) break;
        performSCLoopClosure();
        performSpatialLoopClosure();
    }
} // process_lcd

// 回环 ICP 线程：把候选回环转换成精确的位姿约束。
void process_icp(void)
{
    while(!g_shutdown_requested.load())
    {
		while ( !g_shutdown_requested.load() && !scLoopICPBuf.empty() )
        {
            if( scLoopICPBuf.size() > 30 ) {
                std::cout << "[WARN] Too many loop closure candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)" << std::endl;
            }

            mBuf.lock();
            std::pair<int, int> loop_idx_pair = scLoopICPBuf.front();
            scLoopICPBuf.pop();
            mBuf.unlock();

            const int prev_node_idx = loop_idx_pair.first;
            const int curr_node_idx = loop_idx_pair.second;

            // 二次去重：防止竞态下同一对重复入队
            mBuf.lock();
            bool alreadyProcessed = processedLoopPairs.count(loop_idx_pair) > 0;
            mBuf.unlock();
            if (alreadyProcessed) continue;

            auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx);
            if(relative_pose_optional) {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                mtxPosegraph.lock();
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relative_pose, robustLoopNoise));
                writeEdge({prev_node_idx, curr_node_idx}, relative_pose, robustLoopNoise, edges_str); // giseop
                mtxPosegraph.unlock();

                // 记录已处理，防止后续重复检测
                mBuf.lock();
                processedLoopPairs.insert(loop_idx_pair);
                mBuf.unlock();
            }
        }

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_icp

// 定时运行 iSAM2，并把最新优化结果导出到磁盘。
void process_isam(void)
{
    float hz = 1;
    auto cycleTime = std::chrono::duration<double>(1.0 / hz);
    while (!g_shutdown_requested.load()) {
        // 响应式 sleep：拆成 100ms 片段
        auto deadline = std::chrono::steady_clock::now() + cycleTime;
        while (!g_shutdown_requested.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (g_shutdown_requested.load()) break;
        if( gtSAMgraphMade ) {
            mtxPosegraph.lock();
            runISAM2opt();
            cout << "running isam2 optimization ..." << endl;
            mtxPosegraph.unlock();

            mKF.lock();
            saveOptimizedVerticesTUMformat(isamCurrentEstimate, keyframeTimes, pgTUMformat); // pose
            saveOdometryVerticesKITTIformat(odomKITTIformat); // pose
            saveGTSAMgraphG2oFormat(isamCurrentEstimate);
            mKF.unlock();

            // Publish global point cloud map after each PGO optimization
            if (g_wsPublisher.enabled) {
                pcl::PointCloud<PointType>::Ptr globalMap(new pcl::PointCloud<PointType>());
                mKF.lock();
                for (size_t i = 0; i < keyframeLaserCloudsFull.size() && i < keyframePosesUpdated.size(); i++) {
                    *globalMap += *local2global(keyframeLaserCloudsFull[i], keyframePosesUpdated[i]);
                }
                mKF.unlock();
                double ts = keyframeTimes.empty() ? 0.0 : keyframeTimes.back();
                g_wsPublisher.publishGlobalMap(globalMap, ts);
            }
        }
    }
}

// pubMap 和 process_viz_map 已移除 —— 去 ROS 化后不再需要 ROS 地图可视化发布。
// 最终地图通过 saveGlobalMap() 直接保存为 PCD 文件。

// 将所有关键帧点云按最终优化位姿拼接、滤波后保存成 PCD 全局地图。
void saveGlobalMap(const std::string& _filename)
{
    pcl::PointCloud<PointType>::Ptr globalMap(new pcl::PointCloud<PointType>());

    mKF.lock();
    for (size_t i = 0; i < keyframeLaserCloudsFull.size() && i < keyframePosesUpdated.size(); i++) {
        *globalMap += *local2global(keyframeLaserCloudsFull[i], keyframePosesUpdated[i]);
    }
    mKF.unlock();

    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(0.05f, 0.05f, 0.05f);
    pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
    voxel.setInputCloud(globalMap);
    voxel.filter(*filtered);

    pcl::io::savePCDFileBinary(_filename, *filtered);
    cout << "Global map saved: " << _filename << " (" << filtered->size() << " points)" << endl;
} // saveGlobalMap

// 将每个关键帧的点云、位姿与元数据保存为与 interactive_slam 兼容的目录结构。
void saveKeyframes(void)
{
    if (!saveKeyframesEnabled || saveKeyframesDirectory.empty()) return;

    std::string kf_root = saveKeyframesDirectory + "/keyframes/";
    boost::filesystem::create_directories(kf_root);

    mKF.lock();
    size_t n = std::min({keyframeLaserClouds.size(),
                         keyframePosesUpdated.size(),
                         keyframeTimes.size()});

    for (size_t i = 0; i < n; i++) {
        std::string kf_dir = (boost::format("%s/%06d") % kf_root % i).str();
        boost::filesystem::create_directories(kf_dir);

        // cloud.pcd (降采样)
        pcl::io::savePCDFileBinary(kf_dir + "/cloud.pcd", *keyframeLaserClouds[i]);

        // raw.pcd (全分辨率，可选)
        if (saveKeyframesFullCloud && i < keyframeLaserCloudsFull.size()) {
            pcl::io::savePCDFileBinary(kf_dir + "/raw.pcd", *keyframeLaserCloudsFull[i]);
        }

        // data 元数据
        std::ofstream ofs(kf_dir + "/data");
        if (!ofs) continue;

        // 时间戳：double seconds → sec + usec
        double t = keyframeTimes[i];
        unsigned long sec  = static_cast<unsigned long>(std::floor(t));
        unsigned long usec = static_cast<unsigned long>((t - sec) * 1e6);

        ofs << "stamp " << sec << " " << usec << std::endl;

        // estimate + odom: 统一使用优化后位姿
        const Pose6D& pose = keyframePosesUpdated[i];
        Eigen::Affine3f T = pcl::getTransformation(pose.x, pose.y, pose.z,
                                                   pose.roll, pose.pitch, pose.yaw);
        ofs << "estimate" << std::endl << T.matrix() << std::endl;
        ofs << "odom " << std::endl << T.matrix() << std::endl;

        ofs << "id " << i << std::endl;
    }
    mKF.unlock();

    cout << "Keyframes saved: " << n << " nodes → " << kf_root << endl;
} // saveKeyframes

void recoverAllPosesTUM(const std::string& _filename)
{
    if (allFrameTimestamps.empty() || keyframePosesUpdated.empty()) return;

    // 先把每个关键帧映射回原始帧序号，后面才能把优化后的位姿传播到每一帧。
    std::vector<int> kfAllIdx(keyframeTimes.size(), -1);
    for (size_t k = 0; k < keyframeTimes.size(); k++) {
        for (size_t a = 0; a < allFrameTimestamps.size(); a++) {
            if (std::abs(allFrameTimestamps[a] - keyframeTimes[k]) < 1e-9) {
                kfAllIdx[k] = (int)a;
                break;
            }
        }
    }

    std::fstream stream(_filename.c_str(), std::fstream::out);
    stream << std::fixed << std::setprecision(6);

    int lastKfKfIdx = 0;
    for (size_t i = 0; i < allFrameTimestamps.size(); i++) {
        // 找到当前原始帧对应的最近一个关键帧。
        int kfKfIdx = lastKfKfIdx;
        for (size_t k = (size_t)lastKfKfIdx + 1; k < keyframeTimes.size(); k++) {
            if (kfAllIdx[k] >= 0 && kfAllIdx[k] <= (int)i)
                kfKfIdx = (int)k;
            else
                break;
        }
        lastKfKfIdx = kfKfIdx;
        if (kfKfIdx >= (int)keyframePosesUpdated.size()) continue;

        // 从该关键帧的优化结果出发，再用关键帧后面的原始里程计增量恢复当前帧位姿。
        Pose6D& kfPose = keyframePosesUpdated[kfKfIdx];
        Eigen::Affine3f result = pcl::getTransformation(
            kfPose.x, kfPose.y, kfPose.z,
            kfPose.roll, kfPose.pitch, kfPose.yaw);

        int kfAnchor = kfAllIdx[kfKfIdx];

        // 把关键帧到当前帧之间的里程计增量逐步串起来，补回非关键帧轨迹。
        if ((int)i > kfAnchor && kfAnchor >= 0) {
            for (int j = kfAnchor + 1; j <= (int)i; j++) {
                Eigen::Affine3f prevOdom = pcl::getTransformation(
                    allFrameOdomPoses[j-1].x, allFrameOdomPoses[j-1].y, allFrameOdomPoses[j-1].z,
                    allFrameOdomPoses[j-1].roll, allFrameOdomPoses[j-1].pitch, allFrameOdomPoses[j-1].yaw);
                Eigen::Affine3f currOdom = pcl::getTransformation(
                    allFrameOdomPoses[j].x, allFrameOdomPoses[j].y, allFrameOdomPoses[j].z,
                    allFrameOdomPoses[j].roll, allFrameOdomPoses[j].pitch, allFrameOdomPoses[j].yaw);
                Eigen::Matrix4f delta = prevOdom.matrix().inverse() * currOdom.matrix();
                result.matrix() = result.matrix() * delta;
            }
        }

        Eigen::Vector3f t = result.translation();
        Eigen::Quaternionf q(result.rotation());

        stream << allFrameTimestamps[i] << " "
               << t.x() << " " << t.y() << " " << t.z() << " "
               << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }

    stream.close();
    cout << "All poses recovered: " << _filename << " (" << allFrameTimestamps.size() << " frames)" << endl;
} // recoverAllPosesTUM


// 程序入口：加载 YAML 配置 → 读入数据 → 启动处理线程 → 等待完成 → 保存结果。
int main(int argc, char **argv)
{
    signal(SIGINT,  gracefulShutdownHandler);
    signal(SIGTERM, gracefulShutdownHandler);

    // -------------------- 1. 加载 YAML 配置 --------------------
    std::string configPath = (argc >= 2) ? argv[1] : "config/default.yaml";
    std::cout << "[INFO] Loading config: " << configPath << std::endl;

    YAML::Node cfg;
    try {
        cfg = YAML::LoadFile(configPath);
        if (!cfg.IsMap()) {
            std::cerr << "[FATAL] Config file parsed but root is not a YAML map. "
                      << "All parameters will use defaults." << std::endl;
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "[FATAL] YAML parse error: " << e.what() << std::endl;
        std::cerr << "[FATAL] Please fix the YAML syntax in: " << configPath << std::endl;
        return 1;
    }

    // 输出路径
    save_directory              = getParamOrDefaultDeep<std::string>(cfg, "output.save_directory", "output/");
    saveKeyframesEnabled        = getParamOrDefaultDeep<bool>(cfg, "output.save_keyframes", false);
    saveKeyframesDirectory      = getParamOrDefaultDeep<std::string>(cfg, "output.save_keyframes_directory", "");
    saveKeyframesFullCloud      = getParamOrDefaultDeep<bool>(cfg, "output.save_keyframes_full_cloud", true);

    // 确保输出目录以 / 结尾
    if (!save_directory.empty() && save_directory.back() != '/') save_directory += '/';

    pgTUMformat = save_directory + "optimized_poses.txt";
    odomKITTIformat = save_directory + "odom_poses.txt";

    pgTimeSaveStream = std::fstream(save_directory + "times.txt", std::fstream::out);
    pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);

    // 算法参数
    keyframeMeterGap  = getParamOrDefaultDeep<double>(cfg, "keyframe.meter_gap", 2.0);
    keyframeDegGap    = getParamOrDefaultDeep<double>(cfg, "keyframe.deg_gap", 10.0);
    keyframeRadGap    = deg2rad(keyframeDegGap);

    scDistThres       = getParamOrDefaultDeep<double>(cfg, "scan_context.dist_thres", 0.2);
    scMaximumRadius   = getParamOrDefaultDeep<double>(cfg, "scan_context.max_radius", 80.0);
    scManager.LIDAR_HEIGHT = getParamOrDefaultDeep<double>(cfg, "scan_context.lidar_height", 2.0);
    scLoopMaxWorldDistance = getParamOrDefaultDeep<double>(cfg, "scan_context.max_world_distance", 30.0);

    useGroundRemoval        = getParamOrDefault<bool>(cfg, "use_ground_removal", true);
    useICPSubmapEnhancement = getParamOrDefault<bool>(cfg, "use_icp_submap_enhancement", true);

    icpMaxCorrespondenceDistance = getParamOrDefaultDeep<double>(cfg, "icp.max_correspondence_distance", 150.0);
    icpFitnessScoreThreshold     = getParamOrDefaultDeep<double>(cfg, "icp.fitness_score_threshold", 0.3);
    icpMinSourcePoints           = getParamOrDefaultDeep<int>(cfg, "icp.min_source_points", 50);
    icpMinTargetPoints           = getParamOrDefaultDeep<int>(cfg, "icp.min_target_points", 50);
    icpMaxTranslation            = getParamOrDefaultDeep<double>(cfg, "icp.max_translation", 50.0);
    double icpMaxRotationDeg     = getParamOrDefaultDeep<double>(cfg, "icp.max_rotation_deg", 30.0);
    icpMaxRotationRad            = deg2rad(icpMaxRotationDeg);

    loopNoiseScore  = getParamOrDefaultDeep<double>(cfg, "loop.noise_score", 0.5);
    loopKernelParam = getParamOrDefaultDeep<double>(cfg, "loop.kernel_param", 1.0);
    loopKernelType  = getParamOrDefaultDeep<std::string>(cfg, "loop.kernel_type", std::string("geman_mcclure"));

    useSpatialLoopClosure    = getParamOrDefaultDeep<bool>(cfg, "spatial_loop.enabled", true);
    spatialLoopRadius        = getParamOrDefaultDeep<double>(cfg, "spatial_loop.radius", 10.0);
    spatialLoopFitnessThres  = getParamOrDefaultDeep<double>(cfg, "spatial_loop.fitness_thres", 0.1);
    spatialLoopMinSeparation = getParamOrDefaultDeep<int>(cfg, "spatial_loop.min_separation", 50);

    double mapVizFilterSize  = getParamOrDefault<double>(cfg, "mapviz_filter_size", 0.4);
    double inputSilenceTimeout = getParamOrDefault<double>(cfg, "input_silence_timeout", 8.0);
    simulatedSensorHz = getParamOrDefault<double>(cfg, "simulated_sensor_hz", 10.0);

    useGPS = getParamOrDefaultDeep<bool>(cfg, "input.use_gps", false);

    // 初始化优化器和噪声
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    initNoises();

    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    // 下采样滤波器
    float sc_filter_size = 0.05;
    float icp_filter_size = 0.4;
    downSizeFilterScancontext.setLeafSize(sc_filter_size, sc_filter_size, sc_filter_size);
    downSizeFilterICP.setLeafSize(icp_filter_size, icp_filter_size, icp_filter_size);
    downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);

    // -------------------- 2. 读取输入数据到 deque --------------------
    std::string pcdDir   = getParamOrDefaultDeep<std::string>(cfg, "input.pcd_dir", "all_pcd_body/");
    std::string tumPath  = getParamOrDefaultDeep<std::string>(cfg, "input.tum_poses", "all_pcd_body/lidar_poses.txt");
    std::string gpsPath  = getParamOrDefaultDeep<std::string>(cfg, "input.gps_file", "");

    auto pcdFiles = scanPcdDirectory(pcdDir);
    if (pcdFiles.empty()) {
        std::cerr << "[ERROR] No PCD files found in " << pcdDir << std::endl;
        return 1;
    }

    auto tumPoses = loadTumPoses(tumPath);

    // 加载 GPS（可选）
    if (useGPS && !gpsPath.empty()) {
        std::ifstream gpsFile(gpsPath);
        if (gpsFile) {
            std::string line;
            while (std::getline(gpsFile, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                GpsData gps;
                if (iss >> gps.timestamp >> gps.latitude >> gps.longitude >> gps.altitude) {
                    gpsBuf.push(gps);
                }
            }
            std::cout << "[INFO] Loaded GPS data from " << gpsPath << std::endl;
        }
    }

    // 加载 PCD 点云并与 TUM 位姿匹配（两个数据源已按时间戳排序，顺序对齐）
    int loadedCount = 0;
    for (auto& [ts, pcdPath] : pcdFiles) {
        if (tumPoses.empty()) {
            std::cerr << "[WARN] No more TUM poses, skipping remaining PCDs" << std::endl;
            break;
        }

        auto cloud = std::make_shared<pcl::PointCloud<PointType>>();
        if (pcl::io::loadPCDFile<PointType>(pcdPath, *cloud) == -1) {
            std::cerr << "[WARN] Failed to load " << pcdPath << std::endl;
            tumPoses.pop_front();  // 跳过此帧位姿以保持对齐
            continue;
        }

        odometryBuf.push(tumPoses.front());
        tumPoses.pop_front();
        fullResBuf.push(cloud);
        loadedCount++;
    }
    std::cout << "[INFO] Loaded " << loadedCount << " frames (PCD + poses)" << std::endl;

    if (loadedCount == 0) {
        std::cerr << "[ERROR] No frames loaded. Check config paths and file format." << std::endl;
        return 1;
    }

    g_has_received_input.store(true);
    g_last_input_time.store(
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count());

    // -------------------- Foxglove WebSocket init --------------------
    {
        bool wsEnabled = getParamOrDefaultDeep<bool>(cfg, "websocket.enabled", true);
        if (wsEnabled) {
            std::string wsHost = getParamOrDefaultDeep<std::string>(cfg, "websocket.host",
                                                                   std::string("0.0.0.0"));
            uint16_t wsPort = static_cast<uint16_t>(
                getParamOrDefaultDeep<int>(cfg, "websocket.port", 8765));
            bool wsPubPC  = getParamOrDefaultDeep<bool>(cfg, "websocket.publish_pointcloud", true);
            bool wsPubPose = getParamOrDefaultDeep<bool>(cfg, "websocket.publish_pose", true);
            double wsLeaf = getParamOrDefaultDeep<double>(cfg, "websocket.pointcloud_downsample_leaf", 0.1);
            int wsMaxPts  = getParamOrDefaultDeep<int>(cfg, "websocket.max_points_per_message", 50000);
            g_wsPublisher.publishICPDetail   = getParamOrDefaultDeep<bool>(cfg, "websocket.publish_icp_detail", true);
            g_wsPublisher.publishGlobalMapFlag = getParamOrDefaultDeep<bool>(cfg, "websocket.publish_global_map", true);
            g_wsPublisher.globalMapLeafSize  = getParamOrDefaultDeep<double>(cfg, "websocket.global_map_leaf_size", 0.2);
            g_wsPublisher.init(wsHost, wsPort, wsPubPC, wsPubPose, wsLeaf, wsMaxPts);
        }
    }

    // -------------------- 3. 启动后台处理线程 --------------------
    std::thread posegraph_slam {process_pg};
    std::thread lc_detection  {process_lcd};
    std::thread icp_calculation {process_icp};
    std::thread isam_update   {process_isam};

    // -------------------- 4. 主循环：等待处理完成 --------------------
    // 数据已全部入队，主线程等待队列被消费完 + 输入静默超时 + Ctrl+C
    while (!g_shutdown_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // 检查是否所有数据都已处理完
        bool allDone = true;
        mBuf.lock();
        allDone = odometryBuf.empty() && fullResBuf.empty();
        mBuf.unlock();

        if (allDone && inputSilenceTimeout > 0.0 && g_has_received_input.load()) {
            double now = std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            double elapsed = now - g_last_input_time.load();
            if (elapsed > inputSilenceTimeout) {
                std::cout << "[INFO] All frames processed and silence timeout reached (" << elapsed
                          << "s > " << inputSilenceTimeout << "s). Auto-saving..." << std::endl;
                g_shutdown_requested.store(true);
            }
        }
    }

    std::cout << "[INFO] Shutdown triggered. Stopping processing threads..." << std::endl;

    posegraph_slam.join();
    lc_detection.join();
    icp_calculation.join();
    isam_update.join();

    // Shutdown WebSocket before saving results
    g_wsPublisher.shutdown();

    // -------------------- 5. 保存最终结果 --------------------
    std::cout << "[INFO] All threads stopped. Saving final results -- DO NOT INTERRUPT..." << std::endl;
    std::cout << "[INFO]   [1/6] Saving optimized poses (TUM)..." << std::endl;
    saveOptimizedVerticesTUMformat(isamCurrentEstimate, keyframeTimes, pgTUMformat);
    std::cout << "[INFO]   [2/6] Saving odometry poses (KITTI)..." << std::endl;
    saveOdometryVerticesKITTIformat(odomKITTIformat);
    std::cout << "[INFO]   [3/6] Saving pose graph (g2o)..." << std::endl;
    saveGTSAMgraphG2oFormat(isamCurrentEstimate);
    std::cout << "[INFO]   [4/6] Saving global map (PCD)..." << std::endl;
    saveGlobalMap(save_directory + "global_map.pcd");
    std::cout << "[INFO]   [5/6] Saving keyframes..." << std::endl;
    saveKeyframes();
    std::cout << "[INFO]   [6/6] Recovering all poses (TUM)..." << std::endl;
    recoverAllPosesTUM(save_directory + "all_optimized_poses.txt");

    pgTimeSaveStream.close();

    std::cout << "[INFO] All saves complete. Exiting." << std::endl;
    return 0;
}
