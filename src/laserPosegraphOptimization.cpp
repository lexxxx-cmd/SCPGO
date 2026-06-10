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
#include <optional>
#include <iomanip>

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
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <eigen3/Eigen/Dense>

// Workaround for Ceres < 2.1 + Eigen >= 3.4 incompatibility.
// Eigen 3.4 removed the ScalarBinaryOpTraits class template that older Ceres
// tries to specialize in jet.h. Provide a minimal primary template so the
// Ceres partial specializations compile.
#if EIGEN_VERSION_AT_LEAST(3,4,0)
namespace Eigen {
template <typename BinaryOp, typename LhsScalar, typename RhsScalar>
struct ScalarBinaryOpTraits {
  enum { Defined = 0 };
};
}  // namespace Eigen
#endif

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

// ------------------------- 输入缓存：回调只负责入队 -------------------------
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<sensor_msgs::NavSatFix::ConstPtr> gpsBuf;
std::queue<std::pair<int, int> > scLoopICPBuf;
std::set<std::pair<int, int>> processedLoopPairs; // 已处理的回环对，防止重复添加

std::mutex mBuf;
std::mutex mKF;

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
std::mutex mtxPosegraph;
std::mutex mtxRecentPose;

// ------------------------- 可视化地图与 GPS 相关状态 -------------------------
pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType>());
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;
bool laserCloudMapPGORedraw = true;

bool useGPS = true;
// bool useGPS = false;
sensor_msgs::NavSatFix::ConstPtr currGPS;
bool hasGPSforThisKF = false;
bool gpsOffsetInitialized = false; 
double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX = 0.0;
double recentOptimizedY = 0.0;

// ------------------------- ROS 发布器与导出文件 -------------------------
ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO;
ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal;
ros::Publisher pubLoopScanIcp, pubLoopSubmapIcp;
ros::Publisher pubWindowSubmapSC, pubWindowSubmapNoGround;
ros::Publisher pubOdomRepubVerifier;

std::string save_directory;
std::string pgTUMformat;
std::string odomKITTIformat;
std::fstream pgG2oSaveStream, pgTimeSaveStream;

std::vector<std::string> edges_str; // used in writeEdge

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
void writeEdge(const std::pair<int, int> _node_idx_pair, const gtsam::Pose3& _relPose, std::vector<std::string>& edges_str)
{
    gtsam::Point3 t = _relPose.translation();
    gtsam::Rot3 R = _relPose.rotation();

    std::string curEdgeInfo {
        "EDGE_SE3:QUAT " + std::to_string(_node_idx_pair.first) + " " + std::to_string(_node_idx_pair.second) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

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

    pgG2oSaveStream = std::fstream(save_directory + "singlesession_posegraph.g2o", std::fstream::out);

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

// 里程计回调只做缓存，不在回调线程里做耗时计算。
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(_laserOdometry);
	mBuf.unlock();
} // laserOdometryHandler

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &_laserCloudFullRes)
// 点云回调同样只入队，真正的处理放到工作线程里。
{
	mBuf.lock();
	fullResBuf.push(_laserCloudFullRes);
	mBuf.unlock();
} // laserCloudFullResHandler

void gpsHandler(const sensor_msgs::NavSatFix::ConstPtr &_gps)
// GPS 回调：只有启用 GPS 时才缓存到队列。
{
    if(useGPS) {
        mBuf.lock();
        gpsBuf.push(_gps);
        mBuf.unlock();
    }
} // gpsHandler

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

    double loopNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );

    double bigNoiseTolerentToXY = 1000000000.0; // 1e9
    double gpsAltitudeNoiseScore = 250.0; // if height is misaligned after loop clsosing, use this value bigger
    gtsam::Vector robustNoiseVector3(3); // gps factor has 3 elements (xyz)
    // 这里只重点约束 GPS 的高度分量，X/Y 给足够大的噪声，避免平面位置把轨迹拉偏。
    robustNoiseVector3 << bigNoiseTolerentToXY, bigNoiseTolerentToXY, gpsAltitudeNoiseScore; // means only caring altitude here. (because LOAM-like-methods tends to be asymptotically flyging)
    robustGPSNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3) );

} // initNoises
// 从里程计消息里提取平移和 RPY 姿态。
Pose6D getOdom(nav_msgs::Odometry::ConstPtr _odom)
{
    auto tx = _odom->pose.pose.position.x;
    auto ty = _odom->pose.pose.position.y;
    auto tz = _odom->pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw}; 
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
        cloudOut->points[i].intensity = pointFrom.intensity;
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
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

// 发布优化后的最后一帧位姿和整条路径，同时广播 TF。
void pubPath( void )
{
    // pub odom and path 
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path pathAftPGO;
    pathAftPGO.header.frame_id = "camera_init";
    mKF.lock(); 
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()) - 1; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    for (int node_idx=0; node_idx < recentIdxUpdated.load(); node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    {
        const Pose6D& pose_est = keyframePosesUpdated.at(node_idx); // upodated poses
        // const gtsam::Pose3& pose_est = isamCurrentEstimate.at<gtsam::Pose3>(node_idx);

        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id = "camera_init";
        odomAftPGOthis.child_frame_id = "aft_pgo";
        odomAftPGOthis.header.stamp = ros::Time().fromSec(keyframeTimes.at(node_idx));
        odomAftPGOthis.pose.pose.position.x = pose_est.x;
        odomAftPGOthis.pose.pose.position.y = pose_est.y;
        odomAftPGOthis.pose.pose.position.z = pose_est.z;
        odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
        odomAftPGO = odomAftPGOthis;

        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

        pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
        pathAftPGO.header.frame_id = "camera_init";
        pathAftPGO.poses.push_back(poseStampAftPGO);
    }
    mKF.unlock(); 
    pubOdomAftPGO.publish(odomAftPGO); // last pose 
    pubPathAftPGO.publish(pathAftPGO); // poses 

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
    q.setW(odomAftPGO.pose.pose.orientation.w);
    q.setX(odomAftPGO.pose.pose.orientation.x);
    q.setY(odomAftPGO.pose.pose.orientation.y);
    q.setZ(odomAftPGO.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "camera_init", "aft_pgo"));
} // pubPath

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
        cloudOut->points[i].intensity = pointFrom->intensity;
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
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
} // loopFindNearKeyframesCloud


// 用 ICP 估计回环两端的精确相对位姿；失败则返回空值。
std::optional<gtsam::Pose3> doICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx )
{
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
            downSizeFilterICP.setInputCloud(cureKeyframeCloud);
            downSizeFilterICP.filter(*cloud_temp);
            *cureKeyframeCloud = *cloud_temp;
        }

        // ---- 历史帧子地图：时间窗口 ±25 + 空间近邻（15m 半径） ----
        loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx);

        // 空间近邻：暴力搜索 15m 半径内的历史关键帧，合并到 target submap
        {
            const double spatialRadius = 15.0;
            const double spatialRadiusSq = spatialRadius * spatialRadius;
            Pose6D& loopPose = keyframePosesUpdated[_loop_kf_idx];
            for (int i = 0; i < int(keyframeLaserClouds.size()); ++i) {
                // 跳过已在时间窗口内的帧，避免重复合并
                if (std::abs(i - _loop_kf_idx) <= historyKeyframeSearchNum)
                    continue;
                Pose6D& pose_i = keyframePosesUpdated[i];
                double dx = pose_i.x - loopPose.x;
                double dy = pose_i.y - loopPose.y;
                double dz = pose_i.z - loopPose.z;
                double distSq = dx*dx + dy*dy + dz*dz;
                if (distSq < spatialRadiusSq) {
                    mKF.lock();
                    *targetKeyframeCloud += *local2global(keyframeLaserClouds[i], keyframePosesUpdated[i]);
                    mKF.unlock();
                }
            }
        }
        // 对合并空间近邻后的 target 再次降采样
        {
            pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
            downSizeFilterICP.setInputCloud(targetKeyframeCloud);
            downSizeFilterICP.filter(*cloud_temp);
            *targetKeyframeCloud = *cloud_temp;
        }
    } else {
        // 原始逻辑：当前帧仅自己，历史帧仅时间窗口
        loopFindNearKeyframesCloud(cureKeyframeCloud, _curr_kf_idx, 0, _loop_kf_idx);
        loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx);
    }

    // loop verification
    sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
    pcl::toROSMsg(*cureKeyframeCloud, cureKeyframeCloudMsg);
    cureKeyframeCloudMsg.header.frame_id = "camera_init";
    pubLoopScanLocal.publish(cureKeyframeCloudMsg);

    sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
    pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
    targetKeyframeCloudMsg.header.frame_id = "camera_init";
    pubLoopSubmapLocal.publish(targetKeyframeCloudMsg);

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align pointclouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    float loopFitnessScoreThreshold = 0.3; // user parameter but fixed low value is safe.
    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
        std::cout << "[SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop." << std::endl;
        return std::nullopt;
    } else {
        std::cout << "[SC loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this SC loop." << std::endl;

        // 发布经过 ICP 验证的成功匹配回环关键帧（ICP 配准后位置）和历史回环子地图
        sensor_msgs::PointCloud2 loopScanIcpMsg;
        pcl::toROSMsg(*unused_result, loopScanIcpMsg);
        loopScanIcpMsg.header.frame_id = "camera_init";
        pubLoopScanIcp.publish(loopScanIcpMsg);

        sensor_msgs::PointCloud2 loopSubmapIcpMsg;
        pcl::toROSMsg(*targetKeyframeCloud, loopSubmapIcpMsg);
        loopSubmapIcpMsg.header.frame_id = "camera_init";
        pubLoopSubmapIcp.publish(loopSubmapIcpMsg);
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);

    // After submap fix (each keyframe uses its own pose in local2global),
    // both clouds are at their true global positions. T_icp is a global correction.
    // Compute the real relative pose: P_loop.between(T_icp * P_curr)
    // 先把当前帧通过 ICP 校正到全局系，再和历史帧做 between，得到真正可加入图优化的相对约束。
    gtsam::Pose3 poseLoop = Pose6DtoGTSAMPose3(keyframePosesUpdated[_loop_kf_idx]);
    gtsam::Pose3 poseCurr = Pose6DtoGTSAMPose3(keyframePosesUpdated[_curr_kf_idx]);
    gtsam::Pose3 T_icp = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseCurrCorrected = T_icp.compose(poseCurr);

    return poseLoop.between(poseCurrCorrected);
} // doICPVirtualRelative

// 主线程之一：消费里程计/点云/GPS 缓存，抽关键帧并构建初始位姿图。
void process_pg()
{
    while(ros::ok())
    {
		while ( ros::ok() && !odometryBuf.empty() && !fullResBuf.empty() )
        {
            //
            // pop and check keyframe is or not  
            // 
			mBuf.lock();       
            while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
                odometryBuf.pop();
            if (odometryBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            // 这里要求里程计和点云时间尽量对齐；先丢掉更早的里程计帧，避免错配。
            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
            timeLaser = fullResBuf.front()->header.stamp.toSec();
            // TODO

            laserCloudFullRes->clear();
            pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*fullResBuf.front(), *thisKeyFrame);
            fullResBuf.pop();

            Pose6D pose_curr = getOdom(odometryBuf.front());
            odometryBuf.pop();

            // find nearest gps 
            double eps = 0.1; // find a gps topioc arrived within eps second 
            while (!gpsBuf.empty()) {
                auto thisGPS = gpsBuf.front();
                auto thisGPSTime = thisGPS->header.stamp.toSec();
                if( abs(thisGPSTime - timeLaserOdometry) < eps ) {
                    // 找到与当前关键帧时间最接近的 GPS 数据，就把它绑定到这一帧。
                    currGPS = thisGPS;
                    hasGPSforThisKF = true; 
                    break;
                } else {
                    hasGPSforThisKF = false;
                }
                gpsBuf.pop();
            }
            mBuf.unlock(); 

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
                    gpsAltitudeInitOffset = currGPS->altitude;
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
                // 3) 发布原始合并子地图（去地面前，用于对比查看）
                {
                    sensor_msgs::PointCloud2 rawMsg;
                    pcl::toROSMsg(*windowSubmap, rawMsg);
                    rawMsg.header.frame_id = "camera_init";
                    rawMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                    pubWindowSubmapSC.publish(rawMsg);
                }

                // 4) RANSAC 去地面
                pcl::PointCloud<PointType>::Ptr windowSubmapNoGround = removeGroundRANSAC(windowSubmap);

                // 5) 发布去地面后子地图（用于 RViz 检查效果）
                {
                    sensor_msgs::PointCloud2 noGroundMsg;
                    pcl::toROSMsg(*windowSubmapNoGround, noGroundMsg);
                    noGroundMsg.header.frame_id = "camera_init";
                    noGroundMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                    pubWindowSubmapNoGround.publish(noGroundMsg);
                }

                // 6) 对去地面后子地图降采样，控制点数
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
                        double curr_altitude_offseted = currGPS->altitude - gpsAltitudeInitOffset;
                        mtxRecentPose.lock();
                        gtsam::Point3 gpsConstraint(recentOptimizedX, recentOptimizedY, curr_altitude_offseted); // in this example, only adjusting altitude (for x and y, very big noises are set) 
                        mtxRecentPose.unlock();
                        gtSAMgraph.add(gtsam::GPSFactor(curr_node_idx, gpsConstraint, robustGPSNoise));
                        cout << "GPS factor added at node " << curr_node_idx << endl;
                    }
                    initialEstimate.insert(curr_node_idx, poseTo);                
                    writeEdge({prev_node_idx, curr_node_idx}, relPose, edges_str); // giseop
                    // runISAM2opt();
                }
                mtxPosegraph.unlock();

                if(curr_node_idx % 100 == 0)
                    cout << "posegraph odom node " << curr_node_idx << " added." << endl;
            }
            // if want to print the current graph, use gtSAMgraph.print("\nFactor Graph:\n");

            // save utility
            pgTimeSaveStream << timeLaser << std::endl; // path 
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
        scLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
        mBuf.unlock();

        cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;
    }
} // performSCLoopClosure

// 回环检测线程：固定频率运行 Scan Context 检测。
void process_lcd(void)
{
    float loopClosureFrequency = 1.0; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        performSCLoopClosure();
        // performRSLoopClosure(); // TODO
    }
} // process_lcd

// 回环 ICP 线程：把候选回环转换成精确的位姿约束。
void process_icp(void)
{
    while(ros::ok())
    {
		while ( ros::ok() && !scLoopICPBuf.empty() )
        {
            if( scLoopICPBuf.size() > 30 ) {
                ROS_WARN("Too many loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
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
                writeEdge({prev_node_idx, curr_node_idx}, relative_pose, edges_str); // giseop
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

// 轨迹可视化线程：高频发布优化后的路径。
void process_viz_path(void)
{
    float hz = 10.0; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated.load() > 1) {
            pubPath();
        }
    }
}

// 定时运行 iSAM2，并把最新优化结果导出到磁盘。
void process_isam(void)
{
    float hz = 1; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
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
        }
    }
}

// 把已优化的关键帧点云拼成稀疏全局地图并发布。
void pubMap(void)
{
    int SKIP_FRAMES = 2; // sparse map visulalization to save computations 
    int counter = 0;

    laserCloudMapPGO->clear();

    mKF.lock(); 
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()); node_idx++) {
    for (int node_idx=0; node_idx < recentIdxUpdated.load(); node_idx++) {
        if(counter % SKIP_FRAMES == 0) {
            *laserCloudMapPGO += *local2global(keyframeLaserClouds[node_idx], keyframePosesUpdated[node_idx]);
        }
        counter++;
    }
    mKF.unlock(); 

    downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
    downSizeFilterMapPGO.filter(*laserCloudMapPGO);

    sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
    pcl::toROSMsg(*laserCloudMapPGO, laserCloudMapPGOMsg);
    laserCloudMapPGOMsg.header.frame_id = "camera_init";
    pubMapAftPGO.publish(laserCloudMapPGOMsg);
}

// 地图可视化线程：低频发布全局地图，减少计算开销。
void process_viz_map(void)
{
    float vizmapFrequency = 0.1; // 0.1 means run onces every 10s
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated.load() > 1) {
            pubMap();
        }
    }
} // pointcloud_viz

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


// 程序入口：初始化 ROS、参数、线程，然后在退出时统一保存结果。
int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserPGO");
	ros::NodeHandle nh;
	ros::NodeHandle pnh("~");


    // ------------------------- 输出文件路径 -------------------------
	nh.param<std::string>("save_directory", save_directory, "./"); // pose assignment every k m move 

    pgTUMformat = save_directory + "optimized_poses.txt";
    odomKITTIformat = save_directory + "odom_poses.txt";

    // pgG2oSaveStream = std::fstream(save_directory + "singlesession_posegraph.g2o", std::fstream::out);

    pgTimeSaveStream = std::fstream(save_directory + "times.txt", std::fstream::out); 
    pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);


    // ------------------------- 算法参数：关键帧、回环、地图滤波 -------------------------
	nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 2.0); // pose assignment every k m move 
	nh.param<double>("keyframe_deg_gap", keyframeDegGap, 10.0); // pose assignment every k deg rot 
    keyframeRadGap = deg2rad(keyframeDegGap);

	nh.param<double>("sc_dist_thres", scDistThres, 0.2);
	nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor
	pnh.param<bool>("use_ground_removal", useGroundRemoval, true);             // 是否启用 RANSAC 去地面
	pnh.param<bool>("use_icp_submap_enhancement", useICPSubmapEnhancement, true); // 是否启用 ICP 子地图增强

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    initNoises();

    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);


	// 关键帧点云、ICP 子地图分别使用不同下采样尺度。
    // 滑动窗口/空间近邻合并后点云范围大幅增加，leaf 过小会导致 VoxelGrid 整数索引溢出
    float sc_filter_size = 0.05;   // SC 描述子：5cm，7 帧滑动窗口
    float icp_filter_size = 0.1;   // ICP 子地图：10cm，时间窗口 + 空间近邻可能跨越 50m+
    downSizeFilterScancontext.setLeafSize(sc_filter_size, sc_filter_size, sc_filter_size);
    downSizeFilterICP.setLeafSize(icp_filter_size, icp_filter_size, icp_filter_size);

    double mapVizFilterSize;
	nh.param<double>("mapviz_filter_size", mapVizFilterSize, 0.4); // pose assignment every k frames 
    downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);


    // ------------------------- 订阅输入话题 -------------------------
    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_registered_local", 100, laserCloudFullResHandler);
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, laserOdometryHandler);
	ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>("/gps/fix", 100, gpsHandler);

    // ------------------------- 发布输出话题 -------------------------
	pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
	pubOdomRepubVerifier = nh.advertise<nav_msgs::Odometry>("/repub_odom", 100);
	pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
	pubMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);

	pubLoopScanLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local", 100);
	pubLoopSubmapLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_local", 100);
	pubLoopScanIcp = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_icp", 100);
	pubLoopSubmapIcp = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_icp", 100);
	pubWindowSubmapSC = nh.advertise<sensor_msgs::PointCloud2>("/window_submap_raw", 100);
	pubWindowSubmapNoGround = nh.advertise<sensor_msgs::PointCloud2>("/window_submap_noground", 100);


    // ------------------------- 后台工作线程 -------------------------
    std::thread posegraph_slam {process_pg}; // pose graph construction
	std::thread lc_detection {process_lcd}; // loop closure detection 
	std::thread icp_calculation {process_icp}; // loop constraint calculation via icp 
	std::thread isam_update {process_isam}; // if you want to call less isam2 run (for saving redundant computations and no real-time visulization is required), uncommment this and comment all the above runisam2opt when node is added. 

	std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
	std::thread viz_path {process_viz_path}; // visualization - path (high frequency)

 	ros::spin();

	// 退出后先等待所有线程结束，再统一保存最终结果。
    posegraph_slam.join();
    lc_detection.join();
    icp_calculation.join();
    isam_update.join();
    viz_map.join();
    viz_path.join();

    // save final results
    saveOptimizedVerticesTUMformat(isamCurrentEstimate, keyframeTimes, pgTUMformat);
    saveOdometryVerticesKITTIformat(odomKITTIformat);
    saveGTSAMgraphG2oFormat(isamCurrentEstimate);
    saveGlobalMap(save_directory + "global_map.pcd");
    recoverAllPosesTUM(save_directory + "all_optimized_poses.txt");

    pgTimeSaveStream.close();

	return 0;
}
