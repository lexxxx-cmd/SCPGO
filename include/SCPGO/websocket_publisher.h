#pragma once

#include <foxglove/websocket/server_factory.hpp>
#include <foxglove/websocket/websocket_notls.hpp>
#include <foxglove/websocket/websocket_server.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>

#include <memory>
#include <string>

// Forward declarations — these types are defined in laserPosegraphOptimization.cpp
struct Pose6D;
typedef pcl::PointXYZRGB PointType;

// TUM-format odometry data (replaces nav_msgs::Odometry).
// Defined here so both the main TU and the publisher implementation
// can see the full definition.
struct OdomData {
    double timestamp;
    double x, y, z;
    double qx, qy, qz, qw;  // quaternion
};

// ------------------------------------------------------------
// WebSocketPublisher
//
// Owns a Foxglove WebSocket server and two advertised channels
// ("pointcloud" / "pose").  Call publishPairedFrame() for each
// frame consumed from the input queues — it sends the point cloud
// first, then the pose, using the same timestamp so the receiver
// can pair them.
// ------------------------------------------------------------
struct WebSocketPublisher {
    bool enabled         = false;
    bool publishPointCloud = true;
    bool publishPose     = true;
    bool publishICPDetail = true;
    bool publishGlobalMapFlag = true;

    std::unique_ptr<foxglove::ServerInterface<foxglove::ConnHandle>> server;

    foxglove::ChannelId pointcloudChannelId       = 0;
    foxglove::ChannelId poseChannelId             = 0;
    foxglove::ChannelId frameTransformChannelId   = 0;
    foxglove::ChannelId icpSourceChannelId        = 0;
    foxglove::ChannelId icpTargetChannelId        = 0;
    foxglove::ChannelId globalMapChannelId        = 0;

    // Down-sampling filters (created once, reused across frames)
    pcl::VoxelGrid<PointType>   voxelFilter;
    pcl::RandomSample<PointType> randomSampler;
    pcl::VoxelGrid<PointType>   globalMapVoxelFilter;
    double downsampleLeafSize = 0.0;
    double globalMapLeafSize  = 0.2;
    int    maxPoints          = 0;

    // ---- lifecycle ----

    /// Create the server, start listening, and advertise channels.
    /// Returns true on success.
    bool init(const std::string& host, uint16_t port,
              bool pubPC, bool pubPose, double dsLeaf, int maxPts);

    /// Publish a paired frame: point cloud FIRST, then pose SECOND.
    /// Both messages share `timestampSec` so the receiver can match them.
    /// Called from process_pg() after one frame is popped from the queues.
    void publishPairedFrame(const pcl::PointCloud<PointType>::Ptr& cloud,
                            double timestampSec,
                            const OdomData& odom);

    /// Publish ICP detail: source and target submap clouds (world frame).
    /// Called from doICPVirtualRelative() for debugging loop closures.
    void publishICP(const pcl::PointCloud<PointType>::Ptr& source,
                          const pcl::PointCloud<PointType>::Ptr& target,
                          double timestampSec);

    /// Publish global point cloud map after PGO optimization.
    /// `cloud` must already be in world frame; it will be down-sampled
    /// before sending.
    void publishGlobalMap(const pcl::PointCloud<PointType>::Ptr& cloud,
                          double timestampSec);

    /// Remove channels, stop server, release resources.
    void shutdown();
};
