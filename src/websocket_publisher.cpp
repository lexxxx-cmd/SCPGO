#include "SCPGO/websocket_publisher.h"

#include <foxglove/websocket/base64.hpp>
#include <foxglove/websocket/common.hpp>

#include <nlohmann/json.hpp>

#include <iostream>
#include <cmath>

// ------------------------------------------------------------
// Foxglove PointCloud JSON schema (advertised to clients)
// ------------------------------------------------------------
static const char* POINTCLOUD_SCHEMA = R"raw(
{
  "type": "object",
  "properties": {
    "timestamp": {
      "type": "object",
      "properties": {
        "sec":  {"type": "integer"},
        "nsec": {"type": "integer"}
      }
    },
    "frame_id": {"type": "string"},
    "pose": {
      "type": "object",
      "properties": {
        "position": {
          "type": "object",
          "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"}
          }
        },
        "orientation": {
          "type": "object",
          "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
            "w": {"type": "number"}
          }
        }
      }
    },
    "point_stride": {"type": "integer", "const": 16},
    "fields": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name":   {"type": "string"},
          "offset": {"type": "integer"},
          "type":   {"type": "integer", "enum": [1,2,3,4,5,6,7,8]}
        }
      }
    },
    "data": {
      "type": "string",
      "contentEncoding": "base64"
    }
  }
}
)raw";

// ------------------------------------------------------------
// Foxglove PoseInFrame JSON schema
// ------------------------------------------------------------
static const char* POSEINFRAME_SCHEMA = R"raw(
{
  "type": "object",
  "properties": {
    "timestamp": {
      "type": "object",
      "properties": {
        "sec":  {"type": "integer"},
        "nsec": {"type": "integer"}
      }
    },
    "frame_id": {"type": "string"},
    "pose": {
      "type": "object",
      "properties": {
        "position": {
          "type": "object",
          "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"}
          }
        },
        "orientation": {
          "type": "object",
          "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
            "w": {"type": "number"}
          }
        }
      }
    }
  }
}
)raw";

// ------------------------------------------------------------
// Foxglove FrameTransform JSON schema
// ------------------------------------------------------------
static const char* FRAMETRANSFORM_SCHEMA = R"raw(
{
  "type": "object",
  "properties": {
    "timestamp": {
      "type": "object",
      "properties": {
        "sec":  {"type": "integer"},
        "nsec": {"type": "integer"}
      }
    },
    "parent_frame_id": {"type": "string"},
    "child_frame_id":  {"type": "string"},
    "translation": {
      "type": "object",
      "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"}
      }
    },
    "rotation": {
      "type": "object",
      "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
        "w": {"type": "number"}
      }
    }
  }
}
)raw";

// ================================================================
//  init
// ================================================================
bool WebSocketPublisher::init(const std::string& host, uint16_t port,
                              bool pubPC, bool pubPose,
                              double dsLeaf, int maxPts)
{
    enabled          = true;
    publishPointCloud = pubPC;
    publishPose      = pubPose;
    downsampleLeafSize = dsLeaf;
    maxPoints         = maxPts;

    // Configure down-sample filters once
    if (dsLeaf > 0.0) {
        voxelFilter.setLeafSize(dsLeaf, dsLeaf, dsLeaf);
    }
    if (maxPts > 0) {
        randomSampler.setSample(maxPts);
    }

    // --- Create server ---
    auto logHandler = [](foxglove::WebSocketLogLevel, char const* msg) {
        std::cerr << "[WS] " << msg << std::endl;
    };

    foxglove::ServerOptions opts;
    // Publication-only server — no need for clientPublish or other capabilities
    opts.capabilities = {};
    opts.supportedEncodings = {"json"};

    server = foxglove::ServerFactory::createServer<foxglove::ConnHandle>(
        "SCPGO", logHandler, opts);

    // Handlers — subscribe / unsubscribe are required for Foxglove
    // Studio to receive any data at all. Without them the server rejects
    // client subscription requests and no messages flow.
    foxglove::ServerHandlers<foxglove::ConnHandle> hdlrs;
    hdlrs.subscribeHandler = [](foxglove::ChannelId chanId, foxglove::ConnHandle) {
        std::cout << "[WS] Client subscribed to channel " << chanId << std::endl;
    };
    hdlrs.unsubscribeHandler = [](foxglove::ChannelId chanId, foxglove::ConnHandle) {
        std::cout << "[WS] Client unsubscribed from channel " << chanId << std::endl;
    };
    server->setHandlers(std::move(hdlrs));

    // --- Start listening ---
    server->start(host, port);
    std::cout << "[WS] Server started on " << host << ":" << port << std::endl;

    // --- Advertise channels ---
    std::vector<foxglove::ChannelWithoutId> channels;

    if (publishPointCloud) {
        channels.push_back({"pointcloud", "json", "foxglove.PointCloud", POINTCLOUD_SCHEMA});
    }
    if (publishPose) {
        channels.push_back({"pose", "json", "foxglove.PoseInFrame", POSEINFRAME_SCHEMA});
    }
    // Always advertise the frame-transform channel (required for 3D panel to
    // position point clouds / poses in a common reference frame).
    channels.push_back({"frame_transform", "json", "foxglove.FrameTransform", FRAMETRANSFORM_SCHEMA});

    // ICP detail & global map channels (world-frame point clouds)
    if (publishICPDetail) {
        channels.push_back({"icp_source",  "json", "foxglove.PointCloud", POINTCLOUD_SCHEMA});
        channels.push_back({"icp_target",  "json", "foxglove.PointCloud", POINTCLOUD_SCHEMA});
    }
    if (publishGlobalMapFlag) {
        channels.push_back({"global_map",  "json", "foxglove.PointCloud", POINTCLOUD_SCHEMA});
    }

    auto chanIds = server->addChannels(channels);

    int idx = 0;
    if (publishPointCloud) {
        pointcloudChannelId = chanIds[idx++];
        std::cout << "[WS] Advertised 'pointcloud' (chan " << pointcloudChannelId << ")" << std::endl;
    }
    if (publishPose) {
        poseChannelId = chanIds[idx++];
        std::cout << "[WS] Advertised 'pose' (chan " << poseChannelId << ")" << std::endl;
    }
    frameTransformChannelId = chanIds[idx++];
    std::cout << "[WS] Advertised 'frame_transform' (chan " << frameTransformChannelId << ")" << std::endl;
    if (publishICPDetail) {
        icpSourceChannelId = chanIds[idx++];
        icpTargetChannelId = chanIds[idx++];
        std::cout << "[WS] Advertised 'icp_source' (chan " << icpSourceChannelId << ")" << std::endl;
        std::cout << "[WS] Advertised 'icp_target' (chan " << icpTargetChannelId << ")" << std::endl;
    }
    if (publishGlobalMapFlag) {
        globalMapChannelId = chanIds[idx++];
        std::cout << "[WS] Advertised 'global_map' (chan " << globalMapChannelId << ")" << std::endl;
    }

    // Configure global-map filter
    if (globalMapLeafSize > 0.0) {
        globalMapVoxelFilter.setLeafSize(globalMapLeafSize, globalMapLeafSize, globalMapLeafSize);
    }

    return true;
}

// ================================================================
//  publishPairedFrame — point cloud FIRST, then pose SECOND
// ================================================================
void WebSocketPublisher::publishPairedFrame(
    const pcl::PointCloud<PointType>::Ptr& cloud,
    double timestampSec,
    const OdomData& odom)
{
    if (!enabled || !server) return;

    uint64_t ns  = static_cast<uint64_t>(timestampSec * 1e9);
    int64_t  sec = static_cast<int64_t>(timestampSec);
    int32_t  nsec = static_cast<int32_t>(ns - static_cast<uint64_t>(sec) * 1000000000ULL);

    // ================================================================
    //  STEP 1: Publish point cloud FIRST
    // ================================================================
    if (publishPointCloud && pointcloudChannelId != 0) {
        // -- down-sample (voxel grid, then random if still over cap) --
        pcl::PointCloud<PointType> filtered;
        const pcl::PointCloud<PointType>* src = cloud.get();

        if (downsampleLeafSize > 0.0) {
            voxelFilter.setInputCloud(cloud);
            voxelFilter.filter(filtered);
            src = &filtered;
        }

        pcl::PointCloud<PointType> sampled;
        if (maxPoints > 0 && static_cast<int>(src->size()) > maxPoints) {
            randomSampler.setInputCloud(src->makeShared());
            randomSampler.filter(sampled);
            src = &sampled;
        }

        // -- base64-encode the raw binary point data --
        const uint8_t* rawData = reinterpret_cast<const uint8_t*>(src->points.data());
        size_t rawSize = src->size() * sizeof(PointType);
        std::string b64 = foxglove::base64Encode(
            std::string_view(reinterpret_cast<const char*>(rawData), rawSize));

        // -- build JSON message --
        nlohmann::json msg;
        msg["timestamp"]["sec"]  = sec;
        msg["timestamp"]["nsec"] = nsec;
        msg["frame_id"] = "base_link";
        msg["pose"]["position"]["x"] = 0.0;
        msg["pose"]["position"]["y"] = 0.0;
        msg["pose"]["position"]["z"] = 0.0;
        msg["pose"]["orientation"]["x"] = 0.0;
        msg["pose"]["orientation"]["y"] = 0.0;
        msg["pose"]["orientation"]["z"] = 0.0;
        msg["pose"]["orientation"]["w"] = 1.0;
        msg["point_stride"] = 16;
        msg["fields"] = {
            {{"name", "x"},   {"offset", 0},  {"type", 7}},   // FLOAT32
            {{"name", "y"},   {"offset", 4},  {"type", 7}},   // FLOAT32
            {{"name", "z"},   {"offset", 8},  {"type", 7}},   // FLOAT32
            {{"name", "rgb"}, {"offset", 12}, {"type", 6}}    // UINT32
        };
        msg["data"] = b64;

        std::string payload = msg.dump();
        server->broadcastMessage(pointcloudChannelId, ns,
                                 reinterpret_cast<const uint8_t*>(payload.data()),
                                 payload.size());
    }

    // ================================================================
    //  STEP 2: Publish pose SECOND  (same timestamp — paired frame)
    // ================================================================
    if (publishPose && poseChannelId != 0) {
        nlohmann::json msg;
        msg["timestamp"]["sec"]  = sec;
        msg["timestamp"]["nsec"] = nsec;
        msg["frame_id"] = "map";
        msg["pose"]["position"]["x"] = odom.x;
        msg["pose"]["position"]["y"] = odom.y;
        msg["pose"]["position"]["z"] = odom.z;
        msg["pose"]["orientation"]["x"] = odom.qx;
        msg["pose"]["orientation"]["y"] = odom.qy;
        msg["pose"]["orientation"]["z"] = odom.qz;
        msg["pose"]["orientation"]["w"] = odom.qw;

        std::string payload = msg.dump();
        server->broadcastMessage(poseChannelId, ns,
                                 reinterpret_cast<const uint8_t*>(payload.data()),
                                 payload.size());
    }

    // ================================================================
    //  STEP 3: Publish frame transform (map -> base_link)
    //          Foxglove 3D panel uses this to position all frames
    //          relative to the display frame ("map").
    // ================================================================
    if (frameTransformChannelId != 0) {
        nlohmann::json tf;
        tf["timestamp"]["sec"]  = sec;
        tf["timestamp"]["nsec"] = nsec;
        tf["parent_frame_id"] = "map";
        tf["child_frame_id"]  = "base_link";
        tf["translation"]["x"] = odom.x;
        tf["translation"]["y"] = odom.y;
        tf["translation"]["z"] = odom.z;
        tf["rotation"]["x"] = odom.qx;
        tf["rotation"]["y"] = odom.qy;
        tf["rotation"]["z"] = odom.qz;
        tf["rotation"]["w"] = odom.qw;

        std::string tfPayload = tf.dump();
        server->broadcastMessage(frameTransformChannelId, ns,
                                 reinterpret_cast<const uint8_t*>(tfPayload.data()),
                                 tfPayload.size());
    }
}

// ================================================================
//  Helper: build and broadcast a PointCloud JSON message
// ================================================================
static void broadcastPointCloudMsg(
    foxglove::ServerInterface<foxglove::ConnHandle>& server,
    foxglove::ChannelId chanId,
    const pcl::PointCloud<PointType>::Ptr& cloud,
    int64_t sec, int32_t nsec, uint64_t ns,
    const std::string& frame_id)
{
    const uint8_t* rawData = reinterpret_cast<const uint8_t*>(cloud->points.data());
    size_t rawSize = cloud->size() * sizeof(PointType);
    std::string b64 = foxglove::base64Encode(
        std::string_view(reinterpret_cast<const char*>(rawData), rawSize));

    nlohmann::json msg;
    msg["timestamp"]["sec"]  = sec;
    msg["timestamp"]["nsec"] = nsec;
    msg["frame_id"] = frame_id;
    msg["pose"]["position"]["x"] = 0.0;
    msg["pose"]["position"]["y"] = 0.0;
    msg["pose"]["position"]["z"] = 0.0;
    msg["pose"]["orientation"]["x"] = 0.0;
    msg["pose"]["orientation"]["y"] = 0.0;
    msg["pose"]["orientation"]["z"] = 0.0;
    msg["pose"]["orientation"]["w"] = 1.0;
    msg["point_stride"] = 16;
    msg["fields"] = {
        {{"name", "x"},   {"offset", 0},  {"type", 7}},
        {{"name", "y"},   {"offset", 4},  {"type", 7}},
        {{"name", "z"},   {"offset", 8},  {"type", 7}},
        {{"name", "rgb"}, {"offset", 12}, {"type", 6}}
    };
    msg["data"] = b64;

    std::string payload = msg.dump();
    server.broadcastMessage(chanId, ns,
                            reinterpret_cast<const uint8_t*>(payload.data()),
                            payload.size());
}

// ================================================================
//  publishICPDetail
// ================================================================
void WebSocketPublisher::publishICP(
    const pcl::PointCloud<PointType>::Ptr& source,
    const pcl::PointCloud<PointType>::Ptr& target,
    double timestampSec)
{
    if (!enabled || !server) return;
    if (!publishICPDetail) return;

    uint64_t ns  = static_cast<uint64_t>(timestampSec * 1e9);
    int64_t  sec = static_cast<int64_t>(timestampSec);
    int32_t  nsec = static_cast<int32_t>(ns - static_cast<uint64_t>(sec) * 1000000000ULL);

    // Both clouds are already in world frame (built via local2global)
    if (icpSourceChannelId != 0 && source && !source->empty()) {
        broadcastPointCloudMsg(*server, icpSourceChannelId, source, sec, nsec, ns, "map");
    }
    if (icpTargetChannelId != 0 && target && !target->empty()) {
        broadcastPointCloudMsg(*server, icpTargetChannelId, target, sec, nsec, ns, "map");
    }
}

// ================================================================
//  publishGlobalMap
// ================================================================
void WebSocketPublisher::publishGlobalMap(
    const pcl::PointCloud<PointType>::Ptr& cloud,
    double timestampSec)
{
    if (!enabled || !server) return;
    if (!publishGlobalMapFlag) return;
    if (globalMapChannelId == 0) return;
    if (!cloud || cloud->empty()) return;

    // Down-sample for network transport
    pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
    if (globalMapLeafSize > 0.0) {
        globalMapVoxelFilter.setInputCloud(cloud);
        globalMapVoxelFilter.filter(*filtered);
    } else {
        *filtered = *cloud;
    }

    uint64_t ns  = static_cast<uint64_t>(timestampSec * 1e9);
    int64_t  sec = static_cast<int64_t>(timestampSec);
    int32_t  nsec = static_cast<int32_t>(ns - static_cast<uint64_t>(sec) * 1000000000ULL);

    broadcastPointCloudMsg(*server, globalMapChannelId, filtered, sec, nsec, ns, "map");
}

// ================================================================
//  shutdown
// ================================================================
void WebSocketPublisher::shutdown()
{
    if (!enabled || !server) return;

    std::cout << "[WS] Shutting down..." << std::endl;

    std::vector<foxglove::ChannelId> chanIds;
    if (publishPointCloud && pointcloudChannelId != 0)
        chanIds.push_back(pointcloudChannelId);
    if (publishPose && poseChannelId != 0)
        chanIds.push_back(poseChannelId);
    if (frameTransformChannelId != 0)
        chanIds.push_back(frameTransformChannelId);
    if (publishICPDetail && icpSourceChannelId != 0)
        chanIds.push_back(icpSourceChannelId);
    if (publishICPDetail && icpTargetChannelId != 0)
        chanIds.push_back(icpTargetChannelId);
    if (publishGlobalMapFlag && globalMapChannelId != 0)
        chanIds.push_back(globalMapChannelId);

    if (!chanIds.empty())
        server->removeChannels(chanIds);

    server->stop();
    server.reset();

    std::cout << "[WS] Server stopped." << std::endl;
}
