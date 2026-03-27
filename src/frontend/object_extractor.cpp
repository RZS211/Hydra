/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include "hydra/frontend/object_extractor.h"

#include <config_utilities/config.h>
#include <config_utilities/types/conversions.h>
#include <config_utilities/types/enum.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/bounding_box_extraction.h>
#include <spark_dsg/printing.h>

#include "hydra/utils/mesh_utilities.h"
#include "hydra/utils/timing_utilities.h"

namespace hydra {
namespace {

inline std::string printLabels(const std::set<uint32_t>& labels) {
  std::stringstream ss;
  ss << "[";
  auto iter = labels.begin();
  while (iter != labels.end()) {
    ss << static_cast<uint64_t>(*iter);
    ++iter;
    if (iter != labels.end()) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

}  // namespace

using spark_dsg::BoundingBox;
using spark_dsg::DynamicSceneGraph;
using spark_dsg::NodeSymbol;
using spark_dsg::ObjectNodeAttributes;
using spatial_hash::IndexSet;
using timing::ScopedTimer;

using PosMap = spatial_hash::IndexHashMap<Eigen::Vector3f>;

float ObjectExtractor::ObjectInfo::intersection(const PosMap& points) const {
  if (active.empty() && frozen.empty()) {
    return 0.0f;
  }

  size_t num_equal = 0;
  for (const auto& [idx, _] : points) {
    num_equal += active.count(idx) || frozen.count(idx);
  }

  return static_cast<float>(num_equal) / (active.size() + frozen.size());
}

ObjectExtractor::Config::Config() : VerbosityConfig("[object_extractor] ") {}

void declare_config(ObjectExtractor::Config& config) {
  using namespace config;
  name("ObjectExtractor::Config");
  base<VerbosityConfig>(config);
  field(config.layer_id, "layer_id");
  field(config.grid_resolution_m, "grid_resolution_m");
  enum_field(config.bounding_box_type,
             "bounding_box_type",
             {{BoundingBox::Type::INVALID, "INVALID"},
              {BoundingBox::Type::AABB, "AABB"},
              {BoundingBox::Type::OBB, "OBB"},
              {BoundingBox::Type::RAABB, "RAABB"}});
  field(config.sinks, "sinks");

  check(config.grid_resolution_m, GT, 0.0f, "grid_resolution_m");
}

ObjectExtractor::ObjectExtractor(const Config& config, const std::set<uint32_t>& labels)
    : config(config::checkValid(config)),
      grid_(config.grid_resolution_m),
      sinks_(Sink::instantiate(config.sinks)),
      labels_(labels),
      next_node_id_('O', 0) {
  MLOG(1) << "using labels: " << printLabels(labels_);
}

void ObjectExtractor::detect(const ActiveWindowOutput& msg) {
  ScopedTimer timer("frontend/object_detection", msg.timestamp_ns, true, 1, false);
  updatePoints(msg);

  for (const auto& [label, cloud] : points_) {
    if (cloud.size() < config.min_object_size) {
      MLOG(2) << "skipping label " << label << " with " << cloud.size() << " points";
      continue;
    }

    const auto clusters = getConnectedComponents(cloud, config.cluster_tolerance);
    MLOG(2) << "found " << clusters.size() << " cluster(s) of label " << label;
    for (const auto& cluster : clusters) {
      if (cluster.size() < config.min_object_size) {
        continue;
      }

      bool matched = false;
      spatial_hash::IndexHashMap<Point> curr_cloud;
      for (const auto idx : cluster) {
        const auto pos = cloud[idx];
        curr_cloud.emplace(grid_.toIndex(pos), pos);
      }

      for (const auto& [node_id, info] : objects_) {
        if (info.label != label) {
          continue;
        }

        const auto intersection = info.intersection(curr_cloud);
        MLOG(3) << "Intersection with node " << NodeSymbol(node_id).str() << ": "
                << intersection;
        if (intersection > config.min_intersection_volume) {
          matched = true;
          // TODO(nathan) merge clouds
          break;
        }
      }

      if (!matched) {
        objects_.emplace(next_node_id_, ObjectInfo{label, std::move(curr_cloud), {}});
        ++next_node_id_;
      }
    }
  }

  Sink::callAll(sinks_, msg);
}

void ObjectExtractor::updatePoints(const ActiveWindowOutput& msg) {
  const auto& map = msg.map();
  const auto& tsdf = map.getTsdfLayer();
  for (auto& [label, cloud] : points_) {
    cloud.erase([&tsdf, this](const HashedCloud::Pos& pos) -> bool {
      const auto voxel = tsdf.getVoxelPtr(pos);
      if (!voxel || voxel->weight < config.min_observation_weight) {
        return false;
      }

      return voxel->distance > 0.0f;
    });
  }

  const auto& mesh = map.getMeshLayer();
  for (const auto& block : mesh) {
    if (!block.has_labels) {
      continue;
    }

    for (size_t idx = 0; idx < block.numVertices(); ++idx) {
      const auto label = block.label(idx);
      if (!labels_.count(label)) {
        continue;
      }

      auto iter = points_.find(label);
      if (iter == points_.end()) {
        iter = points_.emplace(label, HashedCloud(config.grid_resolution_m)).first;
      }

      iter->second.addPoint(block.pos(idx));
    }
  }
}

void ObjectExtractor::updateGraph(uint64_t timestamp_ns, DynamicSceneGraph& graph) {
  ScopedTimer timer("frontend/object_graph_update", timestamp_ns, true, 1, false);
  for (const auto node : deleted_) {
    graph.removeNode(node);
  }

  for (const auto node_id : archived_) {
    graph.getNode(node_id).attributes().is_active = false;
  }

  deleted_.clear();
  archived_.clear();

  for (const auto& [node_id, info] : objects_) {
    auto attrs = std::make_unique<ObjectNodeAttributes>();
    attrs->last_update_time_ns = timestamp_ns;
    attrs->is_active = true;
    attrs->semantic_label = info.label;
    attrs->bounding_box = BoundingBox(info.cloud->points(), config.bounding_box_type);
    attrs->position = attrs->bounding_box.world_P_center.cast<double>();
    // TODO(nathan) fill object with points
    graph.addOrUpdateNode(config.layer_id, node_id, std::move(attrs));
  }
}

}  // namespace hydra
