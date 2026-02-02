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

#include <deque>

#include "hydra/utils/nearest_neighbor_utilities.h"
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

using timing::ScopedTimer;

using spark_dsg::BoundingBox;
using spark_dsg::DynamicSceneGraph;
using spatial_hash::IndexSet;

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
      sinks_(Sink::instantiate(config.sinks)),
      labels_(labels),
      next_node_id_('O', 0),
      grid_(config.grid_resolution_m) {
  MLOG(1) << "using labels: " << printLabels(labels_);
}

void ObjectExtractor::detect(const ActiveWindowOutput& msg) {
  ScopedTimer timer("frontend/object_detection", msg.timestamp_ns, true, 1, false);
  updatePoints(msg);

  /*
  LabelClusters label_clusters;
  if (!delta.getNumActiveVertices()) {
    VLOG(2) << "[Mesh Segmenter] No active indices in mesh";
    return label_clusters;
  }

  const auto label_indices = clustering::getLabelIndices(labels_, delta);
  if (label_indices.empty()) {
    VLOG(2) << "[Mesh Segmenter] No vertices found matching desired labels";
    return label_clusters;
  }

  for (const auto& [label, indices] : label_indices) {
    if (indices.size() < config.clustering.min_cluster_size) {
      continue;
    }

    const auto result = clustering::findClusters(config.clustering, delta, indices);

    auto iter = label_clusters.insert({label, {}}).first;
    auto& clusters = iter->second;
    for (const auto& cluster_indices : result) {
      auto& cluster = clusters.emplace_back();
      for (const auto local_idx : cluster_indices) {
        cluster.indices.push_back(local_idx);
        cluster.centroid += delta.getVertex(local_idx).pos.cast<double>();
      }

      if (cluster_indices.size()) {
        cluster.centroid /= cluster_indices.size();
      }
    }

    VLOG(2) << "[Mesh Segmenter] Found " << clusters.size() << " cluster(s) of label "
            << label;
  }
  */

  Sink::callAll(sinks_, msg);
}

void ObjectExtractor::updatePoints(const ActiveWindowOutput& msg) {
  const auto& map = msg.map();
  // TODO(nathan) clear freespace points with TSDF

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

      const auto& pos = block.pos(idx);
      const auto grid_idx = grid_.toIndex(pos);
      auto iter = points_.find(label);
      if (iter == points_.end()) {
        iter = points_.emplace(label, IndexSet{}).first;
      }

      iter->second.insert(grid_idx);
    }
  }
}

std::vector<ObjectExtractor::Cluster> segmentPoints(
    const std::vector<Eigen::Vector3f>& points, float radius_m) {
  PointNeighborSearch search(points);
  std::vector<bool> seen(points.size(), false);

  std::vector<ObjectExtractor::Cluster> clusters;
  for (size_t seed = 0; seed < points.size(); ++seed) {
    if (seen[seed]) {
      continue;
    }

    std::deque<size_t> frontier{seed};
    auto& cluster = clusters.emplace_back();
    while (!frontier.empty()) {
      const size_t i = frontier.front();
      const auto& pos = points[i];
      cluster.indices.push_back(i);
      cluster.centroid += pos;
      frontier.pop_front();
      const auto neighbors = search.pointsInRadius(pos, radius_m);
      for (const auto idx : neighbors) {
        if (seen[idx]) {
          continue;
        }

        frontier.push_back(idx);
        seen[idx] = true;
      }
    }

    cluster.centroid /= cluster.indices.size();
  }

  return clusters;
}

void ObjectExtractor::updateGraph(uint64_t timestamp_ns, DynamicSceneGraph& graph) {}

/*
void ObjectExtractor::updateOldNodes(const kimera_pgmo::MeshOffsetInfo& offsets,
                                     DynamicSceneGraph& graph) {
  for (auto& [label, label_nodes] : active_nodes_) {
    auto iter = label_nodes.begin();
    while (iter != label_nodes.end()) {
      const auto node_id = *iter;
      auto& attrs = graph.getNode(node_id).attributes<ObjectNodeAttributes>();

      // remap and prune mesh connections
      VLOG(20) << "Updating node " << NodeSymbol(node_id).str() << " with connections "
               << attrs.mesh_connections;
      kimera_pgmo::MeshOffsetInfo::RemapStats stats;
      offsets.remapVertexIndices(attrs.mesh_connections, &stats);
      VLOG(20) << "After update: " << attrs.mesh_connections << std::boolalpha
               << " (active: " << !stats.all_archived << ")";
      if (attrs.mesh_connections.size() < config.clustering.min_cluster_size) {
        graph.removeNode(node_id);
        iter = label_nodes.erase(iter);
        continue;
      }

      attrs.is_active = !stats.all_archived;
      if (!attrs.is_active) {
        iter = label_nodes.erase(iter);
      } else {
        ++iter;
      }
    }
  }

  for (const auto& [label, label_nodes] : active_nodes_) {
    VLOG(10) << "Active nodes for label " << label << ": "
             << displayNodeSymbolContainer(label_nodes);
  }
}

void ObjectExtractor::updateGraph(uint64_t timestamp_ns,
                                  const kimera_pgmo::MeshOffsetInfo& offsets,
                                  const LabelClusters& clusters,
                                  DynamicSceneGraph& graph) {
  ScopedTimer timer(config.timer_namespace + "_graph_update", timestamp_ns);
  updateOldNodes(offsets, graph);
  if (!graph.hasMesh()) {
    LOG(ERROR) << "Unable to update graph without mesh!";
    return;
  }

  for (auto&& [label, clusters_for_label] : clusters) {
    for (const auto& cluster : clusters_for_label) {
      bool matches_prev_node = false;
      std::vector<NodeId> nodes_not_in_graph;
      for (const auto& prev_node_id : active_nodes_.at(label)) {
        const auto& prev_node = graph.getNode(prev_node_id);
        if (nodesMatch(cluster, prev_node)) {
          updateNodeInGraph(graph, cluster, prev_node, timestamp_ns);
          matches_prev_node = true;
          break;
        }
      }

      if (!matches_prev_node) {
        addNodeToGraph(graph, cluster, label, timestamp_ns);
      }

      mergeActiveNodes(graph, label);
    }
  }
}

void ObjectExtractor::mergeActiveNodes(DynamicSceneGraph& graph, uint32_t label) {
  std::set<NodeId> merged_nodes;

  auto& curr_active = active_nodes_.at(label);
  for (const auto& node_id : curr_active) {
    if (merged_nodes.count(node_id)) {
      continue;
    }
    const auto& node = graph.getNode(node_id);

    std::list<NodeId> to_merge;
    for (const auto& other_id : curr_active) {
      if (node_id == other_id) {
        continue;
      }

      if (merged_nodes.count(other_id)) {
        continue;
      }

      const auto& other = graph.getNode(other_id);
      if (nodesMatch(node, other) || nodesMatch(other, node)) {
        to_merge.push_back(other_id);
      }
    }

    auto& attrs = node.attributes<ObjectNodeAttributes>();
    for (const auto& other_id : to_merge) {
      const auto& other = graph.getNode(other_id);
      auto& other_attrs = other.attributes<ObjectNodeAttributes>();
      mergeList(attrs.mesh_connections, other_attrs.mesh_connections);
      graph.removeNode(other_id);
      merged_nodes.insert(other_id);
    }

    if (!to_merge.empty()) {
      updateObjectGeometry(*graph.mesh(), attrs);
    }
  }

  for (const auto& node_id : merged_nodes) {
    curr_active.erase(node_id);
  }
}

std::unordered_set<NodeId> ObjectExtractor::getActiveNodes() const {
  std::unordered_set<NodeId> active_nodes;
  for (const auto& label_nodes_pair : active_nodes_) {
    active_nodes.insert(label_nodes_pair.second.begin(), label_nodes_pair.second.end());
  }
  return active_nodes;
}

void ObjectExtractor::updateNodeInGraph(DynamicSceneGraph& graph,
                                        const Cluster& cluster,
                                        const SceneGraphNode& node,
                                        uint64_t timestamp) {
  auto& attrs = node.attributes<ObjectNodeAttributes>();
  attrs.last_update_time_ns = timestamp;
  attrs.is_active = true;

  mergeList(attrs.mesh_connections, cluster.indices);
  updateObjectGeometry(*graph.mesh(), attrs);
}

void ObjectExtractor::addNodeToGraph(DynamicSceneGraph& graph,
                                     const Cluster& cluster,
                                     uint32_t label,
                                     uint64_t timestamp) {
  if (cluster.indices.empty()) {
    LOG(ERROR) << "Encountered empty cluster with label" << static_cast<int>(label)
               << " @ " << timestamp << "[ns]";
    return;
  }

  auto attrs = std::make_unique<ObjectNodeAttributes>();
  attrs->last_update_time_ns = timestamp;
  attrs->is_active = true;
  attrs->semantic_label = label;
  attrs->mesh_connections.insert(
      attrs->mesh_connections.begin(), cluster.indices.begin(), cluster.indices.end());

  updateObjectGeometry(*graph.mesh(), *attrs, nullptr, config.bounding_box_type);

  graph.emplaceNode(config.layer_id, next_node_id_, std::move(attrs));
  active_nodes_.at(label).insert(next_node_id_);
  ++next_node_id_;
}
*/

}  // namespace hydra
