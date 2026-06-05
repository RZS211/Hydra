#include "hydra/frontend/view_database.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <glog/logging.h>

#include <spark_dsg/dynamic_scene_graph.h>

#include "hydra/common/pipeline_queues.h"
#include "hydra/input/input_data.h"

namespace hydra {

void declare_config(ViewDatabase::Config& config) {
  using namespace config;
  name("ViewDatabase::Config");
  field(config.view_selection_method, "view_selection_method");
}

using NodeSet = std::unordered_set<NodeId>;

ViewDatabase::ViewDatabase(const Config& config)
    : view_selector_(config::create<ViewSelector>(config.view_selection_method)) {
  CHECK(view_selector_);
}

ViewDatabase::~ViewDatabase() {}

void ViewDatabase::updateAssignments(const DynamicSceneGraph& graph,
                                     const NodeSet& active_places) const {
  auto& queue = PipelineQueues::instance().input_features_queue;
  size_t new_views = 0;
  while (!queue.empty()) {
    ++new_views;
    auto view = queue.pop();
    if (view) {
      // Accumulate ALL views for finalize step (camera pos + feature only)
      all_views_.push_back({view->sensor_T_world.inverse().translation(), view->feature});
    }
    views_.push_back(std::move(view));
  }

  VLOG(2) << "Got " << new_views << " new views!";

  auto iter = views_.begin();
  while (iter != views_.end()) {
    const auto& view = *iter;
    if (!view) {
      iter = views_.erase(iter);
      continue;
    }

    // Keep views that are within range of any active place (distance-based,
    // not frustum-based) so that place nodes outside the camera frustum can
    // still receive features via NearestViewSelector.
    const Eigen::Vector3d cam_w = view->sensor_T_world.inverse().translation();
    bool visible = false;
    for (const auto node_id : active_places) {
      const auto& pos = graph.getNode(node_id).attributes().position;
      if ((pos - cam_w).norm() < 15.0) {
        visible = true;
        break;
      }
    }

    if (!visible) {
      iter = views_.erase(iter);
      continue;
    }

    ++iter;
  }

  if (views_.empty()) {
    VLOG(2) << "No views assigned!";
    return;
  }

  VLOG(2) << "Assigning features with " << views_.size() << " active view(s)";
  for (const auto node_id : active_places) {
    auto& attrs = graph.getNode(node_id).attributes<SemanticNodeAttributes>();
    view_selector_->selectFeature(views_, attrs);
  }
}

void ViewDatabase::finalizeFeatures(const DynamicSceneGraph& graph) const {
  if (all_views_.empty()) {
    LOG(WARNING) << "[ViewDatabase] finalizeFeatures: no views accumulated";
    return;
  }
  if (!graph.hasLayer(DsgLayers::PLACES)) {
    return;
  }

  const auto& places_layer = graph.getLayer(DsgLayers::PLACES);
  size_t assigned = 0;
  size_t already_assigned = 0;

  for (const auto& id_node : places_layer.nodes()) {
    auto& attrs = id_node.second->attributes<SemanticNodeAttributes>();

    // Skip places that already have a valid feature
    if (attrs.semantic_feature.size() > 0 &&
        attrs.semantic_feature.norm() > 1e-6) {
      ++already_assigned;
      continue;
    }

    // Find nearest camera position from all accumulated views
    double min_dist = std::numeric_limits<double>::max();
    const ViewRecord* best = nullptr;
    for (const auto& rec : all_views_) {
      const double dist = (attrs.position - rec.cam_w).norm();
      if (dist < min_dist) {
        min_dist = dist;
        best = &rec;
      }
    }

    if (best) {
      attrs.semantic_feature = best->feature;
      ++assigned;
    }
  }

  LOG(INFO) << "[ViewDatabase] finalizeFeatures: " << assigned << " newly assigned, "
            << already_assigned << " already had features, from "
            << all_views_.size() << " total views";
}

}  // namespace hydra
