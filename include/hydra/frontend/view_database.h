#pragma once
#include "hydra/common/dsg_types.h"
#include "hydra/frontend/view_selector.h"

namespace hydra {

struct InputData;

class ViewDatabase {
 public:
  using Ptr = std::shared_ptr<ViewDatabase>;
  struct Config {
    std::string view_selection_method = "boundary";
  };

  explicit ViewDatabase(const Config& config);

  ~ViewDatabase();

  void updateAssignments(const DynamicSceneGraph& graph,
                         const std::unordered_set<NodeId>& active_places) const;

  // Assign features to ALL unassigned place nodes using all accumulated views.
  // Call this after frontend thread stops (no race condition).
  void finalizeFeatures(const DynamicSceneGraph& graph) const;

 protected:
  // Active views (pruned by distance from active places)
  mutable ViewSelector::FeatureList views_;
  // All views ever seen (camera position + feature), for finalize step
  struct ViewRecord {
    Eigen::Vector3d cam_w;
    FeatureVector feature;
  };
  mutable std::vector<ViewRecord> all_views_;
  std::unique_ptr<ViewSelector> view_selector_;
};

void declare_config(ViewDatabase::Config& config);

}  // namespace hydra
