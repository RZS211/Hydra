/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * all rights reserved
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
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <hydra/frontend/object_extractor.h>

namespace hydra {

using namespace spark_dsg;

namespace {

struct NodeResult {
  uint32_t label;
  BoundingBox bbox;
  bool is_active = true;
};

bool checkGraph(const DynamicSceneGraph& graph,
                const std::map<NodeId, NodeResult>& expected_nodes,
                float tolerance = 1.0e-4f) {
  const auto& layer = graph.getLayer(DsgLayers::OBJECTS);
  EXPECT_EQ(layer.numNodes(), expected_nodes.size());

  bool all_present = true;
  for (const auto& [node_id, expected] : expected_nodes) {
    SCOPED_TRACE("Object " + NodeSymbol(node_id).str());
    const auto node = layer.findNode(node_id);
    if (!node) {
      all_present = false;
      continue;
    }

    auto& result = node->attributes<ObjectNodeAttributes>();
    const auto& result_bbox = result.bounding_box;
    EXPECT_EQ(expected.label, result.semantic_label);
    EXPECT_EQ(expected.is_active, result.is_active);
    EXPECT_NEAR(expected.bbox.dimensions.x(), result_bbox.dimensions.x(), tolerance);
    EXPECT_NEAR(expected.bbox.dimensions.y(), result_bbox.dimensions.y(), tolerance);
    EXPECT_NEAR(expected.bbox.dimensions.z(), result_bbox.dimensions.z(), tolerance);
    EXPECT_NEAR(
        expected.bbox.world_P_center.x(), result_bbox.world_P_center.x(), tolerance);
    EXPECT_NEAR(
        expected.bbox.world_P_center.y(), result_bbox.world_P_center.y(), tolerance);
    EXPECT_NEAR(
        expected.bbox.world_P_center.z(), result_bbox.world_P_center.z(), tolerance);
    EXPECT_NEAR(expected.bbox.world_P_center.x(), result.position.x(), tolerance);
    EXPECT_NEAR(expected.bbox.world_P_center.y(), result.position.y(), tolerance);
    EXPECT_NEAR(expected.bbox.world_P_center.z(), result.position.z(), tolerance);
  }

  return all_present;
}

class MapFixture {
 public:
  MapFixture() : map_(std::make_shared<VolumetricMap>(VolumetricMap::Config{})) {}

  void addPoints(uint32_t label, Eigen::Vector3f centroid, Eigen::Vector3f dims) {
    auto& mesh = map_->getMeshLayer();
    BoundingBox box(dims, centroid);
    const auto points = box.corners();
    for (const auto& point : points) {
      const auto block_idx = mesh.getBlockIndex(point);
      if (!mesh.hasBlock(block_idx)) {
        mesh.allocateBlock(block_idx, true, false);
      }

      auto& block = mesh.getBlock(block_idx);
      block.points.push_back(point);
      block.labels.push_back(label);
    }
  }

  ActiveWindowOutput makeMsg() {
    ActiveWindowOutput msg;
    msg.timestamp_ns = 0;
    msg.world_t_body = Eigen::Vector3d::Zero();
    msg.world_R_body = Eigen::Quaterniond::Identity();
    msg.setMap(map_);
    return msg;
  }

 private:
  std::shared_ptr<VolumetricMap> map_;
};

}  // namespace

TEST(ObjectExtractor, Clustering) {
  const auto dims = Eigen::Vector3f::Constant(0.1);

  SceneGraph graph;
  ObjectExtractor::Config config;
  config.min_object_size = 4;
  ObjectExtractor extractor(config, {1, 2});

  { // setup original objects
    MapFixture fixture;
    fixture.addPoints(1, {1, 2, 3}, dims);
    fixture.addPoints(2, {4, 5, 6}, dims);
    extractor.detect(fixture.makeMsg());
    extractor.updateGraph(0, graph);

    const std::map<NodeId, NodeResult> expected{
        {"O0"_id, {1, BoundingBox(dims, {1, 2, 3}), true}},
        {"O1"_id, {2, BoundingBox(dims, {4, 5, 6}), true}}};
    checkGraph(graph, expected);
  }

  { // readd original objects
    MapFixture fixture;
    fixture.addPoints(1, {1, 2, 3}, dims);
    fixture.addPoints(2, {4, 5, 6}, dims);
    extractor.detect(fixture.makeMsg());
    extractor.updateGraph(0, graph);

    const std::map<NodeId, NodeResult> expected{
        {"O0"_id, {1, BoundingBox(dims, {1, 2, 3}), true}},
        {"O1"_id, {2, BoundingBox(dims, {4, 5, 6}), true}}};
    checkGraph(graph, expected);
  }
}

TEST(ObjectExtractor, DeletedObject) {
  const auto dims = Eigen::Vector3f::Constant(0.1);

  SceneGraph graph;
  ObjectExtractor::Config config;
  config.min_object_size = 4;
  ObjectExtractor extractor(config, {1, 2});

  {  // setup original objects
    MapFixture fixture;
    fixture.addPoints(1, {1, 2, 3}, Eigen::Vector3f::Constant(0.1));
    fixture.addPoints(2, {4, 5, 6}, Eigen::Vector3f::Constant(0.1));
    extractor.detect(fixture.makeMsg());
    extractor.updateGraph(0, graph);

    const std::map<NodeId, NodeResult> expected{
        {"O0"_id, {1, BoundingBox(dims, {1, 2, 3}), true}},
        {"O1"_id, {2, BoundingBox(dims, {4, 5, 6}), true}},
    };
    checkGraph(graph, expected);
  }

  {  // delete first object
    MapFixture fixture;
    fixture.addPoints(2, {4, 5, 6}, Eigen::Vector3f::Constant(0.1));
    extractor.detect(fixture.makeMsg());
    extractor.updateGraph(0, graph);

    const std::map<NodeId, NodeResult> expected{
        {"O1"_id, {2, BoundingBox(dims, {4, 5, 6}), true}},
    };
    checkGraph(graph, expected);
  }

  {  // delete second object
    MapFixture fixture;
    fixture.addPoints(1, {1, 2, 3}, Eigen::Vector3f::Constant(0.1));
    extractor.detect(fixture.makeMsg());
    extractor.updateGraph(0, graph);

    const std::map<NodeId, NodeResult> expected{
        {"O2"_id, {1, BoundingBox(dims, {1, 2, 3}), true}},
    };
    checkGraph(graph, expected);
  }

  {  // delete all objects
    MapFixture fixture;
    extractor.detect(fixture.makeMsg());
    extractor.updateGraph(0, graph);

    const std::map<NodeId, NodeResult> expected{};
    checkGraph(graph, expected);
  }
}

/*
TEST(ObjectExtractor, TestArchivedObject) {
  Eigen::Vector3f dims = Eigen::Vector3f::Constant(0.1);
  const BoundingBox b1(dims, Eigen::Vector3f(1, 2, 3));
  const BoundingBox b2(dims, Eigen::Vector3f(4, 5, 6));

  ObjectExtractor::Config config;
  config.clustering.min_cluster_size = 4;
  ObjectExtractor segmenter(config, {1, 2});

  DynamicSceneGraph graph;
  kimera_pgmo::MeshOffsetInfo offsets;
  graph.setMesh(std::make_shared<spark_dsg::Mesh>());

  {  // setup original objects
    MeshDelta delta({0, 0, 0});
    addPoints(delta, 1, {1, 2, 3}, dims);
    addPoints(delta, 2, {4, 5, 6}, dims);
    stepSegmenter(delta, offsets, segmenter, graph);
  }

  {  // archive object two
    std::map<size_t, size_t> remap;
    for (size_t i = 0; i < 8; ++i) {
      // archived object gets moved down
      remap[i + 8] = i;
      // active object gets moved up
      remap[i] = i + 8;
    }

    const auto tracking = MeshDelta::TrackingInfo::with_remap(0, 16, 0, remap);
    MeshDelta delta(tracking);
    addPoints(delta, 2, {4, 5, 6}, dims, true);
    addPoints(delta, 1, {1, 2, 3}, dims);
    addPoints(delta, 2, {4, 5, 6}, dims);
    stepSegmenter(delta, offsets, segmenter, graph);

    const std::map<NodeId, NodeResult> expected_nodes{
        {"O0"_id, {1, {8, 9, 10, 11, 12, 13, 14, 15}, b1}},
        {"O1"_id, {2, {0, 1, 2, 3, 4, 5, 6, 7}, b2, false}},
        {"O2"_id, {2, {16, 17, 18, 19, 20, 21, 22, 23}, b2}},
    };
    for (const auto& [node_id, expected] : expected_nodes) {
      SCOPED_TRACE("Object " + NodeSymbol(node_id).str());
      EXPECT_TRUE(checkNode(graph, node_id, expected));
    }
  }
}

TEST(ObjectExtractor, TestDeltaWithOffset) {
  Eigen::Vector3f dims = Eigen::Vector3f::Constant(0.1);
  const BoundingBox b1(dims, Eigen::Vector3f(1, 2, 3));

  ObjectExtractor::Config config;
  config.clustering.min_cluster_size = 4;
  ObjectExtractor segmenter(config, {1, 2});

  DynamicSceneGraph graph;
  kimera_pgmo::MeshOffsetInfo offsets;
  graph.setMesh(std::make_shared<spark_dsg::Mesh>());

  {  // add 8 archived vertices to mesh
    MeshDelta delta({0, 0, 0});
    addPoints(delta, 1, {1, 2, 3}, dims, true);
    delta.updateMesh(*graph.mesh(), offsets);
  }

  {  // add actual object
    MeshDelta delta({0, 0, 0});
    addPoints(delta, 1, {1, 2, 3}, dims);
    stepSegmenter(delta, offsets, segmenter, graph);

    const std::map<NodeId, NodeResult> expected_nodes{
        {"O0"_id, {1, {8, 9, 10, 11, 12, 13, 14, 15}, b1}},
    };
    for (const auto& [node_id, expected] : expected_nodes) {
      SCOPED_TRACE("Object " + NodeSymbol(node_id).str());
      EXPECT_TRUE(checkNode(graph, node_id, expected));
    }
  }
}
*/

}  // namespace hydra
