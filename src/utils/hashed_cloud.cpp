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
#include "hydra/utils/hashed_cloud.h"

#include <glog/logging.h>

namespace hydra {

using spatial_hash::IndexSet;

HashedCloud::HashedCloud(float resolution_m) : grid_(resolution_m) {}

size_t HashedCloud::size() const { return points_.size(); }

HashedCloud::Pos HashedCloud::get(size_t index) const {
  return points_.at(index).pos;
}

void HashedCloud::addPoint(const Pos& pos, Mode mode) {
  const auto idx = grid_.toIndex(pos);
  auto iter = lookup_.find(idx);
  if (iter == lookup_.end()) {
    lookup_.emplace(idx, points_.size());
    points_.push_back({pos, idx});
    return;
  }

  auto& point = points_[iter->second];

  float ratio;
  switch (mode) {
    case Mode::OVERRIDE:
      point.pos = pos;
      break;
    case Mode::MERGE:
      ratio = 1.0 / point.count;
      point.pos = (1.0f - ratio) * point.pos + ratio * pos;
      ++point.count;
      break;
    case Mode::DISCARD:
    default:
      return;
  }
}

void HashedCloud::erase(const std::function<bool(const Pos&)>& should_erase,
                        PosMap* erased) {
  std::vector<Entry> new_points;
  for (const auto& point : points_) {
    if (!should_erase(point.pos)) {
      lookup_[point.index] = new_points.size();
      new_points.push_back(point);
      continue;
    }

    lookup_.erase(point.index);
    if (erased) {
      erased->emplace(point.index, point.pos);
    }
  }

  points_ = std::move(new_points);
}

float HashedCloud::intersection(const HashedCloud& other) const {
  size_t num_equal = 0;
  for (const auto& [idx, _] : lookup_) {
    num_equal += other.lookup_.count(idx);
  }

  return static_cast<float>(num_equal) / lookup_.size();
}

float HashedCloud::iou(const HashedCloud& other) const {
  if (lookup_.empty() && other.lookup_.empty()) {
    return 0.0f;
  }

  size_t num_equal = 0;
  for (const auto& [idx, _] : lookup_) {
    num_equal += other.lookup_.count(idx);
  }

  size_t total = lookup_.size() + other.lookup_.size();
  return static_cast<float>(num_equal) / (total - num_equal);
}

}  // namespace hydra
