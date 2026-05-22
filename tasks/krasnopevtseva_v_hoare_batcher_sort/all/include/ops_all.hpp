#pragma once

#include <vector>

#include "krasnopevtseva_v_hoare_batcher_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krasnopevtseva_v_hoare_batcher_sort {

class KrasnopevtsevaVHoareBatcherSortALL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kALL;
  }
  explicit KrasnopevtsevaVHoareBatcherSortALL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> input_data_;
  std::vector<int> output_data_;

  static void SplitPartition(std::vector<int> &arr, int &left, int &right, int &i, int &j);
  static void ProcessPartition(std::vector<int> &arr, int &left, int &right, std::vector<std::pair<int, int>> &stack);
  static void QuickSort(std::vector<int> &arr, int left, int right);
  static void ParallelLocalSort(std::vector<int> &arr);
  static void CompareAndSwapWithNeighbor(std::vector<int> &local_arr, int neighbor, bool keep_low_half);
  static void BatcherExchangeStep(std::vector<int> &local_data, int rank, int world_size, int p_step, int k_step);
  static void BatcherMerge(std::vector<int> &local_data, int rank, int world_size);
};
}  // namespace krasnopevtseva_v_hoare_batcher_sort
