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

  static int Partition(std::vector<int> &arr, int left, int right);
  static void OddEvenMerge(std::vector<int> &arr, int left, int right);
  static void SequentialSort(std::vector<int> &arr, int left, int right);
  static void ParallelSortImpl(std::vector<int> &arr, int left, int right);
  static void BuildScatterLayout(int n, int comm_size, std::vector<int> &counts, std::vector<int> &displs);
  static std::vector<int> Merge(const std::vector<int> &gathered, const std::vector<int> &counts,
                                const std::vector<int> &displs, int comm_size);

  std::vector<int> input_data_;
  std::vector<int> output_data_;
};
}  // namespace krasnopevtseva_v_hoare_batcher_sort
