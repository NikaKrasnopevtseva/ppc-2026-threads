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

  static void QuickSort(std::vector<int> &arr, int left, int right);
  static void InsertionSort(std::vector<int> &arr, int left, int right);
  static int Partition(std::vector<int> &arr, int left, int right);
  static void ParallelLocalSort(std::vector<int> &arr);
  static void BatcherMerge(std::vector<int *> &pointers, std::vector<int> &sizes);
  static void BatcherMergeBlocksStep(int *left_ptr, int &left_size, int *right_ptr, int &right_size);
  static void ParallelSortChunks(std::vector<int> &arr, int n, int num_threads);

  std::vector<int> input_data_;
  std::vector<int> output_data_;
};
}  // namespace krasnopevtseva_v_hoare_batcher_sort
