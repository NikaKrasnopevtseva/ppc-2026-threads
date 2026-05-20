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
  struct Chunk {
    int *ptr;
    int size;
    int left;
    int right;
  };

  static int Partition(std::vector<int> &arr, int first, int last);
  static void InsertionSort(std::vector<int> &arr, int first, int last);
  static void QuickSort(std::vector<int> &arr, int first, int last);
  static void BatcherMergeBlocksStep(int *left_pointer, int &left_size, int *right_pointer, int &right_size);
  static void BatcherMerge(int thread_input_size, std::vector<int *> &pointers, std::vector<int> &sizes,
                           int par_if_greater);
  static void ParallelSortChunksOpenMP(std::vector<int> &res, int n, int numthreads);
  static void MergeSortedVectors(std::vector<int> &result, const std::vector<int> &other);

  static void ComputeDistribution(int n, int size, std::vector<int> &send_counts, std::vector<int> &displs);
  static void ScatterData(const std::vector<int> &input, std::vector<int> &local_data, int rank, int size);
  static void SortLocalData(std::vector<int> &local_data);
  static void GatherData(std::vector<int> &local_data, std::vector<int> &global_data, int rank, int size, int n);
  static void FinalMerge(std::vector<int> &global_data, const std::vector<int> &send_counts,
                         const std::vector<int> &displs, int size);

  int rank_ = 0;
  int proc_size_ = 1;
  bool mpi_initialized_ = false;
};
}  // namespace krasnopevtseva_v_hoare_batcher_sort
