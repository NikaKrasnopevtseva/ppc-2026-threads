#include "krasnopevtseva_v_hoare_batcher_sort/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stack>
#include <utility>
#include <vector>

#include "krasnopevtseva_v_hoare_batcher_sort/common/include/common.hpp"

namespace krasnopevtseva_v_hoare_batcher_sort {

KrasnopevtsevaVHoareBatcherSortALL::KrasnopevtsevaVHoareBatcherSortALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool KrasnopevtsevaVHoareBatcherSortALL::ValidationImpl() {
  return !GetInput().empty();
}

bool KrasnopevtsevaVHoareBatcherSortALL::PreProcessingImpl() {
  input_data_ = GetInput();
  return true;
}

bool KrasnopevtsevaVHoareBatcherSortALL::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput() = output_data_;
  }
  return true;
}

int KrasnopevtsevaVHoareBatcherSortALL::Partition(std::vector<int> &arr, int left, int right) {
  int pivot = arr[left + (right - left) / 2];
  int i = left;
  int j = right;

  while (i <= j) {
    while (arr[i] < pivot) {
      ++i;
    }
    while (arr[j] > pivot) {
      --j;
    }
    if (i <= j) {
      std::swap(arr[i], arr[j]);
      ++i;
      --j;
    }
  }
  return i;
}

void KrasnopevtsevaVHoareBatcherSortALL::InsertionSort(std::vector<int> &arr, int left, int right) {
  for (int i = left + 1; i <= right; ++i) {
    int key = arr[i];
    int j = i - 1;
    while (j >= left && arr[j] > key) {
      arr[j + 1] = arr[j];
      --j;
    }
    arr[j + 1] = key;
  }
}

void KrasnopevtsevaVHoareBatcherSortALL::QuickSort(std::vector<int> &arr, int left, int right) {
  std::stack<std::pair<int, int>> stack;
  stack.emplace(left, right);

  while (!stack.empty()) {
    auto [l, r] = stack.top();
    stack.pop();

    while (l < r) {
      if (r - l < 16) {
        InsertionSort(arr, l, r);
        break;
      }

      int pivot_idx = Partition(arr, l, r);

      if (pivot_idx - l < r - pivot_idx + 1) {
        stack.emplace(pivot_idx, r);
        r = pivot_idx - 1;
      } else {
        stack.emplace(l, pivot_idx - 1);
        l = pivot_idx;
      }
    }
  }
}

void KrasnopevtsevaVHoareBatcherSortALL::BatcherMergeBlocksStep(int *left_ptr, int &left_size, int *right_ptr,
                                                                int &right_size) {
  std::inplace_merge(left_ptr, right_ptr, right_ptr + right_size);
  left_size += right_size;
}

void KrasnopevtsevaVHoareBatcherSortALL::BatcherMerge(std::vector<int *> &pointers, std::vector<int> &sizes) {
  int pack = static_cast<int>(pointers.size());
  for (int step = 1; pack > 1; step *= 2, pack /= 2) {
#pragma omp parallel for default(none) shared(pointers, sizes, pack, step)
    for (int off = 0; off < pack / 2; ++off) {
      size_t idx1 = static_cast<size_t>(2 * step) * static_cast<size_t>(off);
      size_t idx2 = idx1 + static_cast<size_t>(step);
      BatcherMergeBlocksStep(pointers[idx1], sizes[idx1], pointers[idx2], sizes[idx2]);
    }
    if ((pack / 2) - 1 == 0) {
      BatcherMergeBlocksStep(pointers[0], sizes[sizes.size() - 1], pointers[pointers.size() - 1],
                             sizes[sizes.size() - 1]);
    } else if ((pack / 2) % 2 != 0) {
      size_t idx1 = static_cast<size_t>(2 * step) * static_cast<size_t>((pack / 2) - 2);
      size_t idx2 = static_cast<size_t>(2 * step) * static_cast<size_t>((pack / 2) - 1);
      BatcherMergeBlocksStep(pointers[idx1], sizes[idx1], pointers[idx2], sizes[idx2]);
    }
  }
}

void KrasnopevtsevaVHoareBatcherSortALL::ParallelSortChunks(std::vector<int> &arr, int n, int num_threads) {
  int chunk_size = n / num_threads;
  int remainder = n % num_threads;

  std::vector<int *> pointers(num_threads);
  std::vector<int> sizes(num_threads);

  int offset = 0;
  for (int i = 0; i < num_threads; ++i) {
    pointers[i] = arr.data() + offset;
    sizes[i] = chunk_size + (i < remainder ? 1 : 0);
    offset += sizes[i];
  }

#pragma omp parallel for default(none) shared(arr, pointers, sizes, num_threads)
  for (int i = 0; i < num_threads; ++i) {
    int left = static_cast<int>(pointers[i] - arr.data());
    int right = left + sizes[i] - 1;
    QuickSort(arr, left, right);
  }

  BatcherMerge(pointers, sizes);
}

void KrasnopevtsevaVHoareBatcherSortALL::ParallelLocalSort(std::vector<int> &arr) {
  int n = static_cast<int>(arr.size());
  if (n <= 1) {
    return;
  }

  int max_threads = omp_get_max_threads();
  if (max_threads <= 0) {
    max_threads = 1;
  }

  int num_threads = std::min(max_threads, n);
  num_threads = std::max(1, num_threads);

  if (num_threads == 1 || n < 1000) {
    QuickSort(arr, 0, n - 1);
  } else {
    ParallelSortChunks(arr, n, num_threads);
  }
}

bool KrasnopevtsevaVHoareBatcherSortALL::RunImpl() {
  int rank = 0;
  int world_size = 1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int total_n = (rank == 0) ? static_cast<int>(input_data_.size()) : 0;
  MPI_Bcast(&total_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_n <= 1) {
    if (rank == 0) {
      output_data_ = input_data_;
    }
    return true;
  }

  if (world_size == 1 || total_n < 1000) {
    std::vector<int> result = input_data_;
    ParallelLocalSort(result);
    if (rank == 0) {
      output_data_ = std::move(result);
    }
    return true;
  }

  int chunk_size = (total_n + world_size - 1) / world_size;
  int padded_size = chunk_size * world_size;

  std::vector<int> local_data(static_cast<size_t>(chunk_size));
  std::vector<int> send_buffer;

  if (rank == 0) {
    send_buffer = input_data_;
    send_buffer.resize(static_cast<size_t>(padded_size), std::numeric_limits<int>::max());
  }

  MPI_Scatter(send_buffer.data(), chunk_size, MPI_INT, local_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

  ParallelLocalSort(local_data);

  std::vector<int> gather_buffer;
  if (rank == 0) {
    gather_buffer.resize(static_cast<size_t>(padded_size));
  }

  MPI_Gather(local_data.data(), chunk_size, MPI_INT, gather_buffer.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    gather_buffer.resize(static_cast<size_t>(total_n));
    std::sort(gather_buffer.begin(), gather_buffer.end());
    output_data_ = std::move(gather_buffer);
  }

  return true;
}

}  // namespace krasnopevtseva_v_hoare_batcher_sort
