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
  return true;
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

void KrasnopevtsevaVHoareBatcherSortALL::SplitPartition(std::vector<int> &arr, int &left, int &right, int &i, int &j) {
  int pivot = arr[left + ((right - left) / 2)];
  i = left;
  j = right;
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
}

void KrasnopevtsevaVHoareBatcherSortALL::ProcessPartition(std::vector<int> &arr, int &left, int &right,
                                                          std::vector<std::pair<int, int>> &stack) {
  int i = 0;
  int j = 0;
  SplitPartition(arr, left, right, i, j);

  if (j - left < right - i) {
    if (i < right) {
      stack.emplace_back(i, right);
    }
    right = j;
  } else {
    if (left < j) {
      stack.emplace_back(left, j);
    }
    left = i;
  }
}

void KrasnopevtsevaVHoareBatcherSortALL::QuickSort(std::vector<int> &arr, int left, int right) {
  if (left >= right) {
    return;
  }
  std::vector<std::pair<int, int>> stack;
  stack.reserve(64);
  stack.emplace_back(left, right);

  while (!stack.empty()) {
    auto [l, r] = stack.back();
    stack.pop_back();
    while (l < r) {
      ProcessPartition(arr, l, r, stack);
    }
  }
}

void KrasnopevtsevaVHoareBatcherSortALL::ParallelLocalSort(std::vector<int> &arr) {
  int size = static_cast<int>(arr.size());
  if (size <= 1) {
    return;
  }

  int max_threads = omp_get_max_threads();
  if (max_threads <= 0) {
    max_threads = 1;
  }

  int num_threads = 1;
  while (num_threads * 2 <= max_threads && num_threads * 2 <= size) {
    num_threads *= 2;
  }

  if (num_threads <= 1) {
    QuickSort(arr, 0, size - 1);
    return;
  }

#pragma omp parallel for default(none) shared(arr, num_threads, size)
  for (int i = 0; i < num_threads; ++i) {
    int chunk_size = size / num_threads;
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? size - 1 : (start + chunk_size - 1);
    QuickSort(arr, start, end);
  }

  QuickSort(arr, 0, size - 1);
}

void KrasnopevtsevaVHoareBatcherSortALL::CompareAndSwapWithNeighbor(std::vector<int> &local_arr, int neighbor,
                                                                    bool keep_low_half) {
  int size = static_cast<int>(local_arr.size());
  std::vector<int> neighbor_arr(static_cast<std::size_t>(size));

  MPI_Sendrecv(local_arr.data(), size, MPI_INT, neighbor, 0, neighbor_arr.data(), size, MPI_INT, neighbor, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<int> merged_arr(static_cast<std::size_t>(size) * 2);
  std::merge(local_arr.begin(), local_arr.end(), neighbor_arr.begin(), neighbor_arr.end(), merged_arr.begin());

  if (keep_low_half) {
    std::copy(merged_arr.begin(), merged_arr.begin() + size, local_arr.begin());
  } else {
    std::copy(merged_arr.begin() + size, merged_arr.end(), local_arr.begin());
  }
}

void KrasnopevtsevaVHoareBatcherSortALL::BatcherExchangeStep(std::vector<int> &local_data, int rank, int world_size,
                                                             int p_step, int k_step) {
  for (int j_idx = k_step % p_step; j_idx + k_step < world_size; j_idx += (k_step * 2)) {
    int upper_bound = std::min(k_step, world_size - j_idx - k_step);
    for (int i_idx = 0; i_idx < upper_bound; ++i_idx) {
      int r1 = j_idx + i_idx;
      int r2 = j_idx + i_idx + k_step;
      if ((r1 / (p_step * 2)) == (r2 / (p_step * 2))) {
        if (rank == r1) {
          CompareAndSwapWithNeighbor(local_data, r2, true);
        } else if (rank == r2) {
          CompareAndSwapWithNeighbor(local_data, r1, false);
        }
      }
    }
  }
}

void KrasnopevtsevaVHoareBatcherSortALL::BatcherMerge(std::vector<int> &local_data, int rank, int world_size) {
  for (int p_step = 1; p_step < world_size; p_step *= 2) {
    for (int k_step = p_step; k_step > 0; k_step /= 2) {
      BatcherExchangeStep(local_data, rank, world_size, p_step, k_step);
    }
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

  int chunk_size = (total_n + world_size - 1) / world_size;
  int padded_total_size = chunk_size * world_size;

  std::vector<int> local_data(static_cast<std::size_t>(chunk_size));
  std::vector<int> send_buffer;

  if (rank == 0) {
    send_buffer = input_data_;
    send_buffer.resize(static_cast<std::size_t>(padded_total_size), std::numeric_limits<int>::max());
  }

  MPI_Scatter(send_buffer.data(), chunk_size, MPI_INT, local_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

  ParallelLocalSort(local_data);

  BatcherMerge(local_data, rank, world_size);

  std::vector<int> gather_buffer;
  if (rank == 0) {
    gather_buffer.resize(static_cast<std::size_t>(padded_total_size));
  }
  MPI_Gather(local_data.data(), chunk_size, MPI_INT, gather_buffer.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    gather_buffer.resize(static_cast<std::size_t>(total_n));
    output_data_ = std::move(gather_buffer);
  }

  return true;
}

}  // namespace krasnopevtseva_v_hoare_batcher_sort
