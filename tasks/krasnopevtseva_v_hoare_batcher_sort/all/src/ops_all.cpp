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
  GetOutput() = std::vector<int>();
  return true;
}

bool KrasnopevtsevaVHoareBatcherSortALL::PostProcessingImpl() {
  return true;
}

int KrasnopevtsevaVHoareBatcherSortALL::Partition(std::vector<int> &arr, int first, int last) {
  int i = first - 1;
  int value = arr[last];

  for (int j = first; j <= last - 1; ++j) {
    if (arr[j] <= value) {
      ++i;
      std::swap(arr[i], arr[j]);
    }
  }
  std::swap(arr[i + 1], arr[last]);
  return i + 1;
}

void KrasnopevtsevaVHoareBatcherSortALL::InsertionSort(std::vector<int> &arr, int first, int last) {
  for (int i = first + 1; i <= last; ++i) {
    int key = arr[i];
    int j = i - 1;
    while (j >= first && arr[j] > key) {
      arr[j + 1] = arr[j];
      --j;
    }
    arr[j + 1] = key;
  }
}

void KrasnopevtsevaVHoareBatcherSortALL::QuickSort(std::vector<int> &arr, int first, int last) {
  std::stack<std::pair<int, int>> stack;
  stack.emplace(first, last);

  while (!stack.empty()) {
    auto [l, r] = stack.top();
    stack.pop();

    if (l >= r) {
      continue;
    }

    if (r - l < 16) {
      InsertionSort(arr, l, r);
      continue;
    }

    int iter = Partition(arr, l, r);

    if (iter - l < r - iter) {
      stack.emplace(iter + 1, r);
      stack.emplace(l, iter - 1);
    } else {
      stack.emplace(l, iter - 1);
      stack.emplace(iter + 1, r);
    }
  }
}

void KrasnopevtsevaVHoareBatcherSortALL::BatcherMergeBlocksStep(int *left_pointer, int &left_size, int *right_pointer,
                                                                int &right_size) {
  std::inplace_merge(left_pointer, right_pointer, right_pointer + right_size);
  left_size += right_size;
}

void KrasnopevtsevaVHoareBatcherSortALL::BatcherMerge(int thread_input_size, std::vector<int *> &pointers,
                                                      std::vector<int> &sizes, int par_if_greater) {
  int pack = static_cast<int>(pointers.size());
  for (int step = 1; pack > 1; step *= 2, pack /= 2) {
#pragma omp parallel for default(none) shared(pointers, sizes, pack, step, thread_input_size, \
                                                  par_if_greater) if ((thread_input_size / step) > par_if_greater)
    for (int off = 0; off < pack / 2; ++off) {
      auto idx1 = static_cast<std::size_t>(2 * step) * static_cast<std::size_t>(off);
      auto idx2 = idx1 + static_cast<std::size_t>(step);
      BatcherMergeBlocksStep(pointers[idx1], sizes[idx1], pointers[idx2], sizes[idx2]);
    }
    if ((pack / 2) - 1 == 0) {
      BatcherMergeBlocksStep(pointers[0], sizes[sizes.size() - 1], pointers[pointers.size() - 1],
                             sizes[sizes.size() - 1]);
    } else if ((pack / 2) % 2 != 0) {
      auto idx1 = static_cast<std::size_t>(2 * step) * static_cast<std::size_t>((pack / 2) - 2);
      auto idx2 = static_cast<std::size_t>(2 * step) * static_cast<std::size_t>((pack / 2) - 1);
      BatcherMergeBlocksStep(pointers[idx1], sizes[idx1], pointers[idx2], sizes[idx2]);
    }
  }
}

void KrasnopevtsevaVHoareBatcherSortALL::ParallelSortChunksOpenMP(std::vector<int> &res, int n, int numthreads) {
  if (n <= 0) {
    return;
  }

  numthreads = std::min(n, numthreads);
  if (numthreads <= 0) {
    numthreads = 1;
  }

  int thread_input_size = n / numthreads;
  int thread_input_remainder_size = n % numthreads;

  std::vector<int *> pointers(numthreads);
  std::vector<int> sizes(numthreads);

  for (int i = 0; i < numthreads; ++i) {
    std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(thread_input_size);
    pointers[i] = res.data() + offset;
    sizes[i] = thread_input_size;
  }
  sizes.back() += thread_input_remainder_size;

#pragma omp parallel for default(none) shared(res, pointers, sizes, numthreads)
  for (int i = 0; i < numthreads; ++i) {
    int left = static_cast<int>(pointers[i] - res.data());
    int right = left + sizes[i] - 1;
    if (left < right) {
      QuickSort(res, left, right);
    }
  }

  BatcherMerge(thread_input_size, pointers, sizes, 32);
}

void KrasnopevtsevaVHoareBatcherSortALL::SortLocalData(std::vector<int> &data) {
  int n = static_cast<int>(data.size());
  if (n <= 0) {
    return;
  }

  int numthreads = omp_get_max_threads();
  numthreads = std::min(n, numthreads);

  if (n < 10000) {
    QuickSort(data, 0, n - 1);
  } else {
    ParallelSortChunksOpenMP(data, n, numthreads);
  }
}

bool KrasnopevtsevaVHoareBatcherSortALL::RunImpl() {
  int rank = 0;
  int proc_size = 1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_size);

  const auto &input = GetInput();
  int n = static_cast<int>(input.size());

  if (proc_size == 1 || n <= 10000) {
    std::vector<int> result = input;
    SortLocalData(result);
    if (rank == 0) {
      GetOutput() = std::move(result);
    }
    return true;
  }

  int local_n = n / proc_size;

  if (rank == 0) {
    std::vector<std::vector<int>> all_parts(proc_size);
    int idx = 0;
    for (int i = 0; i < proc_size; ++i) {
      int cnt = local_n + (i < n % proc_size ? 1 : 0);
      all_parts[i].assign(input.begin() + idx, input.begin() + idx + cnt);
      idx += cnt;
    }

    for (int i = 1; i < proc_size; ++i) {
      MPI_Send(all_parts[i].data(), static_cast<int>(all_parts[i].size()), MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    std::vector<int> local_data = all_parts[0];
    SortLocalData(local_data);
    all_parts[0] = local_data;

    for (int i = 1; i < proc_size; ++i) {
      int cnt = local_n + (i < n % proc_size ? 1 : 0);
      std::vector<int> recv_data(cnt);
      MPI_Recv(recv_data.data(), cnt, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      all_parts[i] = recv_data;
    }

    std::vector<int> result = all_parts[0];
    for (int i = 1; i < proc_size; ++i) {
      std::vector<int> merged;
      merged.reserve(result.size() + all_parts[i].size());

      size_t i1 = 0, i2 = 0;
      while (i1 < result.size() && i2 < all_parts[i].size()) {
        if (result[i1] <= all_parts[i][i2]) {
          merged.push_back(result[i1++]);
        } else {
          merged.push_back(all_parts[i][i2++]);
        }
      }
      while (i1 < result.size()) {
        merged.push_back(result[i1++]);
      }
      while (i2 < all_parts[i].size()) {
        merged.push_back(all_parts[i][i2++]);
      }

      result = std::move(merged);
    }

    GetOutput() = std::move(result);

  } else {
    int cnt = local_n + (rank < n % proc_size ? 1 : 0);
    std::vector<int> local_data(cnt);
    MPI_Recv(local_data.data(), cnt, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    SortLocalData(local_data);

    MPI_Send(local_data.data(), static_cast<int>(local_data.size()), MPI_INT, 0, 1, MPI_COMM_WORLD);
  }

  return true;
}

}  // namespace krasnopevtseva_v_hoare_batcher_sort
