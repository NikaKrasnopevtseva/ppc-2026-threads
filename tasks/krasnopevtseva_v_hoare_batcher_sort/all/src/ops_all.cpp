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
  int initialized = 0;

  MPI_Initialized(&initialized);

  if (!initialized) {
    MPI_Init(nullptr, nullptr);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_size);

  const auto &input = GetInput();
  int n = static_cast<int>(input.size());

  if (n <= 1) {
    if (rank == 0) {
      GetOutput().resize(input.size());
      std::copy(input.begin(), input.end(), GetOutput().begin());
    }
    if (!initialized) {
      MPI_Finalize();
    }
    return true;
  }

  int local_n = n / proc_size;
  int remainder = n % proc_size;
  std::vector<int> send_counts(proc_size);
  std::vector<int> displs(proc_size);

  for (int i = 0; i < proc_size; ++i) {
    send_counts[i] = local_n + (i < remainder ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  int my_n = send_counts[rank];
  std::vector<int> local_data(my_n);

  if (rank == 0) {
    std::vector<int> temp = input;
    MPI_Scatterv(temp.data(), send_counts.data(), displs.data(), MPI_INT, local_data.data(), my_n, MPI_INT, 0,
                 MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, local_data.data(), my_n, MPI_INT, 0, MPI_COMM_WORLD);
  }

  SortLocalData(local_data);

  std::vector<int> global_data;
  if (rank == 0) {
    global_data.resize(static_cast<size_t>(n));
  }

  MPI_Gatherv(local_data.data(), my_n, MPI_INT, global_data.data(), send_counts.data(), displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<std::vector<int>> parts;
    parts.reserve(static_cast<size_t>(proc_size));

    for (int i = 0; i < proc_size; ++i) {
      int cnt = send_counts[i];
      int start = displs[i];
      if (cnt > 0 && start + cnt <= n) {
        parts.emplace_back(global_data.begin() + start, global_data.begin() + start + cnt);
      }
    }

    if (!parts.empty()) {
      std::vector<int> result = std::move(parts[0]);
      for (size_t i = 1; i < parts.size(); ++i) {
        std::vector<int> merged;
        merged.reserve(result.size() + parts[i].size());

        size_t i1 = 0;
        size_t i2 = 0;
        while (i1 < result.size() && i2 < parts[i].size()) {
          if (result[i1] <= parts[i][i2]) {
            merged.push_back(result[i1]);
            ++i1;
          } else {
            merged.push_back(parts[i][i2]);
            ++i2;
          }
        }
        while (i1 < result.size()) {
          merged.push_back(result[i1]);
          ++i1;
        }
        while (i2 < parts[i].size()) {
          merged.push_back(parts[i][i2]);
          ++i2;
        }
        result = std::move(merged);
      }
      global_data = std::move(result);
    }

    GetOutput().resize(global_data.size());
    std::copy(global_data.begin(), global_data.end(), GetOutput().begin());
  }

  if (!initialized) {
    MPI_Finalize();
  }

  return true;
}

}  // namespace krasnopevtseva_v_hoare_batcher_sort
