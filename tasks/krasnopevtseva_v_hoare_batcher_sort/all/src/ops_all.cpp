#include "krasnopevtseva_v_hoare_batcher_sort/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <queue>
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
  return true;
}

bool KrasnopevtsevaVHoareBatcherSortALL::PostProcessingImpl() {
  return true;
}

int KrasnopevtsevaVHoareBatcherSortALL::Partition(std::vector<int> &arr, int left, int right) {
  int pivot = arr[left + ((right - left) / 2)];
  int i = left - 1;
  int j = right + 1;
  while (true) {
    ++i;
    while (arr[i] < pivot) {
      ++i;
    }
    --j;
    while (arr[j] > pivot) {
      --j;
    }
    if (i >= j) {
      return j;
    }
    std::swap(arr[i], arr[j]);
  }
}

void KrasnopevtsevaVHoareBatcherSortALL::OddEvenMerge(std::vector<int> &arr, int left, int right) {
  int n = right - left + 1;
  for (int step = 1; step < n; step *= 2) {
    for (int i = left; i + step <= right; i += step * 2) {
      if (arr[i] > arr[i + step]) {
        std::swap(arr[i], arr[i + step]);
      }
    }
  }
}

void KrasnopevtsevaVHoareBatcherSortALL::SequentialSort(std::vector<int> &arr, int left, int right) {
  std::stack<std::pair<int, int>> stack;
  stack.emplace(left, right);
  while (!stack.empty()) {
    auto [l, r] = stack.top();
    stack.pop();
    if (l >= r) {
      continue;
    }
    int p = Partition(arr, l, r);
    if ((p - l) > (r - p - 1)) {
      stack.emplace(l, p);
      stack.emplace(p + 1, r);
    } else {
      stack.emplace(p + 1, r);
      stack.emplace(l, p);
    }
    OddEvenMerge(arr, l, r);
  }
}

void KrasnopevtsevaVHoareBatcherSortALL::ParallelSortImpl(std::vector<int> &arr, int left, int right) {
  if (left >= right) {
    return;
  }
  if (right - left < 1000) {
    SequentialSort(arr, left, right);
    return;
  }
  int p = Partition(arr, left, right);
  OddEvenMerge(arr, left, right);
#pragma omp task default(none) shared(arr) firstprivate(left, p)
  ParallelSortImpl(arr, left, p);
#pragma omp task default(none) shared(arr) firstprivate(right, p)
  ParallelSortImpl(arr, p + 1, right);
#pragma omp taskwait
}

void KrasnopevtsevaVHoareBatcherSortALL::BuildScatterLayout(int n, int comm_size, std::vector<int> &counts,
                                                            std::vector<int> &displs) {
  counts.resize(static_cast<size_t>(comm_size));
  displs.resize(static_cast<size_t>(comm_size));
  const int base = n / comm_size;
  const int rem = n % comm_size;
  int offset = 0;
  for (int i = 0; i < comm_size; ++i) {
    counts[static_cast<size_t>(i)] = base + (i < rem ? 1 : 0);
    displs[static_cast<size_t>(i)] = offset;
    offset += counts[static_cast<size_t>(i)];
  }
}

std::vector<int> KrasnopevtsevaVHoareBatcherSortALL::Merge(const std::vector<int> &gathered,
                                                           const std::vector<int> &counts,
                                                           const std::vector<int> &displs, int comm_size) {
  struct Item {
    int val;
    int chunk;
    int next_idx;
  };
  auto cmp = [](const Item &a, const Item &b) { return a.val > b.val; };
  std::priority_queue<Item, std::vector<Item>, decltype(cmp)> pq(cmp);
  for (int i = 0; i < comm_size; ++i) {
    int cnt = counts[static_cast<size_t>(i)];
    if (cnt <= 0) {
      continue;
    }
    int dsp = displs[static_cast<size_t>(i)];
    pq.push(Item{.val = gathered[static_cast<size_t>(dsp)], .chunk = i, .next_idx = 1});
  }
  std::vector<int> result;
  result.reserve(gathered.size());
  while (!pq.empty()) {
    Item item = pq.top();
    pq.pop();
    result.push_back(item.val);
    int cnt = counts[static_cast<size_t>(item.chunk)];
    if (item.next_idx < cnt) {
      int dsp = displs[static_cast<size_t>(item.chunk)];
      size_t idx = static_cast<size_t>(dsp) + static_cast<size_t>(item.next_idx);
      pq.push(Item{.val = gathered[idx], .chunk = item.chunk, .next_idx = item.next_idx + 1});
    }
  }
  return result;
}

bool KrasnopevtsevaVHoareBatcherSortALL::RunImpl() {
  int rank = 0;
  int comm_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int n = 0;
  std::vector<int> input_vector;

  if (rank == 0) {
    const auto &input = GetInput();
    n = static_cast<int>(input.size());
    input_vector = input;
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 1) {
    if (rank == 0) {
      GetOutput() = input_vector;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  std::vector<int> counts;
  std::vector<int> displs;
  BuildScatterLayout(n, comm_size, counts, displs);

  int local_n = counts[static_cast<size_t>(rank)];
  std::vector<int> local_data(static_cast<size_t>(local_n));

  const int *send_root = (rank == 0) ? input_vector.data() : nullptr;
  MPI_Scatterv(send_root, counts.data(), displs.data(), MPI_INT, local_data.data(), local_n, MPI_INT, 0,
               MPI_COMM_WORLD);

  if (local_n > 1) {
#pragma omp parallel default(none) shared(local_data, local_n)
#pragma omp single
    ParallelSortImpl(local_data, 0, local_n - 1);
  }

  std::vector<int> gathered;
  int *recv_root = nullptr;
  if (rank == 0) {
    gathered.resize(static_cast<size_t>(n));
    recv_root = gathered.data();
  }
  MPI_Gatherv(local_data.data(), local_n, MPI_INT, recv_root, counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<int> result = Merge(gathered, counts, displs, comm_size);
    std::sort(result.begin(), result.end());
    GetOutput() = std::move(result);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

}  // namespace krasnopevtseva_v_hoare_batcher_sort
