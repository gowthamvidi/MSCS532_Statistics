
# selection_algorithms.py
# Deterministic BFPRT and Randomized Quickselect with 3-way partition.
from typing import List, Tuple
import random

def _partition_3way(a: List[int], left: int, right: int, pivot_index: int) -> Tuple[int, int]:
    pivot = a[pivot_index]
    a[pivot_index], a[right] = a[right], a[pivot_index]
    lt = left
    i = left
    gt = right
    while i <= gt:
        if a[i] < pivot:
            a[lt], a[i] = a[i], a[lt]
            lt += 1
            i += 1
        elif a[i] > pivot:
            a[i], a[gt] = a[gt], a[i]
            gt -= 1
        else:
            i += 1
    return lt, gt

def select_kth_randomized(arr: List[int], k: int) -> int:
    if not 1 <= k <= len(arr):
        raise ValueError("k out of range")
    left, right = 0, len(arr) - 1
    target = k - 1
    while True:
        if left == right:
            return arr[left]
        pivot_index = random.randrange(left, right + 1)
        lo, hi = _partition_3way(arr, left, right, pivot_index)
        if target < lo:
            right = lo - 1
        elif target > hi:
            left = hi + 1
        else:
            return arr[target]

def _median_of_five(a: List[int], left: int, right_exclusive: int) -> int:
    for i in range(left + 1, right_exclusive):
        x = a[i]; j = i - 1
        while j >= left and a[j] > x:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x
    length = right_exclusive - left
    median_offset = (length - 1) // 2
    return left + median_offset

def _choose_pivot_bfprt(a: List[int], left: int, right: int) -> int:
    n = right - left + 1
    if n <= 5:
        return _median_of_five(a, left, right + 1)
    write = left
    i = left
    while i <= right:
        j = min(i + 5, right + 1)
        m = _median_of_five(a, i, j)
        a[write], a[m] = a[m], a[write]
        write += 1
        i += 5
    mid = left + ((write - 1 - left) // 2)
    return _bfprt_select_index(a, left, write - 1, mid)

def _bfprt_select_index(a: List[int], left: int, right: int, target_index: int) -> int:
    while True:
        if left == right:
            return left
        pivot_index = _choose_pivot_bfprt(a, left, right)
        lo, hi = _partition_3way(a, left, right, pivot_index)
        if target_index < lo:
            right = lo - 1
        elif target_index > hi:
            left = hi + 1
        else:
            return target_index

def select_kth_deterministic(arr: List[int], k: int) -> int:
    if not 1 <= k <= len(arr):
        raise ValueError("k out of range")
    idx = _bfprt_select_index(arr, 0, len(arr) - 1, k - 1)
    return arr[idx]

if __name__ == "__main__":
    data = [9, 1, 5, 3, 7, 2, 8, 6, 4, 4, 4]
    for k in [1, 3, 5, 6, 9, 11]:
        a = data.copy(); b = data.copy()
        assert select_kth_randomized(a, k) == sorted(data)[k-1]
        assert select_kth_deterministic(b, k) == sorted(data)[k-1]
    print("Self-check passed.")
