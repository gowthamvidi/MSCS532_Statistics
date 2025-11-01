# Medians and Order Statistics & Elementary Data Structures

This repository contains two Python programs for the assignment.

---

## Part 1 – Selection Algorithms (Order Statistics)

**File:** `selection_algorithms.py`

### Description
Implements algorithms to find the k-th smallest element in an unsorted array:
- **Randomized Quickselect** – Expected O(n)
- **Deterministic Median-of-Medians (BFPRT)** – Worst-case O(n)
- Handles duplicate elements using a 3-way partitioning approach.

### Run Command
```bash
python selection_algorithms.py
```


##  Part 2 – Elementary Data Structures

**File:** `data_structures.py`

### Description
Implements the following data structures completely from scratch:

| Data Structure | Core Operations |
|----------------|----------------|
| **DynamicArray** | insert, delete, access, append |
| **Matrix** | access/set, insert/delete rows and columns |
| **StackArray** | push, pop, peek, is_empty |
| **QueueArray** | enqueue, dequeue, peek, auto-resize circular buffer |
| **SinglyLinkedList** | insert_head, insert_tail, delete_value, traverse |
| **TreeNode** *(optional)* | add_child, DFS, BFS traversals |

### Run Command
```bash
python data_structures.py
```

---

## Requirements
- Python 3.8 or above  
- No external dependencies (uses only the Python standard library)

---

## File Summary
| Part | Python File | Purpose |
|------|--------------|----------|
| 1 | `selection_algorithms.py` | Randomized and deterministic selection algorithms |  
| 2 | `data_structures.py` | Arrays, stacks, queues, linked lists, and trees |

---
