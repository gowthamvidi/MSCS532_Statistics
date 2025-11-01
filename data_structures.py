
"""
data_structures.py
Implementations of elementary data structures in pure Python:
- DynamicArray (array-like wrapper for didactic insert/delete/access)
- Matrix (2D array) with basic row/col operations
- StackArray (stack on top of a Python list)
- QueueArray (circular-buffer queue using a fixed-size list that resizes when full)
- SinglyLinkedList (with head/tail, insertions, deletions, traversal)
- TreeNode (rooted general tree with DFS/BFS traversals)
"""

from typing import Any, Iterable, Optional, List

class DynamicArray:
    """A minimal dynamic array abstraction over Python list for educational purposes."""
    def __init__(self, data: Optional[Iterable[Any]] = None):
        self._a: List[Any] = list(data) if data is not None else []

    def __len__(self) -> int:
        return len(self._a)

    def __getitem__(self, idx: int) -> Any:
        return self._a[idx]

    def __setitem__(self, idx: int, val: Any) -> None:
        self._a[idx] = val

    def access(self, idx: int) -> Any:
        return self._a[idx]

    def insert(self, idx: int, val: Any) -> None:
        """Insert at index (0..len). Amortized O(n) due to shifting."""
        if idx < 0 or idx > len(self._a):
            raise IndexError("index out of bounds")
        self._a.append(None)          # grow by one
        for i in range(len(self._a) - 1, idx, -1):
            self._a[i] = self._a[i-1] # shift right
        self._a[idx] = val

    def append(self, val: Any) -> None:
        self._a.append(val)           # amortized O(1) on CPython

    def delete(self, idx: int) -> Any:
        """Delete and return element at idx. O(n) due to shifting."""
        if idx < 0 or idx >= len(self._a):
            raise IndexError("index out of bounds")
        val = self._a[idx]
        for i in range(idx, len(self._a) - 1):
            self._a[i] = self._a[i+1]
        self._a.pop()
        return val

    def to_list(self) -> List[Any]:
        return list(self._a)

class Matrix:
    """Simple 2D matrix using a list of lists (row-major)."""
    def __init__(self, rows: int, cols: int, fill: Any = 0):
        if rows < 0 or cols < 0:
            raise ValueError("rows/cols must be non-negative")
        self._m: List[List[Any]] = [[fill for _ in range(cols)] for __ in range(rows)]

    @property
    def shape(self):
        return (len(self._m), len(self._m[0]) if self._m else 0)

    def access(self, r: int, c: int) -> Any:
        return self._m[r][c]

    def set(self, r: int, c: int, val: Any) -> None:
        self._m[r][c] = val

    def insert_row(self, r: int, fill: Any = 0) -> None:
        rows, cols = self.shape
        if r < 0 or r > rows:
            raise IndexError("row out of bounds")
        self._m.insert(r, [fill for _ in range(cols)])

    def delete_row(self, r: int) -> List[Any]:
        return self._m.pop(r)

    def insert_col(self, c: int, fill: Any = 0) -> None:
        rows, cols = self.shape
        if c < 0 or c > cols:
            raise IndexError("col out of bounds")
        for r in range(rows):
            self._m[r].insert(c, fill)

    def delete_col(self, c: int) -> List[Any]:
        rows, cols = self.shape
        if c < 0 or c >= cols:
            raise IndexError("col out of bounds")
        removed = []
        for r in range(rows):
            removed.append(self._m[r].pop(c))
        return removed

    def to_list(self) -> List[List[Any]]:
        return [row[:] for row in self._m]

class StackArray:
    """Stack built on Python list with push/pop/peek operations."""
    def __init__(self):
        self._a: List[Any] = []

    def push(self, x: Any) -> None:
        self._a.append(x)  # O(1) amortized

    def pop(self) -> Any:
        if not self._a:
            raise IndexError("pop from empty stack")
        return self._a.pop()  # O(1)

    def peek(self) -> Any:
        if not self._a:
            raise IndexError("peek from empty stack")
        return self._a[-1]

    def is_empty(self) -> bool:
        return not self._a

    def __len__(self) -> int:
        return len(self._a)

class QueueArray:
    """Circular-buffer queue with automatic resize when full."""
    def __init__(self, capacity: int = 8):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._a: List[Any] = [None] * capacity
        self._head = 0
        self._tail = 0
        self._size = 0

    def _resize(self, new_cap: int) -> None:
        b = [None] * new_cap
        for i in range(self._size):
            b[i] = self._a[(self._head + i) % len(self._a)]
        self._a = b
        self._head = 0
        self._tail = self._size

    def enqueue(self, x: Any) -> None:
        if self._size == len(self._a):
            self._resize(len(self._a) * 2)
        self._a[self._tail] = x
        self._tail = (self._tail + 1) % len(self._a)
        self._size += 1

    def dequeue(self) -> Any:
        if self._size == 0:
            raise IndexError("dequeue from empty queue")
        x = self._a[self._head]
        self._a[self._head] = None
        self._head = (self._head + 1) % len(self._a)
        self._size -= 1
        # Optional shrink
        if 0 < self._size <= len(self._a) // 4 and len(self._a) > 8:
            self._resize(len(self._a) // 2)
        return x

    def peek(self) -> Any:
        if self._size == 0:
            raise IndexError("peek from empty queue")
        return self._a[self._head]

    def is_empty(self) -> bool:
        return self._size == 0

    def __len__(self) -> int:
        return self._size

class SinglyLinkedList:
    class _Node:
        __slots__ = ("val", "next")
        def __init__(self, val: Any, next: Optional["SinglyLinkedList._Node"] = None):
            self.val = val
            self.next = next

    def __init__(self):
        self._head: Optional[SinglyLinkedList._Node] = None
        self._tail: Optional[SinglyLinkedList._Node] = None
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def insert_head(self, x: Any) -> None:
        node = self._Node(x, self._head)
        self._head = node
        if self._tail is None:
            self._tail = node
        self._size += 1

    def insert_tail(self, x: Any) -> None:
        node = self._Node(x, None)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._size += 1

    def delete_value(self, x: Any) -> bool:
        """Delete first occurrence of x; return True if deleted."""
        prev = None
        cur = self._head
        while cur:
            if cur.val == x:
                if prev is None:
                    self._head = cur.next
                else:
                    prev.next = cur.next
                if cur is self._tail:
                    self._tail = prev
                self._size -= 1
                return True
            prev, cur = cur, cur.next
        return False

    def traverse(self) -> list:
        out = []
        cur = self._head
        while cur:
            out.append(cur.val)
            cur = cur.next
        return out

class TreeNode:
    """Rooted general tree node (each node has 0..k children)."""
    def __init__(self, val: Any):
        self.val = val
        self.children: List["TreeNode"] = []

    def add_child(self, child: "TreeNode") -> None:
        self.children.append(child)

    def dfs(self) -> List[Any]:
        """Preorder DFS traversal."""
        out = [self.val]
        for ch in self.children:
            out.extend(ch.dfs())
        return out

    def bfs(self) -> List[Any]:
        """Level-order BFS traversal (using QueueArray)."""
        order = []
        q = QueueArray()
        q.enqueue(self)
        while not q.is_empty():
            node = q.dequeue()
            order.append(node.val)
            for ch in node.children:
                q.enqueue(ch)
        return order

# -------------- Minimal self-checks --------------
if __name__ == "__main__":
    # DynamicArray
    a = DynamicArray([1,2,3])
    a.insert(1, 99)
    assert a.to_list() == [1,99,2,3]
    a.delete(2)
    assert a.to_list() == [1,99,3]

    # Matrix
    m = Matrix(2,3,0)
    m.set(0,1,7)
    m.insert_row(1, fill=-1)
    m.insert_col(0, fill=5)
    _ = m.delete_col(0)
    _ = m.delete_row(1)

    # Stack
    s = StackArray()
    s.push(10); s.push(20)
    assert s.pop() == 20
    assert s.peek() == 10

    # Queue
    q = QueueArray(2)
    q.enqueue(1); q.enqueue(2); q.enqueue(3)
    assert q.dequeue() == 1
    assert q.peek() == 2

    # Linked list
    ll = SinglyLinkedList()
    ll.insert_head(1); ll.insert_tail(2); ll.insert_tail(3)
    assert ll.traverse() == [1,2,3]
    assert ll.delete_value(2) is True
    assert ll.traverse() == [1,3]

    # Tree
    root = TreeNode("A")
    b = TreeNode("B"); c = TreeNode("C"); d = TreeNode("D")
    root.add_child(b); root.add_child(c)
    c.add_child(d)
    assert root.dfs() == ["A","B","C","D"]
    assert root.bfs() == ["A","B","C","D"]
    print("Self-checks passed.")
