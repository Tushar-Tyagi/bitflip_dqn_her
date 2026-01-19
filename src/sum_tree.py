"""
Sum Tree Data Structure for Prioritized Sampling
"""

import numpy as np


class SumTree:
    """
    Binary tree data structure where parent node = sum of children.
    
    Tree structure:
                    ___Total___
                   /           \
              __Sum1__      __Sum2__
             /        \    /        \
         Priority1  Priority2  Priority3  Priority4
            [Data1]   [Data2]   [Data3]   [Data4]
    """
    
    def __init__(self, capacity: int):
        """
        Initialize sum tree.
        
        Args:
            capacity: Maximum number of leaf nodes (data points)
        """
        self.capacity = capacity
        self.write_index = 0
        self.full = False
        
        # Tree structure: [parent nodes, leaf nodes]
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        
        # Data array to store actual transitions (indices only)
        self.data = np.zeros(capacity, dtype=np.int32)
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """
        Propagate priority change up the tree.
        
        Args:
            idx: Index in tree array
            change: Change in priority value
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, value: float) -> int:
        """
        Retrieve leaf index corresponding to a priority value.
        
        Args:
            idx: Current node index
            value: Target cumulative priority value
            
        Returns:
            Leaf node index
        """
        left = 2 * idx + 1
        right = left + 1
        
        # If we've reached a leaf node
        if left >= len(self.tree):
            return idx
        
        # Traverse left if value is less than left child sum
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            # Traverse right, subtracting left child sum from value
            return self._retrieve(right, value - self.tree[left])
    
    def total(self) -> float:
        """Return total priority (root node value)."""
        return self.tree[0]
    
    def add(self, priority: float, data_idx: int):
        """
        Add new data with given priority.
        
        Args:
            priority: Priority value
            data_idx: Index of data in external storage
        """
        # Leaf nodes start at index (capacity - 1)
        tree_idx = self.write_index + self.capacity - 1
        
        # Store data index
        self.data[self.write_index] = data_idx
        
        # Update priority
        self.update(tree_idx, priority)
        
        # Move write index
        self.write_index += 1
        if self.write_index >= self.capacity:
            self.write_index = 0
            self.full = True
        
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, tree_idx: int, priority: float):
        """
        Update priority of a leaf node.
        
        Args:
            tree_idx: Index in tree array
            priority: New priority value
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
    
    def get(self, value: float) -> tuple:
        """
        Get leaf node index and data index for a priority value.
        
        Args:
            value: Target cumulative priority value
            
        Returns:
            (tree_idx, data_idx, priority)
        """
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        
        return tree_idx, self.data[data_idx], self.tree[tree_idx]
    
    def get_priority(self, data_idx: int) -> float:
        """
        Get priority for a specific data index.
        
        Args:
            data_idx: Index in data array
            
        Returns:
            Priority value
        """
        tree_idx = data_idx + self.capacity - 1
        return self.tree[tree_idx]
    
    def __len__(self) -> int:
        """Return number of stored entries."""
        return self.n_entries


class MinTree:
    """
    Similar to SumTree but maintains minimum values.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize min tree.
        
        Args:
            capacity: Maximum number of leaf nodes
        """
        self.capacity = capacity
        self.tree = np.full(2 * capacity - 1, float('inf'), dtype=np.float32)
    
    def _propagate_min(self, idx: int):
        """Propagate minimum value up the tree."""
        if idx == 0:
            return
        
        parent = (idx - 1) // 2
        left = 2 * parent + 1
        right = left + 1
        
        if right < len(self.tree):
            self.tree[parent] = min(self.tree[left], self.tree[right])
        else:
            self.tree[parent] = self.tree[left]
        
        self._propagate_min(parent)
    
    def update(self, data_idx: int, value: float):
        """
        Update value at leaf node.
        
        Args:
            data_idx: Index in data array
            value: New value
        """
        tree_idx = data_idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate_min(tree_idx)
    
    def min(self) -> float:
        """Return minimum value (root node)."""
        return self.tree[0]


def test_sum_tree():
    """Test sum tree functionality."""
    print("Testing SumTree...")
    
    capacity = 8
    tree = SumTree(capacity)
    
    # Add priorities
    priorities = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    for i, p in enumerate(priorities):
        tree.add(p, i)
    
    print(f"Total priority: {tree.total()}")
    print(f"Expected: {sum(priorities)}")
    
    # Sample proportionally
    print("\nSampling test:")
    samples = []
    for _ in range(1000):
        value = np.random.uniform(0, tree.total())
        tree_idx, data_idx, priority = tree.get(value)
        samples.append(data_idx)
    
    # Check distribution
    unique, counts = np.unique(samples, return_counts=True)
    print("Sample distribution (should be proportional to priorities):")
    for idx, count in zip(unique, counts):
        expected_ratio = priorities[idx] / sum(priorities)
        actual_ratio = count / 1000
        print(f"  Data {idx} (priority={priorities[idx]}): "
              f"{count}/1000 = {actual_ratio:.3f} (expected: {expected_ratio:.3f})")
    
    # Update test
    print("\nUpdate test:")
    print(f"Before update - Total: {tree.total()}")
    tree.update(tree.capacity - 1 + 3, 10.0)  # Update data index 3 to priority 10.0
    print(f"After update - Total: {tree.total()}")
    print(f"Expected: {sum(priorities) - priorities[3] + 10.0}")


if __name__ == "__main__":
    test_sum_tree()
