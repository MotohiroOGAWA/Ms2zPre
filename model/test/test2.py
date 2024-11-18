from itertools import product

def sort_combinations_by_minimums(arrays):
    """
    Generate all combinations from the given arrays and sort them by:
    1. The smallest value.
    2. The second smallest value.
    3. The third smallest value, and so on.
    
    Parameters:
    arrays (list of list): List of arrays to combine.

    Returns:
    list: Sorted combinations.
    """
    # Generate all combinations
    combinations = list(product(*arrays))
    
    # Sort by all elements in sorted order
    sorted_combinations = sorted(combinations, key=lambda x: sorted(x))
    
    return sorted_combinations

# Example: Define arrays with any number of elements
arrays = [
    [2, 6, 9, 13],  # Array 1
    [4, 11],        # Array 2
    [1, 3],         # Array 3
    [5, 8]          # Array 4 (optional additional array)
]

# Get sorted combinations
sorted_combinations = sort_combinations_by_minimums(arrays)

# Display the results
print("Sorted combinations:")
for combination in sorted_combinations:
    print(combination)
