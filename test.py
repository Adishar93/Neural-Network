import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12]])

# Selecting specific rows using advanced indexing
selected_rows = arr[[1, 3]]  # Selects rows 2 and 10

print("Selected rows:")
print(selected_rows)