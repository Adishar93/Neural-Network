import numpy as np

# Example row of categorical values
row = ['cat', 'dog', 'cat', 'bird']

# Find unique categories
categories = np.unique(row)

# Initialize an array to store the one-hot encoded values
one_hot_encoded = np.zeros((len(row), len(categories)), dtype=int)

# Map each category to an index
category_to_index = {category: i for i, category in enumerate(categories)}

# Encode each value in the row
for i, value in enumerate(row):
    one_hot_encoded[i, category_to_index[value]] = 1

print("One-hot encoded rows:")
print(one_hot_encoded)