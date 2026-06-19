import numpy as np

# The central object in NumPy is ndarray (which stands for "n-dimensional array")
# We can create them:
# - from Python lists: `np.array()` create a 1D array/tensor
# - with filling functions: `np.zeros((3, 4))` create a 3x4 matrix filled with zeros
# - with numerical sequences: `np.arange(0, 10, 2)` create an array containing a
#   sequence of uniformely spaced values (e.g.: 0, 2, 4, 6, 8). The np.linspace(0, 10, 5)
#   creates an exact number of values linearly spaced in a given interval.
arr = np.zeros((3, 4))

print(arr)
print(np.arange(0, 10, 2))
print(np.linspace(0, 10, 5))

# Return the array shape in the (lines, columns) format
print(arr.shape)

# Return the number of axis (dimensions)
print(arr.ndim)

# Return the total values quantity
print(arr.size)

# Return the array with it's dimensions reorganized.
# Note that it must be capable of bearing the same number of values of the original array.
# Also, if you know how many lines you need but not how many columns or vice-versa, you can
# pass -1 as an argument to the function.
print(arr.reshape((6, -1)))

# Arithmetic operations
# The basic arithmetical operators +, -, *, / perform element-wise operations automatically
# when dealing with arrays generating new arrays as results.
arrx = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arry = np.array([[121, 232, 381], [403, 543, 681], [702, 884, 934]])

print(arrx + arry)
print(arrx - arry)
print(arrx * arry)
print(arrx / arry)
print(np.dot(arrx, arry))

# Mathematical aggregations
# If the parameter axis=0, the operation is performed on the columns
# If axis=1, the operation is performed on the lines
print(arrx.sum())
print(arrx.min())
print(arrx.max())
print(arrx.mean())

# Indexing and slicing
# When can use colons to separate the standard Python slicing notation into dimensions
# when dealing with NumPy arrays.
print(arrx[1:, 1:])

# Boolean indexing
# We can apply conditional masks.
# Return all the values above 5 extracted from the original array into a plain (1D) one.
print(arrx[arrx > 5])

# Return all the indices that satisfy the condition.
print(np.where(arrx > 5))
# We can also use the function in the form `np.where(arrx > 5, 1, 0)` to substitute the
# array values based on whether they satisfy or not the condition.

# Some quirks of NumPy you should know about:
# - When assigning slices of one array to another variable it does not copy the data, it
# only creates a view on the memory;
# - Always use b = a.copy() when you need independent data;
# - Always validate the shapes of your matrices.

# Exercises:
# 1. Create a unidimensional array (1D) containing the numbers from 1 to 12 using a Numpy function.
# 2. Transform this array in a 2D matrix with 3 lines and 4 columns.
# 3. Print the matrix, its shape and the total qtd of elements (size).
# 4. Create a "boolean mask of the matrix in the first exercise that identifies which numbers are even.
# 5. Use boolean indexing to create a new array containing only the even numbers extracted from the matrix.
# 6. Use the `np.where` function to create a new matrix where all the numbers greater than 6 are substituted by 0.
# 7. Create a 2x2 matrix filled only with ones using `np.ones`.
# 8. Create a second 2x2 matrix only with pseudorandom or manually defined numbers.
# 9. Calculate the element-wise multiplication of these two matrices.
# 10. Calculate the dot product using the appropriate function or operator and compare the results.
# 11. Create a 3x3 tensor with sequential values going from 1 to 9.
# 12. Calculate the sum of all elements in the matrix.
# 13. Find the biggest value of each column.
# 14. Calculate the mean of each line.
