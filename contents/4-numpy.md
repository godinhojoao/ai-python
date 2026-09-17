# Numpy

## What is Numpy?

- Numpy is a package for scientific computing in Python.
- It provides a multidimensional array object, and various derived objects, also some fast operations on arrays, including mathematical, logical shape manipulation, sorting, selecting, I/O, basic linear algebra, basic statistical operations, and more.

## The `ndarray` object

- Encapsulates **n-dimensional arrays of homogeneous data types**. (items with same size)
  - **Exception**: when array items are python or numpy objects (items with different sizes).
- Implemented in pre-compiled C code with many performance optimizations, such as SIMD vectorization, CPU cache optimization, and optimized memory access using spatial locality and temporal locality, and more.
  - **SIMD - single instruction multiple data** (e.g. AVX operations with vector registers)
  - **Spatial locality** = accessing data close to each other is faster (contiguous memory).
  - **Temporal locality** = accessing data fetched recently is faster.
- Have a **fixed size at creation**. (Changing the size of a ndarray will actually delete the old one and create a new one)
  - NumPy arrays are pretty different than the standard python sequences.
  - Python lists grow dynamically.
- Facilitate advanced mathematical and other types of operations on large numbers of data.

## Basics about `numpy` - theory

- `ndarray` = n-dimensional array.
  - `ndim` = how many axes
  - `shape` = tuple with the size of each axis (e.g. `(2, 3)` = 2 rows, 3 columns).
  - `size` = how many elements in total (product of the shape).
  - `dtype` = data type of the items (all the same, e.g. `int64`, `float64`).
  - `axis = dimension`, 2D array has axis 0: rows and 1: columns.
- `ufunc` = universal function, an operation applied element-wise (np.sort, +, np.maximum).
- `broadcasting` = how numpy treats arrays with different shapes in arithmetic ops: the smaller array is broadcast across the larger so their shapes become compatible (element-wise ops need matching shapes).
  - The stretching is only conceptual, numpy doesn't actually copy the data.

## a

- a

## References

- [numpy](https://numpy.org/doc/stable/)
