from locale import D_T_FMT

import numpy as np

print("------")
print("1. ndarray basic properties")
a = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

print(a)  # [[1., 2., 3.], [1., 2., 3.]]
print(a * 2)  # [[2. 4. 6.] [2. 4. 6.]] (broadcasting)

print(a.shape)  # (2,3) (2 rows 3 cols - in this 2D)
print(a.ndim)  # 2
print(a.size)  # 2*3 = 6
print(a.dtype)  # float64
print(type(a))  # numpy.ndarray

test = np.array([1, 2, 3])
print(test.shape)  # (3,)
print(test.ndim)  # 1
print(test.size)  # 3

print("------")
print("2. ndarray creation using multiple python sequences, np types and python object type")
fromList = np.array([1, 2, 3])
fromTuple = np.array((1, 2, 3))
fromString = np.array("1,2,3")
twoDimFromTuple = np.array(((1, 2, 3), (4, 5, 6)))
specificDtype = np.array([1, 2, 3], dtype=np.int16)  # seting specific np dtype
threeDim = np.array([ [ [1,2] ],  [ [3,4]] ]) # 3 axes with sizes 2,1,2
fourDim = np.zeros((2,2,2,1), dtype=np.int16) #4 axes with sizes 2,2,2,1 full of zeros of type np.int16

print(fromList)  # [1,2,3]
print(fromTuple)  # [1,2,3]

print(fromString)  # 1,2,3 (this is a single item)
print(fromString.shape)  # ()
print(fromString.ndim)  # 0
print(fromString.size)  # 1

print(twoDimFromTuple)  # [ [1 2 3] [4 5 6] ]
print(twoDimFromTuple.shape)  # (2,3)
print(twoDimFromTuple.dtype)  # int64

print(specificDtype)  # [1 2 3]
print(specificDtype.dtype)  # int64

print(threeDim.shape) # (2,1,2) (3 axes with sizes 2,1,2)
print(threeDim.ndim) # 3
print(threeDim.size) # 4

print("understanding shapes is important fourdim below:")
print(fourDim.shape) # (2,2,2,1)
print(fourDim[0].shape) # (2,2,1)
print(fourDim[0,0].shape) # (2,1)
print(fourDim[0,0,0].shape) # (1)
print(fourDim[0,0,0,0].shape) # ()
print(fourDim[0,0,0,0]) # 0 (this is the value - 0D)

print("------")
print("setting dtype as Python object: dtype=object")
print("We usually avoid this because it loses some performance benefits.")
print("But it can be useful in specific scenarios.")

# Example: we need variable-length arrays (np array using np types doesn't allow it)
studentsGrades = [[10, 7], [8, 8, 8]] # first student missed one exam

# it creates a 1D array since the size of inner lists have different length
usingObjectType = np.array(studentsGrades, dtype=object)
print(usingObjectType.size) # 2 (2 items, and each item is a python list)
print(usingObjectType.shape) # (2,)
print(usingObjectType.ndim) # 1
# usingObjectType[0]  # [10, 7]
# usingObjectType[1]  # [8, 7, 8]

# numpy idea is to avoid looping over elements
# it is possible using numpy types, but not python objects
# per student mean
for studGrades in usingObjectType:
  print(np.mean(studGrades))

# all students mean
print(np.concatenate(usingObjectType)) #it concatenates both [10,7] and [8,7,8] into [10,7,8,7,8]
print(np.mean(np.concatenate(usingObjectType))) # now we can calculate all class mean (8.0)

print("------")
print("3. basic np array operations")

print("3.0 broadcasting")
# 0D is generally used as operation's result or to perform operations with broadcasting
print(np.array([1, 2, 3]) * 2)  # [2,4,6] (2 is 0D) - broadcasting [1 2 3] * [2 2 2]
print(np.array([1, 2, 3]) * np.array(2))  # same result
# broadcasting array is conceptual, not actually created as an array
print(np.array(2).ndim)  # 0

# (2,3) + (3,) -> OK
# (2,3) + (1,3) -> OK
# (2,3) + (2,1) -> OK
# (3,1,2) + (2,) -> OK
# (3,1,2) + (1,1,2) -> OK

# (2,3) + (2,) -> ERROR (3 with 0)
# (2,3) + (4,) -> ERROR (3 with 0)
# (3,1,2) + (0,1,2) -> ERROR (3 with 0)
# (3,2) + (3,4) -> ERROR (2 with 4)

# RULE: compare right to left

# same -> OK
# 1 -> matches everything (broadcast - conceptual stretch)
# missing dimension -> treated as 1

# different, neither -> ERROR
# 0 -> ERROR when compared with a nonzero dimension


print("------")
print("3.1 creating views/copying, indexing and slicing")
a = np.array([ [1,2,3], [4,5,6]], dtype=np.int16)

row = a[0] # it doesn't copy, but creates a view
print(row)

row[0] = 10 #changes a too
print(a)

# Instead of a[0][0], we use a[0, 0] in NumPy for better performance.
# It accesses row 0 and column 0 directly in one operation.
a[0,0] = 11 #changes row too
print(row, "\n")

copiedRow = a[0].copy() #copy
copiedRow[0] = 13 #it doesn't affect a row
print(copiedRow)
print(a[0], "\n")

toslice = np.array([ [1,2,3], [4,5,6], [7,8,9]], dtype=np.int16)
# slicing produce views, not copies
# : means all
# a[0,0] means give me row 0 and col 0
# a[:,:] means give me all rows and cols = entire a
# a[:,0] means give me entire row and only col 0
# a[0,:] means give me row 0 and all cols
print(toslice[:,:]) # [ [1,2,3], [4,5,6],[7,8,9]] (2D)
print(toslice[:,0]) # [1, 4, 7] (1D)
print(toslice[0,:], '\n') # [1,2,3] (1D)
# rows: start 1, stop 3 (exclusive) - all cols
# same as we do with python sequences: pylist[start:end:step]
print(toslice[1:3, :]) # [[4,5,6], [7,8,9]] (2D)

# from row 0 give me items from indexes: 0, 2
print("#fancy indexing: ", toslice[0, [0,2]]) # [1,3]

print("------")
print("3.2 boolean masks to filter (returns copy)")
a = np.array([1,5,2])
print(a>3) #[False True False]
print(a[a>3]) #[5] - select where True (returns copy)
print(a[(a>3) | (a==1)]) #[1 5]

# vectorized if/else, keep where True, else 0
print(np.where(a>3, a, 0)) # [0,5,0]

print("------")
print("3.4 reshaping")
res = np.arange(6) # [0 1 2 3 4 5]
print(res.reshape(2,1,3),"\n") # [ [ [0 1 2] ] [ [3 4 5] ] ]
# -1 means infer this axis
print(res.reshape(2,-1)) # [[0 1 2], [3 4 5]]
print(res.reshape(2,3)) # [ [0 1 2], [3 4 5] ]


twod = np.array([ [1,2,3], [4,5,6]])
print(twod.T) # transpose, 2d: rows<->cols: # [ [1 4], [2 5], [3 6] ]

print("------")
print("3.3 element-wise operations")

a = np.array([1,2,3])
b = (a*10).copy() # [10 20 30]

print(a)
print(b)

print(a*2) # scalar broadcast to every element
print(a+b) # [11 22 23]
print(a*b) # [11 40 90] element wise not matrix multiplication
print(a**2) # [1 4 9]
print(a @ b) # 140 dot product (if 1D) /matrix multiplication (if ndim >=2)
print(np.sqrt(a)) # [1. 1.41421356 1.73205081]
print(np.log(a)) # [0. 0.69314718 1.09861229]
print(np.exp(a)) # [2.71828183  7.3890561  20.08553692]

print("------")
print("3.4 aggregations and axis")
m2d = np.array([ [1,2,3], [4,5,6] ])
m3d = np.array([ [ [1,2,3] ], [ [4,5,6] ] ])

# m.sum(axis=N)
# we will remove this dimension N
# combining all its elements to create an array
# with all the other dimensions

# 1st example:
# m.shape -> (2,3)
# m.sum(axis=0) - remove axis 0 (the one with size 2)
# observe that WE SUM ALONG AXIS 0 (moving through this axis)
# sum items [1+4, 2+5, 3+6] = [5, 7, 9]
# new shape: (3)

# 2nd example:
# m.shape -> (2,3)
# m.sum(axis=1) - remove axis 1 (the one with size 3)
# observe that WE SUM ALONG AXIS 1 (moving through this axis)
# sum items [1+2+3, 4+5+6] = [6, 15]
# new shape: (2)

print(m2d.sum()) # 21 -> scalar: sum whole array and leave no dimension (0d)

## 2D example
print(m2d.shape) # (2, 3)
res = m2d.sum(axis=0)
print(res) # [5 7 9]
print(res.shape) # (3)
res2 = m2d.sum(axis=1)
print(res2) # [6 15]
print(res2.shape) # (2)


## 3D example
# [ // axis 0 = 2
#   [ axis 1 = 1
#     [1,2,3] axis2 = 3
#   ],
#   [ axis 1 = 1
#     [4,5,6] axis2 = 3
#   ]
# ]
# sum(axis=0) = [ [1+4, 2+5, 3+6] ] = [ [5, 7, 9] ]
# sum(axis=1) = [ [1 2 3], [4 5 6] ]
# sum(axis=2) = [ [1+2+3], [4+5+6] ] = [ [6], [15] ]

print(m3d.shape) # (2, 1, 3)
res = m3d.sum(axis=0)
print(res)
print(res.shape) # (1,3)
res2 = m3d.sum(axis=1)
print(res2)
print(res2.shape) # (2,3)
res3 = m3d.sum(axis=2)
print(res3)
print(res3.shape) # (2,1)

print("------")
print("3.5 Joining and splitting")
a = np.array([ [1,2] ])
b = np.array([ [3,4] ])

print(a.shape) # (1,2)
print(b.shape) # (1,2)

print(np.concatenate([a,b], axis=0)) # [[1 2] [3 4]] shape: (2,2)
print(np.concatenate([a,b], axis=1)) # [[1 2 3 4]] shape: (1,4)
print(np.concatenate([a,b])) # [[1 2], [3 4]] shape: (2,2)

print("------")
print("3.6 Comparing arrays")
a = np.array([1,2])
a2 = np.array([1,2])
b = np.array([2,2])

print(a==b) # [False, True] -> element wise
print(np.array_equal(a,b)) # False -> whole array
print(np.array_equal(a,a2)) # True


# Float precision comparisons -> np.allclose()
# rtol = relative tolerance (0.1 = 10%)
# atol = absolute tolerance
print(0.1+0.2 == 0.3) # False
print(np.allclose(0.1+0.2, 0.3)) # True
print(np.allclose(0.1+0.2, 0.3, atol=0.001)) # True "0.3 == 0.3"
print(np.allclose(0.2+0.2, 0.3, atol=0.1)) # True "0.4 == 0.3"
print(np.allclose(0.2+0.3, 0.3, atol=0.1)) # False "0.5 == 0.3"
print(np.allclose(0.2+0.3, 0.3, rtol=0.1)) # False "0.5 == 0.3"

print("------")
print("4 Linear algebra")

print("not yet.")

print("------")