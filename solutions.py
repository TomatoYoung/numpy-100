# 1
import numpy as np

# 2
print(np.__version__)
# np.show_config()

# 3
z = np.zeros(10)
# print(z)

# 4
z = np.zeros((10,10))
# print(z.size)
# print(z.itemsize)
# print(f'{z.size*z.itemsize} byte')

# 5
# python -c 'import numpy;numpy.info(numpy.add)'

# 6
z = np.zeros(10)
z[4] = 1
# print(z)

# 7
z = np.arange(10,50)
# print(z)

# 8
z = z[::-1] # 从头到尾，步长为-1
# 从20到40逆序，
# z = z[40:20:-1] # 不是20:40,步长的正负会影响切片的方向
# 或者 z = z[20:40][::-1]
# print(z)

# 9
z = np.arange(9).reshape(3,3)
# print(z)

# 10 Find indices of non-zero elements from [1,2,0,0,4,0]
nz = np.nonzero([1,2,0,0,4,0]) # nonzero()返回数组元组，每个维度对应一个数组，其中包含该维度中非零元素的下标。
# print(nz)

# 11
z = np.eye(3)
# print(z)

# 12
z = np.random.random((3,3,3))
# print(z)

# 13
z = np.random.random((10,10))
zmax, zmin = z.max(),z.min()
print(zmax, zmin)

# 14
z = np.random.random(30)
zmean =  z.mean()
# print(zmean)

# 15
z = np.ones((10,10))
z[1:-1,1:-1] = 0
# Z[1:-1, 1:-1] 表示选取第一个维度（行）和第二个维度（列）中，从索引 1 到倒数第二个索引（不包括最后一个）的所有元素。
# print(z)

# 16
z = np.ones((3,3))
z = np.pad(z,pad_width=1,mode='constant',constant_values=0)
# print(z)

# 17
0 * np.nan # 结果仍为nan
np.nan == np.nan # False
np.inf > np.nan # False，nan不能比较
np.nan - np.nan # nan，计算结果还是nan
np.nan in set([np.nan]) # True 
0.3 == 3 * 0.1 # False

# print(0 * np.nan)
# print(np.nan == np.nan)
# print(np.inf > np.nan)
# print(np.nan - np.nan)
# print(np.nan in set([np.nan]))
# print(0.3 == 3 * 0.1)

# 18
z = np.diag(1+np.arange(4),k=-1) # k=-1规定了对角线偏移量，向下偏移一行
# print(z)

# 19
z = np.zeros((8,8),dtype=int)
z[1::2,::2] = 1
z[::2,1::2] = 1
# print(z)

# 20
print(np.unravel_index(99,(6,7,8)))
# 创建一个形状为 (6, 7, 8) 的三维数组
arr1 = np.arange(6*7*8).reshape(6, 7, 8)
# print("Array shape:", arr1)

# 创建一个形状为 (2,10) 的三维数组
arr2 = np.arange(2*10).reshape(2,10)
# print("Array shape:", arr2)
# ()，最后一位是行，倒数第二是列

# 21
z = np.tile(np.array([[0,1],[1,0]]),(4,4))
# print(z)

# 22
# 答案给出的提示公式是标准化(Standardization)的公式
z = np.random.random((5,5))
z = (z-np.mean(z))/np.std(z)
print(z)

# 23
color = np.dtype([("r", np.ubyte),
                  ("g", np.ubyte),
                  ("b", np.ubyte),
                  ("a", np.ubyte)])
colors_array = np.array([(255, 0, 0, 255),  # 红色
                         (0, 255, 0, 255),  # 绿色
                         (0, 0, 255, 255)],  # 蓝色
                        dtype=color)
# print(colors_array)
# 访问数组中的元素
# print("First color:", colors_array[0])
# print("Red channel of the second color:", colors_array[1]["r"])

# 24
z = np.dot(np.ones((5,3)),np.ones((3,2)))
# print(z)

z = np.ones((5,3)) @ np.ones((3,2))
# print(type(z[0][0])),结果会转为小数

# 25
z = np.arange(1,11)
print(z)
z[ (z>3) & (z<8)] *= -1 # 这个括号要带着
print(z)
