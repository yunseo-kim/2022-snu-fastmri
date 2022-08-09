import h5py
import matplotlib.pyplot as plt
import numpy as np

f = h5py.File('../../input/train/image/brain95.h5', 'r')

# h5py.File acts like a Python dictionary, thus we can check the keys
print(list(f.keys()))
# ['image_grappa', 'image_input', 'image_label']

# Let's examine the data set as a Dataset object
image_grappa = f['image_grappa']
image_input = f['image_input']
image_label = f['image_label']

# The object we obtained isn’t an array, but an HDF5 dataset.
# Like NumPy arrays, datasets have both a shape and a data type:
print("image_grappa: ")
print(image_grappa.shape)
print(image_grappa.dtype)
print('\n')
'''
image_grappa: 
(16, 384, 384)
float32
'''

print("image_input: ")
print(image_input.shape)
print(image_input.dtype)
print('\n')
'''
image_input: 
(16, 384, 384)
float32
'''

print("image_label: ")
print(image_label.shape)
print(image_label.dtype)
print('\n')
'''
image_label: 
(16, 384, 384)
float32
'''

# numpy로 평균과 표준편차, 최댓값, 최솟값 확인
print("[image_grappa]")
print("평균: ", np.mean(image_grappa))
print("표준편차: ", np.std(image_grappa))
print("최댓값: ", np.max(image_grappa))
print("최솟값: ", np.min(image_grappa))
print('\n')

print("[image_input]")
print("평균: ", np.mean(image_input))
print("표준편차: ", np.std(image_input))
print("최댓값: ", np.max(image_input))
print("최솟값: ", np.min(image_input))
print('\n')

print("[image_label]")
print("평균: ", np.mean(image_label))
print("표준편차: ", np.std(image_label))
print("최댓값: ", np.max(image_label))
print("최솟값: ", np.min(image_label))
print('\n')

# 픽셀 데이터의 스케일이 작은 편이다.
# 각 데이터셋은 거의 균일한 스케일을 가지는 것으로 확인됐으며, 별도의 스케일링 작업을 하지 않아도 무방할 듯 하다.

f = h5py.File('../../input/train/kspace/brain95.h5', 'r')
print(list(f.keys()))
# ['kspace', 'mask']

kspace = f['kspace']
mask = f['mask']

print("kspace: ")
print(kspace.shape)
print(kspace.dtype)
print('\n')
'''
kspace: 
(16, 20, 768, 396)
complex64
'''

print("mask: ")
print(mask.shape)
print(mask.dtype)
print('\n')
'''
mask: 
(396,)
float32
'''

print(list(mask))