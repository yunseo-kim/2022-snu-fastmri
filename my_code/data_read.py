import os
import PIL as Image
import numpy as np
import matplotlib.pyplot as plt
import h5py

f = h5py.File('/root/input/train/image/brain1.h5', 'r')
print(list(f.keys()))

image_grappa = f['image_grappa']
image_input = f['image_input']
image_label = f['image_label']

print(image_grappa.shape)
#np.transpose로 파일 형태 성형 필요

nframe = image_grappa.shape[0]

nframe_train = 12
nframe_val = 2
nframe_test = 2

dir_save_train = '/root/input/my_train/image'
dir_save_val = '../val'
dir_save_test = '../test'

"""if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)"""

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

offset_nframe = 0

for i in range(nframe_train):
    grappa_ = np.asarray(image_grappa[i,:,:])
    input_ = np.asarray(image_input[i,:,:])
    label_ = np.asarray(image_label[i,:,:])
    #np.save(os.path.join(dir_save_train, 'grappa_%03d.npy' % i), grappa_)
    #np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)
    #np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)

#plt.subplot(131)
"""plt.imshow(grappa_, cmap = 'binary')
plt.title('grappa')
plt.subplot(132)
plt.imshow(input_, cmap = 'binary')
plt.title('input')
plt.subplot(133)"""
plt.imshow(label_, cmap = 'binary')
plt.title('label')
plt.show()