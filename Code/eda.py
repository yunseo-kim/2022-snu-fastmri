import h5py
import matplotlib.pyplot as plt

f = h5py.File('../result/test_Unet/reconstructions_val/brain95.h5', 'r')
H5_LIST, f