import numpy as np

a = np.array([[1,2,3],[3,4,5],[6,7,8]])
print('value a {}'.format(a))
print('shape a {}'.format(a.shape))
print('sum a {}'.format(a.sum(axis=(1,))))