import numpy as np
arr = np.arange(10)  # 0 to 9
arr = arr.astype('float32')
arr = arr[:, None, None, None]  # (10, 1, 1, 1)
np.save('dummydata.npy', arr)


arr = np.array([4, 9, 1, 2, 3, 7, 1, 8, 5, 2, 0, 4, 6, 2, 5, 7])
arr = arr.astype('float32')
arr = arr[:, None, None, None]  # (10, 1, 1, 1)
np.save('dummydata16.npy', arr)
