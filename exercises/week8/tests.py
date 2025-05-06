import numpy as np

if __name__ == "__main__":

    f_001 = "/work3/s173944/Python/venv_srgan/3D_datasets/datasets/2022_QIM_52_Bone/femur_001/synthetic_lr_4x/volume/f_001.npy"

    data = np.load(f_001)
    print("data shape", data.shape)
    print("data dtype", data.dtype)

    mm = np.memmap(f_001, mode='r')  # memmap cannot infer the shape and dtype, so we get a flat array of uint8
    print("memmap shape", mm.shape)
    print("memmap dtype", mm.dtype)

    mm = np.memmap(f_001, mode='r', shape=(1, 220, 335, 349), dtype=np.float32)
    print("memmap shape w. shape arg", mm.shape)
    print("memmap dtype w. dtype arg", mm.dtype)

