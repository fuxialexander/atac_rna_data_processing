import os

import s3fs
import zarr
import numpy as np
from scipy.sparse import load_npz
from glob import glob


def open_file_with_s3(file_path, mode="r", s3_file_sys=None):
    if s3_file_sys:
        return s3_file_sys.open(file_path, mode=mode)
    else:
        return open(file_path, mode=mode)

def path_exists_with_s3(file_path, s3_file_sys=None):
    if s3_file_sys:
        return s3_file_sys.exists(file_path)
    else:
        return os.path.exists(file_path)

def glob_with_s3(file_path, s3_file_sys=None):
    if s3_file_sys:
        return s3_file_sys.glob(file_path)
    else:
        return glob(file_path)

def load_zarr_with_s3(file_path, mode="r", s3_file_sys=None):
    if s3_file_sys:
        return zarr.open(s3fs.S3Map(file_path, s3=s3_file_sys), mode=mode)
    else:
        return zarr.open(file_path, mode=mode)

def load_np_with_s3(file_path, s3_file_sys=None):
    if s3_file_sys:
        return np.load(s3_file_sys.open(file_path))
    else:
        return np.load(file_path)

def load_npz_with_s3(file_path, s3_file_sys=None):
    if s3_file_sys:
        return load_npz(s3_file_sys.open(file_path))
    else:
        return load_npz(file_path)
