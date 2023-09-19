import os

import numpy as np
import s3fs
import zarr


def path_exists_with_s3(file_path, s3_file_sys=None):
    if s3_file_sys:
        return s3_file_sys.exists(file_path)
    else:
        return os.path.exists(file_path)

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
