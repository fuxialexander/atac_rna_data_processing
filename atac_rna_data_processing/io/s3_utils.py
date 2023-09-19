import numpy as np
import s3fs
import zarr


def path_exists_with_s3(file_path, s3_uri=None, s3_file_sys=None):
    if s3_uri:
        return s3_file_sys.exists(f"{s3_uri}/{file_path}")
    else:
        return os.path.exists(file_path)

def load_zarr_with_s3(file_path, mode="r", s3_uri=None, s3_file_sys=None):
    if s3_uri:
        return zarr.open(s3fs.S3Map(f"{s3_uri}/{file_path}", s3=s3_file_sys))
    else:
        return zarr.open(file_path, mode=mode)

def load_np_with_s3(file_path, s3_uri=None, s3_file_sys=None):
    if s3_uri:
        return np.load(s3_file_sys.open(f"{s3_uri}/{file_path}"))
    else:
        return np.load(file_path)
