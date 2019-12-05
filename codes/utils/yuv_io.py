import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_yuv(filename, dims, format, idx_frm, combine=False):

    with open(filename, 'rb') as fp:
        if format == '420':
            blk_size = np.prod(dims) * 3/2
            d00 = dims[0] // 2
            d01 = dims[1] // 2
        elif format == '422':
            blk_size = np.prod(dims) * 2
            d00 = dims[0]
            d01 = dims[1] // 2
        elif format == '444':
            blk_size = np.prod(dims) * 3
            d00 = dims[0]
            d01 = dims[1]
        else:
            raise NotImplementedError()
        fp.seek(int(blk_size * idx_frm), 0)

        Y = np.reshape(np.frombuffer(fp.read(dims[0] * dims[1]), np.uint8), (dims[0], dims[1]))
        U = np.reshape(np.frombuffer(fp.read(d00 * d01), np.uint8), (d00, d01))
        V = np.reshape(np.frombuffer(fp.read(d00 * d01), np.uint8), (d00, d01))

    if combine:
        YUV = np.zeros((dims[0], dims[1], 3), dtype=np.uint8)
        YUV[:, :, 0] = Y
        if format == '420':
            YUV[0::2, 0::2, 1] = U
            YUV[0::2, 1::2, 1] = U
            YUV[1::2, 1::2, 1] = U
            YUV[1::2, 1::2, 1] = U
            YUV[0::2, 0::2, 2] = V
            YUV[0::2, 1::2, 2] = V
            YUV[1::2, 1::2, 2] = V
            YUV[1::2, 1::2, 2] = V
        elif format == '422':
            YUV[:, 0::2, 1] = U
            YUV[:, 1::2, 1] = U
            YUV[:, 0::2, 2] = V
            YUV[:, 1::2, 2] = V
        elif format == '444':
            YUV[:, :, 1] = U
            YUV[:, :, 2] = V
        else:
            raise NotImplementedError()
        return YUV

    return Y, U, V


def save_yuv(data, filename, format, mode='wb'):

    with open(filename, mode) as fp:
        data = data.astype(np.float)
        if format == '420':
            Y = data[:, :, 0]
            U = (data[0::2, 0::2, 1] + data[0::2, 1::2, 1] + data[1::2, 0::2, 1] + data[1::2, 1::2, 1]) / 4
            V = (data[0::2, 0::2, 2] + data[0::2, 1::2, 2] + data[1::2, 0::2, 2] + data[1::2, 1::2, 2]) / 4
        elif format == '422':
            Y = data[:, :, 0]
            U = (data[:, 0::2, 1] + data[:, 1::2, 1]) / 2
            V = (data[:, 0::2, 2] + data[:, 1::2, 2]) / 2
        elif format == '444':
            Y = data[:, :, 0]
            U = data[:, :, 1]
            V = data[:, :, 2]
        else:
            raise NotImplementedError()

        Y, U, V = Y.astype(np.uint8), U.astype(np.uint8), V.astype(np.uint8)
        fp.write(Y.tobytes())
        fp.write(U.tobytes())
        fp.write(V.tobytes())


if __name__ == '__main__':
    filename = '/home/xiyang/Datasets/4KHDR/SDR_4K_YUV/10091373.yuv'
    YUV = load_yuv(filename=filename, dims=(2160, 3840), format='422', idx_frm=0, combine=True)
