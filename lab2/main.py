import cv2
import numpy as np
import sys


def conv3d(src, weight):
    def _apply_3d_filter(src, x, y, weight):
        w, h, c = weight.shape
        x_radius = int(w / 2)
        y_radius = int(h / 2)
        result = 0
        for k in range(c):
            for j in range(-x_radius, x_radius + 1):
                for i in range(-y_radius, y_radius + 1):
                    index_x = np.clip(x + j, 0, w - 1)
                    index_y = np.clip(y + i, 0, h - 1)
                    result += src[index_x][index_y][k] * weight[j + x_radius][i + y_radius][k]
        return result

    w, h = src.shape[:2]
    dest = np.zeros(shape=(w-1, h-1, 5))
    for k in range(5):
        for j in range(w-1):
            for i in range(h-1):
                dest[j][i][k] = _apply_3d_filter(src, j, i, weight)
    return dest


def ReLU(src):
    dest = src.copy()
    w, h, c = dest.shape
    for k in range(c):
        for j in range(w):
            for i in range(h):
                dest[j][i][k] = max(0, dest[j][i][k])
    return dest


def MaxPooling2x2(src):
    def _apply_2d_filter(src, x, y, channel):
        w, h = src.shape[:2]
        result = src[x][y][channel]
        for j in range(0, 2):
            for i in range(0, 2):
                index_x = np.clip(x + j, 0, w - 1)
                index_y = np.clip(y + i, 0, h - 1)
                result = max(result, src[index_x][index_y][channel])
        return result

    w, h, c = src.shape
    dest = np.zeros(shape=(int(w/2), int(h/2), c))
    for k in range(c):
        for j in range(int(w/2)):
            for i in range(int(h/2)):
                dest[j][i][k] = _apply_2d_filter(src, j*2, i*2, k)
    return dest


def main():
    src = cv2.imread("image.jpg")
    src = cv2.resize(src, (227, 227))
    dest = conv3d(src, np.random.rand(3, 3, 3))
    dest = ReLU(dest)
    dest = MaxPooling2x2(dest)
    print(dest.shape)


if __name__ == "__main__":
    sys.exit(main() or 0)
