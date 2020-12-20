import cv2
import numpy as np
import sys


def plot_corners(src_gray):
    dst = cv2.cornerHarris(src_gray, 2, 3, 0.04)
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    w, h = dst_norm.shape
    for j in range(w):
        for i in range(h):
            if int(dst_norm[j][i]) > 60:
                cv2.circle(src_gray, (i, j), 2, 255)


def median_filter(src, distance):
    def _calculate_pixel_color(image, x, y, radius):
        w, h, _ = image.shape
        b_arr = []
        g_arr = []
        r_arr = []
        for j in range(-radius, radius + 1):
            for i in range(-radius, radius + 1):
                index_x = np.clip(x + j, 0, w - 1)
                index_y = np.clip(y + i, 0, h - 1)
                b_arr.append(image[index_x][index_y][0])
                g_arr.append(image[index_x][index_y][1])
                r_arr.append(image[index_x][index_y][2])
        new_color = np.array([np.median(b_arr), np.median(g_arr), np.median(r_arr)], dtype=np.int8)
        return new_color
    dst = src.copy()
    w, h, _ = src.shape
    for j in range(w):
        for i in range(h):
            dst[j][i] = _calculate_pixel_color(src, j, i, int(0.2 * distance[j][i]))
    return dst


def main():
    src = cv2.imread("image.jpg")
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(dst)
    dst = cv2.Canny(dst, 100, 200)
    plot_corners(dst)
    dst = 255 - dst
    dst = cv2.distanceTransform(dst, cv2.DIST_L2, 3)
    dst = median_filter(src, dst)
    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main() or 0)
