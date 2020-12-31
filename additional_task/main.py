#!/usr/bin/env python
import cv2
import numpy as np
import math

def find_chessboard_points(image, chessboard_shape):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, corners = cv2.findChessboardCorners(gray, chessboard_shape, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

def find_corners(chessboard_points, image_shape):
    h, w = image_shape[:2]
    a, b, c, d = None, None, None, None
    miny, maxy = h, 0
    minx, maxx = w, 0
    for corner in chessboard_points:
        if corner[0][0] < minx:
            minx = corner[0][0]
            a = (corner[0][0], corner[0][1])
        if maxx < corner[0][0]:
            maxx = corner[0][0]
            c = (corner[0][0], corner[0][1])
        if corner[0][1] < miny:
            miny = corner[0][1]
            b = (corner[0][0], corner[0][1])
        if maxy < corner[0][1]:
            maxy = corner[0][1]
            d = (corner[0][0], corner[0][1])
    return np.float32([a, b, c, d])

def get_destination_corners(image_shape):
    h, w = image_shape[:2]
    return np.float32([[w-120, h-1], [w-120, h-90], [w-1, h-90], [w-1, h-1]])

def get_perspective_image(image, source_corners, destination_corners, destination_shape):
    h, w = destination_shape
    matrix = cv2.getPerspectiveTransform(source_corners, destination_corners)
    return cv2.warpPerspective(image, matrix, (w, h))

def get_centers(perspective_image):
    centers = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2GRAY)
    centers = cv2.Canny(centers, 100, 150)
    circles = cv2.HoughCircles(centers, cv2.HOUGH_GRADIENT, 1, 250, param1=35, param2=32)
    circles = np.uint16(np.around(circles))
    points = []
    for i in circles[0,:]:
        points.append((i[0], i[1], i[2]))
        cv2.circle(centers, (i[0], i[1]), 2, 255, 3)
    return centers, points

def calculate_new_position(points):
    A, B = points
    print(A,B)
    l = math.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
    r = A[2] + B[2]
    k = r / l
    new_x = np.uint16(A[0] - np.around((A[0] - B[0]) * k))
    new_y = np.uint16(A[1] - np.around((A[1] - B[1]) * k))
    return (new_x, new_y, B[2])

def main():
    chessboard_shape = (5, 7)
    image = cv2.imread("small_image.jpg")
    points = find_chessboard_points(image, chessboard_shape)
    image = cv2.drawChessboardCorners(image, chessboard_shape, points, True)
    source_corners = find_corners(points, image.shape)
    destination_corners = get_destination_corners(image.shape)
    perspective_image = get_perspective_image(image, source_corners, destination_corners, image.shape[:2])
    image_with_centers, centers = get_centers(perspective_image)
    new_position = calculate_new_position(centers)
    result = perspective_image.copy()
    cv2.circle(result, (new_position[0], new_position[1]), new_position[2], (0, 0, 255), 4)

    cv2.imshow("source image", image)
    cv2.imshow("image with centers", image_with_centers)
    cv2.imshow("result", result)

    cv2.imwrite("results/perspective_image.jpg", perspective_image)
    cv2.imwrite("results/image_with_centers.jpg", image_with_centers)
    cv2.imwrite("results/result.jpg", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    main()
