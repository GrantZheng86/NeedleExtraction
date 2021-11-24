import cv2
import numpy as np

FILE_NAME = 'Test_image.jpg'
# ORIGINAL_IMG = cv2.imread('Needle_with_BG/Picture3.jpg')
ORIGINAL_IMG = cv2.imread(FILE_NAME)
PHYSICAL_POINTS = [[100, 100, 0],
                   [100, 150, 0],
                   [200, 150, 0],
                   [200, 100, 0]]
UL_X = 800
UL_Y = 200
HEIGHT = 100
WIDTH= 200
DESIRED_POINTS = [[UL_X, UL_Y],
                   [UL_X, UL_Y+HEIGHT],
                   [UL_X + WIDTH, UL_Y+HEIGHT],
                   [UL_X+WIDTH, UL_Y]]
PHYSICAL_POINTS = np.array(PHYSICAL_POINTS, dtype=np.float32)


# Corners for the reference block, following a CCW pattern, from the top left corner


def unwarp(img, image_points, desired_points, imshow=False):
    """
    This method unwarps the perspective transformation. Use this method if only using 2D point regression
    :param img:
    :param image_points:
    :param desired_points:
    :return: Perspectivelly unwarpped image
    """
    h, w = img.shape[:2]
    desired_points = np.float32(desired_points)
    M = cv2.getPerspectiveTransform(image_points, desired_points)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if imshow:
        to_show = np.hstack((img, warped))
        cv2.imshow('Unwarpped perspective transform', to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return warped


def extract_reference_locations(img, imshow=False):
    img_copy = img.copy()
    THRESH_VALUE = (0, 220, 220)  # Corner pixels is marked in red. Subject to change, all values in HSV
    THRESH_VALUE_2 = (10, 255, 255)
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), THRESH_VALUE, THRESH_VALUE_2)

    if imshow:
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    _, _, _, centroids = cv2.connectedComponentsWithStats(mask)
    centroids = centroids.astype(np.int32)
    centroids = centroids[1:, :]

    if imshow:
        for centroid in centroids:
            img_copy = cv2.circle(img_copy, tuple(centroid), 5, (0, 255, 255), 1)

        cv2.imshow('detected', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return organizeCornerPoints(centroids)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def organizeCornerPoints(points):
    """
    Organize the points so that the points can be ordered as
    upper left
    lower left
    lower right
    upper right

    The returned object is a dictionary, and the index is the order
    """
    points = np.array(points)
    center_pt = (np.average(points[:, 0]), np.average(points[:, 1]))

    toReturn = {}

    for eachPoint in points:
        if eachPoint[0] < center_pt[0]:
            if eachPoint[1] < center_pt[1]:
                toReturn[0] = eachPoint
            else:
                toReturn[1] = eachPoint
        else:
            if eachPoint[1] > center_pt[1]:
                toReturn[2] = eachPoint
            else:
                toReturn[3] = eachPoint

    return np.array([list(toReturn.get(0)), list(toReturn.get(1)), list(toReturn.get(2)), list(toReturn.get(3))],
                    dtype=np.float32)


def findNeedleLocations(img, imshow=False):
    """
    Finds all the markers on needle, currently the marker is cyan
    :param img:
    :param imshow:
    :return:
    """
    img_bgr = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    upper_value = (95, 255, 255)
    lower_value = (85, 200, 200)
    mask = cv2.inRange(img, lower_value, upper_value)

    if imshow:
        img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        cv2.imshow('Needle Markers', img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    _, _, _, centroids = cv2.connectedComponentsWithStats(mask)
    centroids = centroids.astype(np.int32)
    centroids = centroids[1:, :]

    return centroids

def findBaseBlockCornerLocations(img, imshow=False):
    """
    Finds all the markers on needle, currently the marker is cyan
    :param img:
    :param imshow:
    :return:
    """
    img_bgr = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    upper_value = (5, 255, 255)
    lower_value = (0, 220, 220)
    mask = cv2.inRange(img, lower_value, upper_value)

    if imshow:
        img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        cv2.imshow('Base Block Markers', img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    _, _, _, centroids = cv2.connectedComponentsWithStats(mask)
    centroids = centroids.astype(np.int32)
    centroids = centroids[1:, :]

    return centroids


def fitNeedleCurvature2D(centroids):
    assert len(centroids) >= 4  # Required for 4th order polynomial fit
    x_loc = centroids[:, 0]
    y_loc = centroids[:, 1]

    func = np.polyfit(x_loc, y_loc, 4)
    return np.poly1d(func)

def fitNeedleCurvature3D(points):

    assert len(points) >= 4
    x_loc = points[:, 0]
    y_loc = points[:, 1]
    z_loc = points[:, 2]

    func_xy = np.polyfit(x_loc, y_loc, 4)
    func_xz = np.polyfit(x_loc, z_loc, 4)

    return np.poly1d(func_xy), np.poly1d(func_xz)

def visualize_needle_function(function, needle_points, image):
    needle_points_x = needle_points[:, 0]
    left_x = np.min(needle_points_x)
    right_x = np.max(needle_points_x)
    x_array = np.arange(left_x, right_x, 1)
    y_val = function(x_array)

    point_list = []
    for i in range(len(x_array)):
        x = np.int32(x_array[i])
        y = np.int32(y_val[i])
        point_list.append((x, y))

    point_list = np.array(point_list, np.int32)

    image_with_contour = cv2.polylines(image, [point_list], isClosed=False, color=(200, 100, 30), thickness=2)
    cv2.imshow('Needle Contour', image_with_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cam_mtx = np.load('cam_mtx.npy')
    dist_coeff = np.load('dist_coeff.npy')
    img = cv2.imread(FILE_NAME)
    image_points = extract_reference_locations(img)
    image_points = np.expand_dims(image_points, axis=1)
    ret, rvecs, tvecs = cv2.solvePnP(PHYSICAL_POINTS, image_points, cam_mtx, dist_coeff)

    # Draws a triad, only for 3D illustration purpose
    axis = np.float32([[2, 0, 0], [0, 2, 0], [0, 0, -2]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cam_mtx, dist_coeff)
    # img = draw(ORIGINAL_IMG, image_points, imgpts)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    base_corners = organizeCornerPoints(findBaseBlockCornerLocations(img, True))
    unwarped_image = unwarp(img, base_corners, DESIRED_POINTS, True)
    needleMarkers = findNeedleLocations(unwarped_image, True)
    fitting_function = fitNeedleCurvature2D(needleMarkers)
    visualize_needle_function(fitting_function, needleMarkers, unwarped_image)
