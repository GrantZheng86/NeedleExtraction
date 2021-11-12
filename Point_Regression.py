import cv2
import numpy as np

FILE_NAME = 'Test_image.jpg'
# ORIGINAL_IMG = cv2.imread('Needle_with_BG/Picture3.jpg')
ORIGINAL_IMG = cv2.imread(FILE_NAME)
PHYSICAL_POINTS = [[0, 0, 0],
           [0, 2, 0],
           [3.75, 2, 0],
           [3.75, 0, 0]]
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
    M = cv2.getPerspectiveTransform(image_points, desired_points)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if imshow:
        to_show = np.hstack(img, warped)
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
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
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
    img_bgr = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    upper_value = (130, 255, 255)
    lower_value = (120, 200, 200)
    mask = cv2.inRange(img, lower_value, upper_value)

    if imshow:
        img_bgr = cv2.bitwise_and(mask, img_bgr)
        cv2.imshow('Needle Markers', img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    _, _, _, centroids = cv2.connectedComponentsWithStats(mask)
    centroids = centroids.astype(np.int32)
    centroids = centroids[1:, :]

    return centroids




if __name__ == "__main__":
    cam_mtx = np.load('cam_mtx.npy')
    dist_coeff = np.load('dist_coeff.npy')
    img = cv2.imread(FILE_NAME)
    image_points = extract_reference_locations(img)
    image_points = np.expand_dims(image_points, axis=1)
    ret, rvecs, tvecs = cv2.solvePnP(PHYSICAL_POINTS, image_points, cam_mtx, dist_coeff)

    axis = np.float32([[2, 0, 0], [0, 2, 0], [0, 0, -2]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cam_mtx, dist_coeff)
    img = draw(ORIGINAL_IMG, image_points, imgpts)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


