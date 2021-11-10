import numpy as np
import cv2
import glob

clickLocation = []

def find_camera_matrix(images):

    boardH = 9
    boardW = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    sample_img = images[0]
    _, _, color_channel = sample_img.shape

    if color_channel > 1:
        cvt_color = True
    else:
        cvt_color = False

    objp = np.zeros((boardH * boardW, 3), np.float32)
    objp[:, :2] = np.mgrid[0:boardH, 0:boardW].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for i, each_image in enumerate(images):
        if cvt_color:
            gray = cv2.cvtColor(each_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = each_image

        ret, corners = cv2.findChessboardCorners(gray, (boardH, boardW), None)
        print(ret)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(each_image, (boardH, boardW), corners2, ret)
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.imshow('img', each_image)
            cv2.waitKey(0)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            h, w = each_image.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            dst = cv2.undistort(each_image, mtx, dist, None, newcameramtx)
            # x, y, w, h = roi
            # dst = dst[y:y + h, x:x + w]
            # distortion_comp = np.hstack((dst, each_image))
            distortion_comp = dst
            cv2.putText(distortion_comp, str(dist), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.namedWindow("Distortion Comparison")
            cv2.imshow("Distortion Comparison", distortion_comp)
            cv2.waitKey(0)

            cv2.imwrite('Distortion_comp_{}.png'.format(i), distortion_comp)

        else:
            raise Exception('Detection Error')

    cv2.destroyAllWindows()
    return newcameramtx, mtx, dist

def correct_image(image_path, original_matrix, distortion_coeff, new_camera_matrix, imshow=False):
    img = cv2.imread(image_path)
    restored = cv2.undistort(img, original_matrix, distortion_coeff, None, new_camera_matrix)
    if imshow:
        comp = np.hstack((img, restored))
        cv2.namedWindow("Distortion Comparison", cv2.WINDOW_NORMAL)
        cv2.imshow('Distortion Comparison', comp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return restored

def chooseReferencePoints(image):
    cv2.imshow('Choose two base points', image)
    cv2.setMouseCallback("Choose two base points", getXY)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    show_clicked_points(image)

def show_clicked_points(image):
    global clickLocation
    frameCopy = image.copy()

    center_1 = clickLocation[0]
    center_2 = clickLocation[1]
    cv2.circle(frameCopy,(center_1[0], center_1[1]), 2, (0, 0, 255), 1)
    cv2.circle(frameCopy, (center_2[0], center_2[1]), 2, (0, 0, 255), 1)
    cv2.imshow('Selected', frameCopy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def getXY(event, x, y, flags, param):
    """
    MouseClickListener
    """
    global clickLocation
    if event == cv2.EVENT_LBUTTONDOWN:
        clickLocation.append([x, y])

if __name__ == '__main__':
    image_names = glob.glob('./Camera Calibration/*.JPG')
    images = []

    for each_name in image_names:
        image = cv2.imread(each_name)
        images.append(image)

    new_cam_matrix, Orig_mtx, distortion_coeff = find_camera_matrix(images)

    to_correct_path = 'To_Unwarp.jpg'
    restored_image = correct_image(to_correct_path, Orig_mtx, distortion_coeff, new_cam_matrix, False)
    chooseReferencePoints(restored_image)

