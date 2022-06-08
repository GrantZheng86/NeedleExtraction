import argparse
import os
import shutil

import numpy as np
import cv2
from Needle_with_BG.ColorPicker import WrappedImage
import glob

clickLocation = []


class NeedleMarkers:

    def __init__(self, markers):
        markers = np.array(markers)
        assert len(markers.shape) == 3, "Incorrect dimension of markers"
        self._markers = markers
        self._marker_x = []
        self._marker_y = []
        self._f = None
        self._initialize_markers()
        self._poly_fit()
        assert len(self._marker_x)/2 == len(self._markers), "Marker Init failed"

    def _initialize_markers(self):

        for marker_pairs in self._markers:
            self._marker_x.append(marker_pairs[0, 0])
            self._marker_x.append(marker_pairs[1, 0])
            self._marker_y.append(marker_pairs[0, 1])
            self._marker_y.append(marker_pairs[1, 1])

        self._marker_x = np.array(self._marker_x)
        self._marker_y = np.array(self._marker_y)

    def get_marker(self):
        return self._markers

    def get_marker_x(self):
        return self._marker_x

    def get_marker_y(self):
        return self._marker_y

    def _poly_fit(self, order=4):
        f = np.polyfit(self._marker_x, self._marker_y, order)
        self._f = np.poly1d(f)

    def get_polyfit(self, order=4):
        self._poly_fit(order)
        return self._f

    def draw_polyfit(self, img):
        y_fits = self._f(self._marker_x)
        y_fits = y_fits.astype(np.int32)

        for i in range(len(y_fits) - 1):
            img = cv2.line(img, (self._marker_x[i], y_fits[i]), (self._marker_x[i+1], y_fits[i+1]), (255, 0, 255), 2)

        return img

    def draw_polyfit_entire_frame(self, img):
        self.draw_polyfit(img)
        h, w, _ = img.shape

        left_x = np.arange(start=0, stop=self._marker_x[0])
        left_y = self._f(left_x)
        left_y = left_y.astype(np.int32)
        right_x = np.arange(start=self._marker_x[-1], stop=w)
        right_y = self._f(right_x)
        right_y = right_y.astype(np.int32)

        for i in range(len(left_x) - 1):
            img = cv2.line(img, (left_x[i], left_y[i]), (left_x[i+1], left_y[i+1]), (255, 0, 255), 2)

        for i in range(len(right_x) - 1):
            img = cv2.line(img, (right_x[i], right_y[i]), (right_x[i+1], right_y[i+1]), (255, 0, 255), 2)

        return img





def find_camera_matrix(images):
    # boardH = 8
    # boardW = 4
    boardH = 8
    boardW = 5
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
        if not ret:
            raise Exception("Camera calibration failed, make sure there are {}x{} vertices in the image".format(boardH,
                                                                                                                boardW))
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
    cv2.circle(frameCopy, (center_1[0], center_1[1]), 2, (0, 0, 255), 1)
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
    parser = argparse.ArgumentParser(description="Where are the images located")
    parser.add_argument("--calibration_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    args = parser.parse_args()
    saving_folder = 'processed_images'
    saving_dir = os.path.join(args.image_dir, saving_folder)
    if os.path.exists(saving_dir):
        shutil.rmtree(saving_dir)
    os.makedirs(saving_dir)

    for each_image in glob.glob(os.path.join(args.image_dir, '*.jpg')):
        short_img_name = each_image.split('\\')[-1]

        wrapped_image = WrappedImage(each_image)
        wrapped_image.calibrate_image(args.calibration_dir)
        marker_only_image, marker_ends = wrapped_image.color_filter_HSV(upper_bound=(15, 255, 255), lower_bound=(165, 100, 40))
        wrapped_image.draw_marker_on_image(marker_ends)

        markers = NeedleMarkers(marker_ends)
        # markers.get_polyfit(3)
        polyfit_drawing = markers.draw_polyfit_entire_frame(wrapped_image.get_image())
        cv2.imwrite(os.path.join(saving_dir, short_img_name), polyfit_drawing)
    # cv2.imshow('Polyfit', polyfit_drawing)
    # cv2.imshow('test', marker_only_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()





    # # image_names = glob.glob('./Actual_Photos/*.JPG')
    # dir = "{}\\*.JPG".format(args.calibration_dir)
    # image_names = glob.glob(dir)
    # images = []
    #
    # for each_name in image_names:
    #     image = cv2.imread(each_name)
    #     images.append(image)
    #
    # new_cam_matrix, Orig_mtx, distortion_coeff = find_camera_matrix(images)
    #
    # # to_correct_path = 'To_Unwarp.jpg'
    # # restored_image = correct_image(to_correct_path, Orig_mtx, distortion_coeff, new_cam_matrix, False)
    # # chooseReferencePoints(restored_image)
    # # np.save('new_cam_mtx.npy', new_cam_matrix)
    # # np.save('cam_mtx.npy', Orig_mtx)
    # # np.save('dist_coeff.npy', distortion_coeff)
