import argparse

import cv2
import numpy as np
from tkinter import *


class WrappedImage:

    def __init__(self, file_name):
        self._image = cv2.imread(file_name)
        self._color_space = "BGR"

    def image_dimension(self):
        return self._image.shape

    def get_image(self):
        return self._image

    def get_colo_space(self):
        return self._color_space

    def change_to_HSV(self):
        assert self._color_space != "HSV", "Color Space is already HSV"

        if self._color_space == "BGR":
            self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2HSV)
        elif self._color_space == "GRAY":
            self._image = cv2.cvtColor(self._image, cv2.COLOR_GRAY2BGR)
            self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2HSV)
        else:
            raise Exception("Invalid Color Space")

        self._color_space = "HSV"

    def change_to_BGR(self):
        assert self._color_space != "BGR", "Color Space is already BGR"

        if self._color_space == "HSV":
            self._image = cv2.cvtColor(self._image, cv2.COLOR_HSV2BGR)
        elif self._color_space == "GRAY":
            self._image = cv2.cvtColor(self._image, cv2.COLOR_GRAY2BGR)
        else:
            raise Exception("Invalid Color Space")
        self._color_space = "BGR"

    def change_to_GRAY(self):
        assert self._color_space != "GRAY", "Color Space is already GRAY"

        if self._color_space == "HSV":
            self._image = cv2.cvtColor(self._image, cv2.COLOR_HSV2BGR)
            self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        elif self._color_space == "BGR":
            self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        else:
            raise Exception("Invalid Color Space")
        self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)

    def color_filter_HSV(self, upper_bound, lower_bound):
        if self._color_space != "HSV":
            self.change_to_HSV()

        if upper_bound[0] < lower_bound[1]:
            upper_bound_floor_hue = 0
            lower_bound_celling_hue = 180

            upper_bound_floor = (upper_bound_floor_hue, lower_bound[1], lower_bound[2])
            lower_bound_celling = (lower_bound_celling_hue, upper_bound[1], upper_bound[2])
            mask_1 = cv2.inRange(self._image, upper_bound_floor, upper_bound)
            mask_2 = cv2.inRange(self._image, lower_bound, lower_bound_celling)
            mask = cv2.bitwise_or(mask_1, mask_2)
        else:
            mask = cv2.inRange(self._image, lower_bound, upper_bound)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = WrappedImage.process_mask(mask, 6)
        bounding_rect_list = WrappedImage.find_bounding_rect(mask)
        bounding_ellipse_list = WrappedImage.find_bounding_ellipse(mask)
        marker_points = WrappedImage.find_end_locations(bounding_rect_list)
        masked_image = cv2.bitwise_and(self._image, self._image, mask=mask)
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_HSV2BGR)
        # masked_image = self.draw_bounding_rect_on_image(bounding_rect_list)
        masked_image = self.draw_bounding_ellipse_on_image(bounding_ellipse_list)

        return masked_image, marker_points

    def draw_bounding_rect_on_image(self, rectangles):

        if self._color_space != "BGR":
            self.change_to_BGR()

        to_show = self.get_image().copy()

        for rectangle in rectangles:
            to_show = cv2.drawContours(to_show, [np.array(rectangle)], 0, (255, 0, 0), 2)

        return to_show

    def draw_bounding_ellipse_on_image(self, ellipses):
        if self._color_space != "BGR":
            self.change_to_BGR()

        to_show = self.get_image().copy()

        for ellipse in ellipses:
            to_show = cv2.ellipse(to_show, ellipse, (255, 0, 0), 2)

        return to_show


    def draw_marker_on_image(self, markers):
        if self._color_space != "BGR":
            self.change_to_BGR()

        to_draw = self.get_image().copy()

        for marker in markers:
            left = marker[0]
            right = marker[1]
            to_draw = cv2.circle(to_draw, center=left, radius=4, color=(0, 255, 0), thickness=2)
            to_draw = cv2.circle(to_draw, center=right, radius=4, color=(0, 255, 0), thickness=2)

        cv2.imshow('Marker Ends', to_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_image_gradient(self):

        laplacian = cv2.Laplacian(self._image, 8)
        cv2.imshow('Laplacian', laplacian)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_hue_only(self):
        if self.get_colo_space() != "HSV":
            self.change_to_HSV()

        hue_layer = self._image[:, :, 0]
        cv2.imshow('Hue Only', hue_layer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def process_mask(mask, marker_count):
        """
        Filters out smaller, non-marker masked area
        :param mask: The original mask
        :param marker_count: How many markers are there on the needle
        :return:
        """
        retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        # Stat is [left, top, width, height, area]
        stats = stats.tolist()
        for i in range(len(stats)):
            stats[i].append(i)

        stats.sort(key=lambda x: x[-2])
        useful_stats = stats[-1 - marker_count: -1]  # The actual red markers on needle
        useful_labels = np.array(useful_stats)[:, -1]

        to_return_mask = np.zeros_like(mask)
        for label in useful_labels:
            curr_mask = labels == label
            curr_mask = curr_mask.astype(np.uint8)
            to_return_mask = cv2.bitwise_or(to_return_mask, curr_mask)

        return to_return_mask

    @staticmethod
    def find_bounding_rect(processed_mask):
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect_list = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = WrappedImage.organize_corner_points(box)
            rect_list.append(box)

        return rect_list

    @staticmethod
    def find_bounding_ellipse(processed_mask):
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ellipse_list = []

        for contour in contours:
            ellipse = cv2.fitEllipse(contour)
            ellipse_list.append(ellipse)

        return ellipse_list

    @staticmethod
    def organize_corner_points(point_list):
        clickLocations = np.array(point_list)
        center_pt = (np.average(clickLocations[:, 0]), np.average(clickLocations[:, 1]))

        toReturn = {}

        for eachPoint in clickLocations:
            if (eachPoint[0] < center_pt[0]):
                if (eachPoint[1] < center_pt[1]):
                    toReturn[0] = eachPoint
                else:
                    toReturn[1] = eachPoint
            else:
                if (eachPoint[1] > center_pt[1]):
                    toReturn[2] = eachPoint
                else:
                    toReturn[3] = eachPoint

        toReturn = [toReturn[0].tolist(), toReturn[1].tolist(), toReturn[2].tolist(), toReturn[3].tolist()]
        return toReturn

    @staticmethod
    def find_end_locations(bound_rect_list):
        location_list = []
        for rect in bound_rect_list:
            left_x = int(np.average([rect[0][0], rect[1][0]]))
            left_y = int(np.average([rect[0][1], rect[1][1]]))
            right_x = int(np.average([rect[2][0], rect[3][0]]))
            right_y = int(np.average([rect[2][1], rect[3][1]]))
            location_list.append([(left_x, left_y), (right_x, right_y)])

        return location_list


class ColorSilderGroup:

    def __init__(self, tk_root, command):
        self.hue_slider = Scale(tk_root, from_=0, to=179, orient=HORIZONTAL, label="Upper Hue")
        self.sat_slider = Scale(tk_root, from_=0, to=255, orient=HORIZONTAL, label="Upper Saturation")
        self.val_slider = Scale(tk_root, from_=0, to=255, orient=HORIZONTAL, label="Upper Value")

        self.hue_lower_slider = Scale(tk_root, from_=0, to=179, orient=HORIZONTAL, label="Lower Hue")
        self.sat_lower_slider = Scale(tk_root, from_=0, to=255, orient=HORIZONTAL, label="Lower Saturation")
        self.val_lower_slider = Scale(tk_root, from_=0, to=255, orient=HORIZONTAL, label="Lower Value")

        self.button = Button(tk_root, text="OK", command=command)

        self.hue_slider.pack()
        self.sat_slider.pack()
        self.val_slider.pack()

        self.hue_lower_slider.pack()
        self.sat_lower_slider.pack()
        self.val_lower_slider.pack()
        self.button.pack()

    def get_slider_values(self):
        hu = self.hue_slider.get()
        hl = self.hue_lower_slider.get()
        su = self.sat_slider.get()
        sl = self.sat_lower_slider.get()
        vu = self.val_slider.get()
        vl = self.val_lower_slider.get()

        return {"Hue Upper": hu, "Hue Lower": hl, "Sat Upper": su, "Sat Lower": sl, "Val Upper": vu, "Val Lower": vl}


class ColorPicker:

    def __init__(self, file_name):
        self.image = WrappedImage(file_name)
        self.tk_root = None
        self.color_slider_group = None
        self.initialize_GUI()
        self.tk_root.mainloop()

    def initialize_GUI(self):
        self.tk_root = Tk()
        self.tk_root.title('HSV value selector')
        self.color_slider_group = ColorSilderGroup(self.tk_root, self.display_image_inrange)
        self.image.change_to_HSV()
        self.tk_root.geometry("300x400")

    def display_image_inrange(self):
        cv2.destroyAllWindows()
        if self.image.get_colo_space() != "HSV":
            self.image.change_to_HSV()
        assert self.image.get_colo_space() == "HSV", "Not in the correct Color space"
        slider_values = self.color_slider_group.get_slider_values()

        if slider_values['Hue Upper'] < slider_values['Hue Lower']:
            upper_value = (slider_values['Hue Upper'], slider_values['Sat Upper'], slider_values['Val Upper'])
            upper_value_floor = (0, slider_values['Sat Lower'], slider_values['Val Lower'])
            lower_value_celling = (180, slider_values['Sat Upper'], slider_values['Val Upper'])
            lower_value = (slider_values['Hue Lower'], slider_values['Sat Lower'], slider_values['Val Lower'])
            mask_1 = cv2.inRange(self.image.get_image(), lower_value, lower_value_celling)
            mask_2 = cv2.inRange(self.image.get_image(), upper_value_floor, upper_value)
            mask = cv2.bitwise_or(mask_1, mask_2)
        else:
            upper_value = (slider_values['Hue Upper'], slider_values['Sat Upper'], slider_values['Val Upper'])
            lower_value = (slider_values['Hue Lower'], slider_values['Sat Lower'], slider_values['Val Lower'])
            mask = cv2.inRange(self.image.get_image(), lower_value, upper_value)

        self.image.change_to_BGR()
        masked_image = cv2.bitwise_and(self.image.get_image(), self.image.get_image(), mask=mask)
        cv2.imshow('Masked Image', np.hstack((cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), masked_image)))
        cv2.waitKey(0)

    @staticmethod
    def image_value_tester(hue, saturation, value):
        blank_template = np.ones((100, 100, 3))
        blank_template[:, :, 0] *= hue
        blank_template[:, :, 1] *= saturation
        blank_template[:, :, 2] *= value
        blank_template = blank_template.astype(np.uint8)
        test_image = cv2.cvtColor(blank_template, cv2.COLOR_HSV2BGR)

        cv2.imshow("HSV tester", test_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # ColorPicker.image_value_tester(10, 255, 255)
    parser = argparse.ArgumentParser(description='Color range adjustment')
    parser.add_argument('--photo_dir', type=str, required=True)

    args = parser.parse_args()
    photo_dir = args.photo_dir
    # ColorPicker(photo_dir)
    wrapped_image = WrappedImage(photo_dir)
    wrapped_image.show_hue_only()