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

    def color_filter_HSV(self, hue, saturation, value):
        if self._color_space != "HSV":
            self.change_to_HSV()



class ColorPicker:

    def __init__(self, file_name):
        self.image = WrappedImage(file_name)
        self.tk_root = None
        self.initialize_GUI()
        self.tk_root.mainloop()

    def initialize_GUI(self):
        self.tk_root = Tk()
        self.tk_root.title('HSV value selector')
        hue_slider = Scale(self.tk_root, from_=0, to=179, orient=HORIZONTAL, label="Upper Hue")
        sat_slider = Scale(self.tk_root, from_=0, to=255, orient=HORIZONTAL, label="Upper Saturation")
        val_slider = Scale(self.tk_root, from_=0, to=255, orient=HORIZONTAL, label="Upper Value")


        hue_slider.pack()
        sat_slider.pack()
        val_slider.pack()
        self.tk_root.geometry("300x200")


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
    ColorPicker("Actual_Photos/12_17/12_17_gs/2021-12-17_13-37-26.jpg")