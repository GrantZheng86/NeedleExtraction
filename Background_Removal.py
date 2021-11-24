import cv2
import numpy as np

clickLocation = []


def getXY(event, x, y, flags, param):
    """
    MouseClickListener
    """
    global clickLocation
    if event == cv2.EVENT_LBUTTONDOWN:
        clickLocation.append((x, y))


def show_clicked_points(image, color=(0, 0, 255)):
    global clickLocation
    frameCopy = image.copy()

    for center in clickLocation:
        frameCopy = cv2.circle(frameCopy, center, 2, color, -1)

    cv2.imshow('Selected', frameCopy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    clickLocation.clear()

    return frameCopy


if __name__ == "__main__":
    image_name = 'Needle_with_BG/Picture3.jpg'

    img = cv2.imread(image_name)
    img_copy = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_val = (50, 45, 45)
    upper_val = (80, 255, 255)

    mask = cv2.inRange(img, lower_val, upper_val)
    inverse_mask = cv2.bitwise_not(mask)
    exclude_green = cv2.bitwise_and(img, img, mask=inverse_mask)
    exclude_green = cv2.cvtColor(exclude_green, cv2.COLOR_HSV2BGR)

    img = np.hstack((img_copy, exclude_green))
    cv2.imshow('a', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # TODO : change the file name for "exclude green" to be the one just processed. The file here is only for
    #  perspective transformation test
    exclude_green = cv2.imread('projection_test.jpg')
    cv2.namedWindow("Pick Reference Corners")
    cv2.imshow('Pick Reference Corners', exclude_green)
    cv2.setMouseCallback("Pick Reference Corners", getXY)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exclude_green_with_reference = show_clicked_points(exclude_green)

    pick_needle_name = "Pick Points On Needle"
    cv2.namedWindow(pick_needle_name)
    cv2.imshow(pick_needle_name, exclude_green_with_reference)
    cv2.setMouseCallback(pick_needle_name, getXY)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exclude_green_with_reference_and_needle_marker = show_clicked_points(exclude_green_with_reference,
                                                                         color=(255, 255, 0))

    cv2.imwrite('Test_image.jpg', exclude_green_with_reference_and_needle_marker)

