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

def findGreenBorder(mask):
    """
    Finds a rectangle that can encircle the boarder of the green background using contour approximation
    :param mask:
    :return:
    """
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=1)
    connectivity = 8
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_mask, connectivity, cv2.CV_32S)
    largest_cc_area = np.max(stats[1:, -1])
    largest_cc_index = np.where(stats[:, -1] == largest_cc_area)[0][0]

    green_bg = labels == largest_cc_index
    green_bg = green_bg.astype(int) * 255
    green_bg = green_bg.astype(np.uint8)
    bg_contour, _ = cv2.findContours(green_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bg_contour = sorted(bg_contour, key=len)[-1]

    # Contour approximation loop, starting from 0.05 and look for the first value that can output a 4-corner shape
    appriximation_value = 0.05
    epsilon = appriximation_value * cv2.arcLength(bg_contour, True)
    bg_contour = cv2.approxPolyDP(bg_contour, epsilon, True)
    l, _, _ = bg_contour.shape
    assert l >= 4, "Contour Approximation is too Large, lower epsilon value"

    while l > 4:
        appriximation_value += 0.005
        epsilon = appriximation_value * cv2.arcLength(bg_contour, True)
        bg_contour = cv2.approxPolyDP(bg_contour, epsilon, True)
        l, _, _ = bg_contour.shape
        assert appriximation_value < 1, "Too many while loop iteration"

    assert l == 4, "While Loop Error"

    # Finds the min area rectangle that encircles this shape
    rect = cv2.minAreaRect(bg_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)



    check = cv2.drawContours(cv2.cvtColor(green_bg, cv2.COLOR_GRAY2BGR), [bg_contour], 0, (0, 0, 255), 2)
    check = cv2.drawContours(check, [box], 0, (0, 255, 0), 2)
    cv2.imshow('Check', check)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print()

def backgroundCrop(img, imshow=False):
    lower_val = (50, 45, 45)
    upper_val = (90, 255, 255)


    mask = cv2.inRange(img, lower_val, upper_val)
    findGreenBorder(mask)

    inverse_mask = cv2.bitwise_not(mask)
    exclude_green = cv2.bitwise_and(img, img, mask=inverse_mask)
    exclude_green = cv2.cvtColor(exclude_green, cv2.COLOR_HSV2BGR)

    if imshow:
        img = np.hstack((img_copy, exclude_green))
        cv2.imshow('a', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return exclude_green

def isolate_markers(img, imshow=False):
    """
    The Hue value for hsv wraps around 0, therefore, two stage masks is needed to isolate the landmarks
    :param img:
    :param imshow:
    :return:
    """
    upper_val = (70, 255, 255)
    middle_val = (0, 50, 50)
    lower_val = (200, 30, 30)

    mask_1 = cv2.inRange(img, middle_val, upper_val)
    mask_2 = cv2.inRange(img, lower_val, middle_val)
    mask = cv2.bitwise_or(mask_1, mask_2)
    landmarks_only = cv2.bitwise_and(img, img, mask=mask)
    landmarks_only = cv2.cvtColor(landmarks_only, cv2.COLOR_HSV2BGR)

    if imshow:
        to_show = np.hstack((img, landmarks_only))
        cv2.imshow('Landmarks', to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return landmarks_only




if __name__ == "__main__":
    # image_name = 'Needle_with_BG/Picture3.jpg'
    image_name = 'Actual_Photos/12_16 cap cal/12_16/2021-12-16_16-22-22.jpg'

    img = cv2.imread(image_name)
    img_copy = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    exclude_green = backgroundCrop(img, False)
    landmarks_only = isolate_markers(exclude_green, True)



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

