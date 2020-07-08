# import the required libraries
import cv2
import numpy as np


def nothing():
    pass


class SegmentHSV(object):
    def __init__(self):
        self.image_src = 'image'
        self.trackbar = np.zeros((200, 560, 3), np.uint8)
        self.win_name = ['Input', 'HSV', 'Parameter', 'Mask', 'Transform', 'Open', 'Close', 'Output']
        for win in self.win_name:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 400, 345)
        self.switch = 'Show color'
        self.var_name = ['Hue min', 'Sat min', 'Val min', 'Hue max', 'Sat max', 'Val max']
        self.kernel = ['Erode size', 'Dilate size', 'Open size', 'Close size']
        if self.image_src == 'video_cam':
            self.capture = cv2.VideoCapture(0)
        elif self.image_src == 'image':
            self.capture = cv2.imread('./images-baustelle/test4.jpg', 1)

    def trackbar_create(self):
        """ Create the trackbar and selected color """
        cv2.namedWindow(self.win_name[2])
        for var in self.var_name:
            cv2.createTrackbar(var, self.win_name[2], 0, 255, nothing)
        for kernel in self.kernel:
            cv2.createTrackbar(kernel, self.win_name[2], 0, 20, nothing)
        cv2.createTrackbar(self.switch, 'Parameter', 0, 1, nothing)

    def rect_kernel(self, size):
        """ Rectangular kernel """
        size = size + 1
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    def ellip_kernel(self, size):
        """ Elliptical kernel """
        size = size + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def cross_kernel(self, size):
        """ Cross-shaped kernel """
        size = size + 1
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))

    def start(self):
        self.trackbar_create()
        while True:
            shown_fig = []
            # capture current image
            image = []
            if self.image_src == 'video_cam':
                _, image = self.capture.read()
            elif self.image_src == 'image':
                image = self.capture
            # figure 1 (Input)
            shown_fig.append(image)

            # convert color from RGB to HSV format
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # figure 2 (HSV)
            shown_fig.append(hsv)

            # read params
            hsv_set = []
            for var in self.var_name:
                hsv_set.append(cv2.getTrackbarPos(var, self.win_name[2]))
            switch_val = cv2.getTrackbarPos(self.switch, 'Parameter')
            kernel_size = []
            for kernel in self.kernel:
                kernel_size.append(cv2.getTrackbarPos(kernel, 'Parameter'))
            erode_kernel = np.ones((kernel_size[0], kernel_size[0]), np.uint8)
            dilate_kernel = np.ones((kernel_size[1], kernel_size[1]), np.uint8)

            # show the selected color if switch is on
            if switch_val == 0:
                self.trackbar[:] = 0
            else:
                self.trackbar[:] = (np.array(hsv_set[0:3]) + np.array(hsv_set[3:])) / 2
                self.trackbar = cv2.cvtColor(self.trackbar, cv2.COLOR_HSV2BGR)
            # figure 3 (Parameter)
            shown_fig.append(self.trackbar)

            # color filter
            lower_hsv_val = np.array(hsv_set[0:3])
            upper_hsv_val = np.array(hsv_set[3:])
            result_hsv = cv2.inRange(hsv, lower_hsv_val, upper_hsv_val)
            # figure 4 (Mask)
            shown_fig.append(result_hsv)

            # image processing
            eroded_img = cv2.erode(result_hsv, erode_kernel)
            dilated_image = cv2.dilate(eroded_img, dilate_kernel)
            # figure 5 (Transform)
            shown_fig.append(dilated_image)
            open_img = cv2.morphologyEx(dilated_image, cv2.MORPH_OPEN, self.rect_kernel(kernel_size[2]))
            # figure 6 (Open)
            shown_fig.append(open_img)
            close_img = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, self.rect_kernel(kernel_size[3]))
            # figure 7 (Close)
            shown_fig.append(close_img)

            # find contours
            contours, hierarchy = cv2.findContours(close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # bounding box
            try:
                hierarchy = hierarchy[0]
            except TypeError:
                hierarchy = []

            # get the dimension of image
            height, width = close_img.shape
            min_x, min_y = width, height
            max_x = max_y = 0

            # define the box location and dimension
            output_img = image.copy()
            for contour, _ in zip(contours, hierarchy):
                (x, y, w, h) = cv2.boundingRect(contour)
                min_x, max_x = min(x, min_x), max(x + w, max_x)
                min_y, max_y = min(y, min_y), max(y + h, max_y)
                if w > 50 and h > 50:
                    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

            if max_x - min_x > 0 and max_y - min_y > 0:
                cv2.rectangle(output_img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)
            if contours:
                # show the total area of contour
                cv2.putText(output_img, 'Area of interest : {}'.format(cv2.contourArea(contours[0])), (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)
            # figure 8 (Output)
            shown_fig.append(output_img)

            # result/output display
            for i, win in enumerate(self.win_name):
                cv2.imshow(win, shown_fig[i])

            # wait until 'q' or 'Q' is touched
            _key_val = cv2.waitKey(1)
            if _key_val == ord('q') or _key_val == ord('Q'):
                break

        # release capture and close all windows
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    segment = SegmentHSV()
    segment.start()
