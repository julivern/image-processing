import cv2
import numpy as np


def nothing(x):
    pass


class SegmentHSV(object):
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

        self.trackbar = np.zeros((200, 560, 3), np.uint8)
        self.var_name = ['Hue min', 'Sat min', 'Val min', 'Hue max', 'Sat max', 'Val max']
        self.win_name = 'Parameter'
        self.switch = '0 : OFF \n 1 : ON'

    def trackbar_create(self):
        cv2.namedWindow(self.win_name)
        for var in self.var_name:
            cv2.createTrackbar(var, self.win_name, 0, 255, nothing)
        cv2.createTrackbar('Erode kernel size', self.win_name, 0, 20, nothing)
        cv2.createTrackbar('Dilate kernel size', self.win_name, 0, 20, nothing)
        cv2.createTrackbar('Open kernel size', self.win_name, 0, 20, nothing)
        cv2.createTrackbar('Close kernel size', self.win_name, 0, 20, nothing)
        cv2.createTrackbar(self.switch, 'Parameter', 0, 1, nothing)

    def rect_kernel(self, size):
        size = size + 1
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    def ellip_kernel(self, size):
        size = size + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def cross_kernel(self, size):
        size = size + 1
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))

    def start(self):
        self.trackbar_create()
        while True:
            # capture current image
            _, image = self.capture.read()
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # read parameters
            h_min = cv2.getTrackbarPos('Hue min', 'Parameter')
            s_min = cv2.getTrackbarPos('Sat min', 'Parameter')
            v_min = cv2.getTrackbarPos('Val min', 'Parameter')
            h_max = cv2.getTrackbarPos('Hue max', 'Parameter')
            s_max = cv2.getTrackbarPos('Sat max', 'Parameter')
            v_max = cv2.getTrackbarPos('Val max', 'Parameter')
            swi_val = cv2.getTrackbarPos(self.switch, 'Parameter')
            erosion_kern_size = cv2.getTrackbarPos('Erode kernel size', 'Parameter')
            dilate_kern_size = cv2.getTrackbarPos('Dilate kernel size', 'Parameter')
            open_kern_size = cv2.getTrackbarPos('Open kernel size', 'Parameter')
            close_kern_size = cv2.getTrackbarPos('Close kernel size', 'Parameter')
            erode_kernel = np.ones((erosion_kern_size, erosion_kern_size), np.uint8)
            dilate_kernel = np.ones((dilate_kern_size, dilate_kern_size), np.uint8)

            if swi_val == 0:
                self.trackbar[:] = 0
            else:
                self.trackbar[:] = (np.array([h_min, s_min, v_min]) + np.array([h_max, s_max, v_max])) / 2
                self.trackbar = cv2.cvtColor(self.trackbar, cv2.COLOR_HSV2BGR)

            # color filter
            lower_hsv_val = np.array([h_min, s_min, v_min])
            upper_hsv_val = np.array([h_max, s_max, v_max])
            res_col_img = cv2.inRange(hsv, lower_hsv_val, upper_hsv_val)

            # erode image
            eroded_img = cv2.erode(res_col_img, erode_kernel)

            # dilate image
            dilated_image = cv2.dilate(eroded_img, dilate_kernel)

            # opening and closing image transformation
            open_img = cv2.morphologyEx(dilated_image, cv2.MORPH_OPEN, self.rect_kernel(open_kern_size))
            close_img = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, self.rect_kernel(close_kern_size))

            # find contours
            contours, hierarchy = cv2.findContours(close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # bounding box
            try:
                hierarchy = hierarchy[0]
            except TypeError:
                hierarchy = []

            height, width = close_img.shape
            min_x, min_y = width, height
            max_x = max_y = 0

            output_img = image.copy()
            for contour, _ in zip(contours, hierarchy):
                (x, y, w, h) = cv2.boundingRect(contour)
                min_x, max_x = min(x, min_x), max(x + w, max_x)
                min_y, max_y = min(y, min_y), max(y + h, max_y)
                if w > 80 and h > 80:
                    cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if max_x - min_x > 0 and max_y - min_y > 0:
                cv2.rectangle(output_img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

            # displays
            cv2.imshow('Parameter', self.trackbar)
            cv2.imshow('Input', image)
            cv2.imshow('HSV', hsv)
            cv2.imshow('Mask', res_col_img)
            cv2.imshow('Morph', dilated_image)
            cv2.imshow('Open', open_img)
            cv2.imshow('Close', close_img)
            cv2.imshow('Output', output_img)

            # wait until 'q' is touched
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    segment = SegmentHSV()
    segment.start()
