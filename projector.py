import cv2
import imutils
import numpy as np

def resize(img, ratio):

    w = int(img.shape[1] * ratio)
    h = int(img.shape[0] * ratio)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img

def detect_color(img, color_hsv, check_size=False):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array(color_hsv[0], dtype="uint8")
    upper = np.array(color_hsv[1], dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        rect_center = None
        return img, rect_center, None

    else:
        cnt_to_draw = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        if check_size and cv2.contourArea(cnt_to_draw) < 2000:
            rect_center = None
            return img, rect_center, None
        else:
            (x, y, w, h) = cv2.boundingRect(cnt_to_draw)
            cv2.rectangle(img, (x, y), (x + w, y + h), (8, 46, 225), 2)
            rect_center = (int(x+w/2), int(y+h/2))
            return img, rect_center, (x, y, w, h)


def get_pts(img):
    h, w, _ = img.shape
    tl = (int(w / 8), int(h / 8) - 10)
    tr = (int(7 / 8 * w), int(h / 7))
    bl = (int(w / 8), int(7 / 8 * h))
    br = (int(7 / 8 * w),
          int((tl[1] - tr[1]) / (tl[0] - tr[0]) * (int(7 / 8 * w) - bl[0]) + bl[1]))

    return tl, tr, bl, br

def get_ROI(img, tl, tr, bl, br):

    color = (0, 204, 255)
    cv2.line(img, tl, tr, color, 2)
    cv2.line(img, tr, br, color, 2)
    cv2.line(img, bl, br, color, 2)
    cv2.line(img, tl, bl, color, 2)

def perspective_transfrom(image, tl, tr, bl, br):

    corner_pt = np.float32([tl, tr, br, bl])
    w_tr = int(np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2))
    h_tr = int(bl[1] - tl[1])

    img_params = np.float32([[0, 0], [w_tr, 0], [w_tr, h_tr], [0, h_tr]])

    matrix = cv2.getPerspectiveTransform(corner_pt, img_params)
    img_transformed = cv2.warpPerspective(image, matrix, (w_tr, h_tr))

    return matrix, img_transformed

def pt_transform(matrix, points):

    pts = np.float32(points).reshape(-1, 1, 2)
    transformed_pts = cv2.perspectiveTransform(pts, matrix)

    return transformed_pts[0][0][0], transformed_pts[0][0][1]

x1, y1 = 80, 6
x2, y2 = 383, 22

pxl_ref = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
ratio = 44.5 / pxl_ref

def pxl2cm(len_pxl, ratio):
    return round(len_pxl * ratio, 1)

def get_kb(xl, xr, yl, yr):
    return (yl-yr)/(xl-xr), yl - (yl-yr)/(xl-xr) * xl

def get_dist(x, y, k, b):
    x0 = (y + 1/k*x - b) / (k + 1/k)
    y0 = k * ((y + 1/k*x - b) / (k + 1/k)) + b
    return int(np.sqrt((x0-x)**2 + (y0-y)**2))

def pipeline(img):

    blue_hsv = ((110, 0, 0), (120, 255, 255))
    img_detect, center, box = detect_color(img, blue_hsv, True)

    tl, tr, bl, br = get_pts(img)
    get_ROI(img_detect, tl, tr, bl, br)

    matrix, img_transformed = perspective_transfrom(img, tl, tr, bl, br)
    map = np.ones(img.shape, dtype=np.uint8) * 255
    map[:img_transformed.shape[0], :img_transformed.shape[1], :] = 127

    if center != None:

        center_transformed = pt_transform(matrix, center)
        x, y, w, h = box
        h_tr, w_tr, _ = img_transformed.shape

        dist_up = pxl2cm(int(center_transformed[1]), ratio)
        dist_down = pxl2cm(int(h_tr - center_transformed[1]), ratio)
        dist_left = pxl2cm(int(center_transformed[0]), ratio)
        dist_right = pxl2cm(int(w_tr - center_transformed[0]), ratio)

        cv2.circle(map, center_transformed, 5, (8, 46, 225), -1)

        cv2.putText(img_detect, str(dist_up)+'cm', (int(x), int(y-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img_detect, str(dist_down)+'cm', (int(x+w-60), int(y+h+15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img_detect, str(dist_left)+'cm', (int(x-60), int(y+10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img_detect, str(dist_right)+'cm', (int(x+w+3), int(y+h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return img_detect, map

    else:
        return img_detect, map


def main():
    video = cv2.VideoCapture('obj_detec\input_blue_mug.mov')

    while video.isOpened():

        ret, frame = video.read()

        if not ret:
            break

        frame = resize(frame, 0.5)

        frame, map = pipeline(frame)
        final_frame = np.hstack([frame, map])

        cv2.imshow('ROI + Map', final_frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    main()
