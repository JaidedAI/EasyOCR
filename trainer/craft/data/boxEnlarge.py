import math
import numpy as np


def pointAngle(Apoint, Bpoint):
    angle = (Bpoint[1] - Apoint[1]) / ((Bpoint[0] - Apoint[0]) + 10e-8)
    return angle

def pointDistance(Apoint, Bpoint):
    return math.sqrt((Bpoint[1] - Apoint[1])**2 + (Bpoint[0] - Apoint[0])**2)

def lineBiasAndK(Apoint, Bpoint):

    K = pointAngle(Apoint, Bpoint)
    B = Apoint[1] - K*Apoint[0]
    return K, B

def getX(K, B, Ypoint):
    return int((Ypoint-B)/K)

def sidePoint(Apoint, Bpoint, h, w, placehold, enlarge_size):

    K, B = lineBiasAndK(Apoint, Bpoint)
    angle = abs(math.atan(pointAngle(Apoint, Bpoint)))
    distance = pointDistance(Apoint, Bpoint)

    x_enlarge_size, y_enlarge_size = enlarge_size

    XaxisIncreaseDistance = abs(math.cos(angle) * x_enlarge_size * distance)
    YaxisIncreaseDistance = abs(math.sin(angle) * y_enlarge_size * distance)

    if placehold == 'leftTop':
        x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
        y1 = max(0, Apoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightTop':
        x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
        y1 = max(0, Bpoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightBottom':
        x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
        y1 = min(h, Bpoint[1] + YaxisIncreaseDistance)
    elif placehold == 'leftBottom':
        x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
        y1 = min(h, Apoint[1] + YaxisIncreaseDistance)
    return int(x1), int(y1)

def enlargebox(box, h, w, enlarge_size, horizontal_text_bool):

    if not horizontal_text_bool:
        enlarge_size = (enlarge_size[1], enlarge_size[0])

    box = np.roll(box, -np.argmin(box.sum(axis=1)), axis=0)

    Apoint, Bpoint, Cpoint, Dpoint = box
    K1, B1 = lineBiasAndK(box[0], box[2])
    K2, B2 = lineBiasAndK(box[3], box[1])
    X = (B2 - B1)/(K1 - K2)
    Y = K1 * X + B1
    center = [X, Y]

    x1, y1 = sidePoint(Apoint, center, h, w, 'leftTop', enlarge_size)
    x2, y2 = sidePoint(center, Bpoint, h, w, 'rightTop', enlarge_size)
    x3, y3 = sidePoint(center, Cpoint, h, w, 'rightBottom', enlarge_size)
    x4, y4 = sidePoint(Dpoint, center, h, w, 'leftBottom', enlarge_size)
    newcharbox = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    return newcharbox