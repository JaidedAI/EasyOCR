import numpy as np
from PIL import Image, ImageDraw, ImageFont


def pil2cv(img_pil):
    img_cv_bgr = np.array(img_pil, dtype=np.uint8)[:, :, ::-1]
    return img_cv_bgr


def cv2pil(img_cv):
    img_cv_rgb = img_cv[:, :, ::-1]
    img_pil = Image.fromarray(img_cv_rgb)
    return img_pil

def cv2_putText(img, text, org, font_face, font_scale, color):
    x, y = org
    b, g, r = color
    color_rgb = (r, g, b)
    img_pil = cv2pil(img)
    draw = ImageDraw.Draw(img_pil)
    font_pil = ImageFont.truetype(font=font_face, size=font_scale)
    #  textsize is deprecated and will be removed in Pillow 10 (2023-07-01). Use textbbox or textlength instead.
    w, h = draw.textsize(text, font=font_pil)
    draw.text(xy=(x, y-h), text=text, fill=color_rgb, font=font_pil)
    img_cv = pil2cv(img_pil)
    return img_cv

