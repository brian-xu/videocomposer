import cv2 as cv
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image

def bbox(image):
    arr = np.array(image)
    y = h = 0
    while(np.sum(arr[y,:]) == 0):
        y += 1
    try:
        while(np.sum(arr[y+h,:]) != 0):
            h += 1
    except IndexError:
        pass 
    x = w = 0
    while(np.sum(arr[:,x]) == 0):
        x += 1
    try:
        while(np.sum(arr[:,x+w]) != 0):
            w += 1
    except IndexError:
        pass 
    return x+w//2, y+h//2, w, h

def translate_bbox(image_mask, video_mask):
    width, height = image_mask.size
    x1, y1, w1, h1 = bbox(image_mask)
    masks = []
    prev_frame = None
    for frame in video_mask:
        frame = frame.resize((width, height))
        if prev_frame:
            x2, y2, w2, h2 = bbox(frame)
            x3, y3, w3, h3 = bbox(prev_frame)
            image_mask = image_mask.transform(image_mask.size, Image.AFFINE, (1, 0, x3-x2, 0, 1, y3-y2))
            image_mask = image_mask.resize(((width*w2)//w3, (height*h2)//h3))
            masks.append(np.array(image_mask.crop((x1-width//2, y1-height//2, x1+width//2, y1+height//2))))
        prev_frame = frame
    return masks