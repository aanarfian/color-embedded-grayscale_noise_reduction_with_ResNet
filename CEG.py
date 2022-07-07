import cv2 
import sys
import pywt
import numpy as np
import math
from os import walk

# get filename
f = []
for (dirpath, dirnames, filenames) in walk('../Data/DIV2K_train_LR_x8'):
    f.extend(filenames)
    break

def convertUint8(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def ConvertCEG(img):
    #Datatype conversions
    #convert to float
    img =  np.float64(img)   
    img /= 255

    # Wavelet transform of image
    LL, (LH, HL, HH) = pywt.dwt2(img[:,:,0], 'db1')

    # embed color into grayscale using dwt
    cb = cv2.resize(img[:,:,1], (math.ceil(np.shape(img)[1]/2), math.ceil(np.shape(img)[0]/2)))
    cr = cv2.resize(img[:,:,2], (math.ceil(np.shape(img)[1]/2), math.ceil(np.shape(img)[0]/2)))
    coeffs = (LL, (cr, cb, HH))
    colorEmbeddedGrayscaleImage = pywt.idwt2(coeffs, 'db1')
    colorEmbeddedGrayscaleImage = convertUint8(colorEmbeddedGrayscaleImage, 0, 255, np.uint8)
    return colorEmbeddedGrayscaleImage

img = cv2.imread("baboon.png")
if img is None:
    sys.exit("Could not read the image.")
    
YCbCrImage = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
cv2.imshow("Original Image", img)
cv2.imshow("Display window 1", YCbCrImage)
colorEmbeddedGrayscaleImage = ConvertCEG(YCbCrImage)
cv2.imshow("Display window 2", colorEmbeddedGrayscaleImage)
k = cv2.waitKey(0)
cv2.imwrite("../colorEmbeddedGrayscaleImage.png", colorEmbeddedGrayscaleImage)
for i in f:
    img = cv2.imread("../Data/DIV2K_train_LR_x8/"+i)
    if img is None:
        sys.exit("Could not read the image.")
    YCbCrImage = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    colorEmbeddedGrayscaleImage = ConvertCEG(YCbCrImage)
    cv2.imwrite("../Data/C-E-G/"+i, colorEmbeddedGrayscaleImage)