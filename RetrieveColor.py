import cv2 
import sys
import pywt
import numpy as np
from os import walk

def convertUint8(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def retrieveColor(img):
    #Datatype conversions
    #convert to float
    img =  np.float64(img)   
    img /= 255

    # Wavelet transform of image
    LL, (LH, HL, HH) = pywt.dwt2(img[:,:,0], 'db1')

    cb = cv2.resize(LH, (np.shape(LH)[1]*2, np.shape(LH)[0]*2), interpolation=cv2.INTER_CUBIC)
    cr = cv2.resize(HL, (np.shape(LH)[1]*2, np.shape(HL)[0]*2), interpolation=cv2.INTER_CUBIC)
    y = pywt.idwt2((LL, (np.zeros(np.shape(LH)), np.zeros(np.shape(HL)), HH)), "db1")

    temp = []
    for i in range(int(np.shape(y)[0])):
        temp.append(list())
        for j in range(int(np.shape(y)[1])):
            temp[i].append(list())
            temp[i][j].append(y[i][j])
            temp[i][j].append(cr[i][j])
            temp[i][j].append(cb[i][j])
    temp = np.array(temp)
    retrievedColorImage = convertUint8(temp, 0, 255, np.uint8)
    retrievedColorImage = cv2.cvtColor(retrievedColorImage, cv2.COLOR_YCR_CB2RGB)
    return retrievedColorImage

img = cv2.imread("colorEmbeddedGrayscaleImage.png")
if img is None:
    sys.exit("Could not read the image.")
    
retrievedColorImage = retrieveColor(img)
cv2.imshow("Original Image", img)
cv2.imshow("Display window 1", retrievedColorImage)
k = cv2.waitKey(0)

# get filename
f = []
for (dirpath, dirnames, filenames) in walk('../Data/C-E-G'):
    f.extend(filenames)
    break

for i in f:
    img = cv2.imread("../Data/C-E-G/"+i)
    if img is None:
        sys.exit("Could not read the image.")
    retrievedColorImage = retrieveColor(img)
    cv2.imwrite("../Data/RetrievedColorImage/"+i, retrievedColorImage)


