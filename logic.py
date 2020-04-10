import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True,
    help="path to source image")
ap.add_argument("-r", "--reference", required=True,
    help="path to reference image")
args = vars(ap.parse_args())

def image_stats(image):
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def _scale_array(arr, clip=True):
    if clip:
        scaled = np.clip(arr, 0, 255)
    else:
        scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
        scaled = _min_max_scale(arr, new_range=scale_range)
    return scaled

def main():
    SOURCE_PATH = args["source"]
    REFERENCE_PATH = args["reference"]
    
    sourceImage = cv2.imread(SOURCE_PATH)
    referenceImage = cv2.imread(REFERENCE_PATH)
    
    sourceImage = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2LAB).astype("float32")
    referenceImage = cv2.cvtColor(referenceImage, cv2.COLOR_BGR2LAB).astype("float32")
    
    (lMeanRef, lStdRef, aMeanRef, aStdRef, bMeanRef, bStdRef) = image_stats(referenceImage)

    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(sourceImage)
    
    (l, a, b) = cv2.split(sourceImage)
    
    l -= lMeanSrc
    a -= aMeanSrc
    b -= bMeanSrc
    
    l = (lStdSrc / lStdRef) * l
    a = (aStdSrc / aStdRef) * a
    b = (bStdSrc / bStdRef) * b
    
    l += lMeanRef
    a += aMeanRef
    b += bMeanRef
    
    l = _scale_array(l)
    a = _scale_array(a)
    b = _scale_array(b)
    
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    cv2.imwrite('output.jpg', transfer)
    
if __name__ == "__main__":
    main()