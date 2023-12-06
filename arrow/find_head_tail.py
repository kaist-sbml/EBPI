import cv2
import numpy as np
from PIL import Image
import os

def find_head_tail(src):
    #src= cv2.resize(src,(2*src.shape[1],2*src.shape[0]))
    _, src_bin = cv2.threshold(src,200,255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    #_, src_bin = cv2.threshold(src,200,255,cv2.THRESH_BINARY_INV)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)
    max_len=0
    cnt_number=0
    cnt_coor=(0,0,0,0)
    for i in range(1, cnt):
        (x, y, w, h, area) = stats[i]
        length= (w**2+ h**2)**0.5
        if max_len<length:
            cnt_number= i
            cnt_coor=(x,y,w,h)
            max_len= length

    labels= np.array(np.where(labels ==cnt_number,255,0),dtype=np.uint8)
    skeleton = cv2.ximgproc.thinning(labels, None, 1)

    _, binaryImage = cv2.threshold(skeleton, 128, 10, cv2.THRESH_BINARY)
    
    # Set the end-points kernel:
    h = np.array([[1, 1, 1],
                  [1, 10, 1],
                  [1, 1, 1]])

    # Convolve the image with the kernel:
    imgFiltered = cv2.filter2D(binaryImage, -1, h)

    # Extract only the end-points pixels, those with
    # an intensity value of 110:
    binaryImage = np.where(imgFiltered == 110, 255, 0)
    # The above operation converted the image to 32-bit float,
    # convert back to 8-bit uint
    binaryImage = binaryImage.astype(np.uint8)
    Y, X = binaryImage.nonzero()
    b1_inform='None'
    b2_inform='None'
    if len(X) > 0 or len(Y) > 0:
        Y = Y.reshape(-1,1)
        X = X.reshape(-1,1)
        Z = np.hstack((X, Y))

        # K-means operates on 32-bit float data:
        floatPoints = np.float32(Z)

        # Set the convergence criteria and call K-means:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(floatPoints, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        cluster1Count = np.count_nonzero(label)
        cluster0Count = np.shape(label)[0] - cluster1Count
        # Look for the cluster of max number of points
        # That cluster will be the tip of the arrow:
        maxCluster = 0
        if cluster1Count > cluster0Count:
            maxCluster = 1
        elif cluster1Count == cluster0Count:
            if cluster1Count>=2:
                maxCluster = 2
            else:
                maxCluster = 3 
        grayscaleImageCopy= cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        # Check out the centers of each cluster:
        matRows, matCols = center.shape
        # We need at least 2 points for this operation:
        if matCols >= 2:
            # Store the ordered end-points here:
            orderedPoints = [None] * 2
            # Let's identify and draw the two end-points
            # of the arrow:
            for b in range(matRows):
                # Get cluster center:
                pointX = int(center[b][0])
                pointY = int(center[b][1])
                # Get the "tip"
                if b == maxCluster:
                    inform= [(pointX,pointY),'yes']
                # Get the "tail"
                else:
                    if maxCluster == 3:
                        inform= 'None'
                    else:
                        if maxCluster != 2:
                            inform=[(pointX,pointY),'no']
                        else:
                            inform=[(pointX,pointY),'yes']
                if b==0:
                    b1_inform= inform
                else:
                    b2_inform= inform
    return b1_inform, b2_inform