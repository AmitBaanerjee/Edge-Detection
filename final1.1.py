import cv2
import numpy as np
import math

x_minimum=0.0
x_maximum=0.0

minimum=0.0
maximum=0.0
mag_maximum=0.0

image = cv2.imread('task1.png', 0)

sobelx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])#, dtype = np.float)
sobely = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])#, dtype = np.float)


N,M=image.shape

sobelxImage = np.asarray([[0.0 for col in range(M)] for row in range(N)])
sobelyImage = np.asarray([[0.0 for col in range(M)] for row in range(N)])
sobelGrad = np.asarray([[0.0 for col in range(M)] for row in range(N)])

## Usage of CV2 Library for sobel
# gx = cv2.filter2D(image, -1, sobelx)
# gy = cv2.filter2D(image, -1, sobely)

#Surrounds array with 0's on the outside perimeter
#image = np.pad(image, (1,1), 'edge')

for i in range(1, N-1):
    for j in range(1, M-1):  
        #Calculate gx and gy using Sobel (horizontal and vertical gradients)
        gx = (sobelx[0][0] * image[i-1][j-1]) + (sobelx[0][1] * image[i-1][j]) + \
             (sobelx[0][2] * image[i-1][j+1]) + (sobelx[1][0] * image[i][j-1]) + \
             (sobelx[1][1] * image[i][j]) + (sobelx[1][2] * image[i][j+1]) + \
             (sobelx[2][0] * image[i+1][j-1]) + (sobelx[2][1] * image[i+1][j]) + \
             (sobelx[2][2] * image[i+1][j+1])    
        if(x_maximum<gx):
            x_maximum=gx
        if(x_minimum>gx):
            x_minimum=gx 
        """gx=(gx - x_minimum) / (x_maximum - x_minimum)"""
        sobelxImage[i-1][j-1] = gx
        #sobelxImage[i-1][j-1] = gx
        gy = (sobely[0][0] * image[i-1][j-1]) + (sobely[0][1] * image[i-1][j]) + \
             (sobely[0][2] * image[i-1][j+1]) + (sobely[1][0] * image[i][j-1]) + \
             (sobely[1][1] * image[i][j]) + (sobely[1][2] * image[i][j+1]) + \
             (sobely[2][0] * image[i+1][j-1]) + (sobely[2][1] * image[i+1][j]) + \
             (sobely[2][2] * image[i+1][j+1])   
        if(maximum<gy):
            maximum=gy
        if(minimum>gy):
            minimum=gy 
       
        sobelyImage[i-1][j-1] = gy
        #sobelyImage[i-1][j-1] = gy
        mag_gradient = math.sqrt((gx * gx) + (gy * gy))
        if(mag_maximum<mag_gradient):
            mag_maximum=mag_gradient
        sobelGrad[i-1][j-1] = mag_gradient    
        #Calculate the gradient magnitude
        

cv2.imwrite('custom_2d_convolution_gx.png',sobelxImage) 
cv2.imwrite('custom_2d_convolution_gy.png',sobelyImage)
cv2.imwrite('custom_2d_convolution_gradient.png',sobelGrad)

pos_edge_x = (sobelxImage - x_minimum) / (x_maximum - x_minimum)
cv2.namedWindow('pos_edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_x_dir', pos_edge_x)
cv2.waitKey(0)
cv2.destroyAllWindows()

pos_edge_y = (sobelyImage - minimum) / (maximum - minimum)
cv2.namedWindow('pos_edge_y_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_y_dir', pos_edge_y)
cv2.waitKey(0)
cv2.destroyAllWindows()


