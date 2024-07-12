# CVLAB

Program 6: Display of FTT(1D, 2D) of an image#
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
l=plt.imread('/home/aids5a2-18/Desktop/img.jpg').astype(float)
f1=np.fft.fft2(l)
f2=np.fft.fftshift(f1)
plt.subplot(2,2,1)
plt.imshow(np.abs(f1))
plt.title('Frequency Spectrum')
plt.subplot(2,2,2)
plt.imshow(np.abs(f2))
plt.title('Magnitude Spectrum')
f3=np.log(1+np.abs(f2))
plt.subplot(2,2,3)
plt.imshow(f3)
plt.title('Log(1+np.abs(f2))')
l_fft=fft2(f1)
l1=np.real(l_fft)
plt.subplot(2,2,4)
plt.imshow(l1)
plt.title('2-D FFT')
plt.show()                                           

-----x---------------x-----------------x------------x-----------------x---------------x

Program 7:Computation of mean, Standard Deviation, Correlation coefficient of the given Image

import numpy as np
import matplotlib.pyplot as plt
from skimage import io,color
from scipy.stats import pearsonr
i=io.imread('/home/aids5a2-18/Desktop/img.jpg')
plt.subplot(2,2,1)
plt.imshow(i)
plt.title('Original Image')
g=color.rgb2gray(i)
plt.subplot(2,2,2)
plt.imshow(g,cmap='gray')
plt.title('Grayscale Image')
c=g[100:300,100:300]
plt.subplot(2,2,3)
plt.imshow(c,cmap='gray')
plt.title('Cropped Image')
m=np.mean(c)
s=np.std(c)
print('m',m)
print('s',s)
cb=np.indices((400,400)).sum(axis=0)%2
k=cb>0.8
k1=cb>0.5
plt.figure()
plt.subplot(2,1,1)
plt.imshow(k,cmap='gray')
plt.title('Image1')
plt.subplot(2,1,2)
plt.imshow(k1,cmap='gray')
plt.title('Image2')
r,_=pearsonr(k.flatten(),k1.flatten())
print('r',r)
plt.show()
  
-----x---------------x-----------------x------------x-----------------x---------------x

Program 8: Implementation of Image Smoothening Filters(Mean and Median filtering of an Image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.ndimage import median_filter
I = cv2.imread('/home/aids5a2-18/Desktop/img.jpg')
K = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
J = cv2.randu(K.copy(), 0, 255)
noise = np.random.choice([0, 255], K.shape, p=[0.95, 0.05])
J[noise == 255] = 255
J[noise == 0] = 0
f = median_filter(J, size=(3, 3))
f1 = median_filter(J, size=(10, 10))

plt.figure(figsize=(12, 8))
plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(K, cmap='gray')
plt.title('Gray Image')
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(J, cmap='gray')
plt.title('Noise added Image')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(f, cmap='gray')
plt.title('3x3 Median Filter')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(f1, cmap='gray')
plt.title('10x10 Median Filter')
plt.axis('off')

plt.figure(figsize=(10, 8))

i = cv2.imread('/home/aids5a2-18/Desktop/img.jpg')
g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

g1 = np.ones((3, 3)) / 9.0
b1 = convolve(g, g1)

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(g, cmap='gray')
plt.title('Gray Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(b1, cmap='gray')
plt.title('3x3 Average Filter')
plt.axis('off')

g2 = np.ones((10, 10)) / 100.0
b2 = convolve(g, g2)

plt.subplot(2, 2, 4)
plt.imshow(b2, cmap='gray')
plt.title('10x10 Average Filter')
plt.axis('off')

plt.figure(figsize=(10, 8))

I = cv2.imread('/home/aids5a2-18/Desktop/img1.webp', cv2.IMREAD_GRAYSCALE)
plt.subplot(2, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
plt.axis('off')

a = np.array([[0.001, 0.001, 0.001], [0.001, 0.001, 0.001], [0.001, 0.001, 0.001]])
R = convolve(I, a)

plt.subplot(2, 2, 2)
plt.imshow(R, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

b = np.array([[0.005, 0.005, 0.005], [0.005, 0.005, 0.005], [0.005, 0.005, 0.005]])
R1 = convolve(I, b)

plt.subplot(2, 2, 3)
plt.imshow(R1, cmap='gray')
plt.title('Filtered Image 2')
plt.axis('off')

plt.tight_layout()
plt.show(): 
 
-----x---------------x-----------------x------------x-----------------x---------------x

Program 9: Implementation of image sharpening filters and Edge Detection using Gradient Filters

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy import ndimage
import os

def safe_imread(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")
    return cv2.imread(filename)

def laplacian_filter(img, alpha=0.05):
    kernel = np.array([[0, 1, 0], [1, -4 + alpha, 1], [0, 1, 0]])
    return convolve(img, kernel)

try:
    i = safe_imread('/home/aids5a2-18/Desktop/img.jpg')
    
    plt.subplot(4, 2, 1)
    plt.imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    
    plt.subplot(4, 2, 2)
    plt.imshow(g, cmap='gray')
    plt.title('Gray Image')
    plt.axis('off')

    f = laplacian_filter(g, alpha=0.05)
    
    plt.subplot(4, 2, 3)
    plt.imshow(f, cmap='gray')
    plt.title('Laplacian')
    plt.axis('off')

    s = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    
    plt.subplot(4, 2, 4)
    plt.imshow(s, cmap='gray')
    plt.title('Sobel')
    plt.axis('off')

    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    px = convolve(g, kernelx)
    py = convolve(g, kernely)
    p = np.sqrt(px*2 + py*2)
    
    plt.subplot(4, 2, 5)
    plt.imshow(p, cmap='gray')
    plt.title('Prewitt')
    plt.axis('off')

    kernelx = np.array([[1, 0], [0, -1]])
    kernely = np.array([[0, 1], [-1, 0]])
    rx = convolve(g, kernelx)
    ry = convolve(g, kernely)
    r = np.sqrt(rx*2 + ry*2)
    
    plt.subplot(4, 2, 6)
    plt.imshow(r, cmap='gray')
    plt.title('Roberts')
    plt.axis('off')

    sobel_horizontal = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    
    plt.subplot(4, 2, 7)
    plt.imshow(sobel_horizontal, cmap='gray')
    plt.title('Sobel Horizontal')
    plt.axis('off')

    sobel_vertical = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    
    plt.subplot(4, 2, 8)
    plt.imshow(sobel_vertical, cmap='gray')
    plt.title('Sobel Vertical')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
except FileNotFoundError as e:
    print(e)

 






