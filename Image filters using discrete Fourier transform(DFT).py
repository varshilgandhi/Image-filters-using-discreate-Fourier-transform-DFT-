# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:46:42 2021

@author: abc
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np

#read our image
img = cv2.imread("BSE_Image.jpg", 0)

#define discreate fourier transform
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

#shift this dft from corner to the center
dft_shift = np.fft.fftshift(dft)

#define magnitude spretrum
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[: ,:, 1]))

#Define HIGH PASS FILTER (HPF)
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.ones((rows, cols, 2), np.uint8)
r = 80
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 0

#Apply mask
fshift = dft_shift * mask

#define magnitude into fshift image after appying mask
fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]))

#Let's unshift the image ( back from center to corner )
f_ishift = np.fft.ifftshift(fshift)

#Let's inverse discreate fourier transform (idft)
img_back = cv2.idft(f_ishift)

#calculating magnitude of inverse dicreate fourier transform (idft)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])


#Let's visualize above things
fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img, cmap="gray")
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(magnitude_spectrum, cmap="gray")
ax2.title.set_text("FFT of image")
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(fshift_mask_mag, cmap="gray")
ax3.title.set_text('FFT + Mask')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(img_back, cmap="gray")
ax4.title.set_text('After inverse FFT')
plt.show()






























