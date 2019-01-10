

# Edge detection with High Pass Filters (Convolutional Kenels) 
Using convolutional kernels for image edge dection

### Steps:

 1. Load image
 2. Convert image to gray scale
 3. View Image
 4. Apply blur ( gaussian or mean) 
 5. View image
 6. Apply Sobel filter
 7. View image
 
 ## Get Resources
 
 ```python
 import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

%matplotlib inline
 ```
 
 ## Get Images from my directory
 
 ```python
 # Read in the image
pic = "C:\\Users\\super\\Desktop\\Self Driving Cars\\house1.jpg"
image = mpimg.imread(pic)

plt.imshow(image)
 ```
 
 ![image1](https://github.com/CodeSenpii/edgeDetection/blob/master/ed1.png)
 
 ## Convert image to grayscale
 
 ```python
 # Convert to grayscale for filtering
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#display image
plt.imshow(gray, cmap='gray')
 ```
 ![image1](https://github.com/CodeSenpii/edgeDetection/blob/master/ed1Gray.png)
 
 ## Convolutional Kernels / Filters

** Kernel can be used to transform images **
1. blur images
2. edge detection
3. Sharpen images

etc etc

There are many types of kernels; but the most prominent are listed below


### Sobel x kernel Edge detection : bias towards vertical lines more that horizontal lines

    - [[-1, 0, 1],
       [-2, 0, 2],
       [-1, 0, 1]]
       
### Sobel y kernel edge detection: bias towards horizonatl lines more that vertical lines
    - [[-1,  -2, -1],
       [ 0,   0,  0],
       [ 1,   2,  1]]
       
### Gaussian blur kernel:

    - [[1, 2, 1],
       [2, 4, 2],
       [1, 2, 1]]
       
### Mean blur kernel:

    - [[1,  1, 1],
       [ 1, 1, 1],
       [ 1, 1, 1]]
       
```python
# Create a custom kernel

# gaussian blur filter
gblur = np.array( [ [ 1, 2, 1], 
                   [ 2, 4, 2], 
                   [ 1, 2, 1]]) / 16

#alternative gaussian blur applied directly to image
gblur_alt = cv2.GaussianBlur(gray, (3,3), 0)

#mean blur

mblur = np.array( [ [ 1, 1, 1], 
                   [ 1, 1, 1], 
                   [ 1, 1, 1]]) / 9

#alternative 

mblur_alt = np.ones((3,3), np.float32) / 9


# 3x3 array for edge detection
sobel_y = np.array([[ -1, -2, -1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])



## TODO: Create and apply a Sobel x operator
sobel_x = np.array([[ -1, 0, 1], 
                   [ -2, 0, 2], 
                   [ -1, 2, 1]])

#filtered_image_x = cv2.filter2D(gray, -1, sobel_x)

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
#filtered_image_y = cv2.filter2D(gray, -1, sobel_y)
#filtered_image = cv2.filter2D(filtered_image_x, -1, sobel_y)
filtered_image_blur = cv2.filter2D(gray , -1, mblur)

filtered_image_x = cv2.filter2D(filtered_image_blur , -1, sobel_x)

filtered_image_y = cv2.filter2D(filtered_image_blur, -1, sobel_y)


plt.imshow(filtered_image_y, cmap='gray')
```
## Sobel_y Gray Image
![image1](https://github.com/CodeSenpii/edgeDetection/blob/master/sobel_y.png)
