import numpy as np
import matplotlib.pyplot as plt
import sys

def convolution_2d(image, kernel):

    image_height = image.shape[0]
    image_width = image.shape[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    kernel = kernel.T

    for y in range(output_height):
        for x in range(output_width):

            region = image[y:y+kernel_height, x:x+kernel_width]

            output[y, x] = np.sum(region * kernel)

    return output

def normalize_image(image, max_gray):
    image_min = np.min(image)
    image_max = np.max(image)
    normalized_image = (image - image_min) / (image_max - image_min) * max_gray
    return normalized_image.astype(np.uint8)

def max_gray(filename):

    with open(filename, 'rb') as f:

        if f.readline() == b'P5\n':
          while True:
              l = f.readline()
              if not l.startswith(b'#'):
                  break

          width, height = map(int, l.split())
          max_gray = int(f.readline().strip())

          return max_gray
        return -1


input_filename = sys.argv[1]
image = plt.imread(input_filename)
max_g = max_gray(input_filename)

if max_g != -1:

    box_kernel = np.ones((3, 3), np.float32) / 9
    gaussian_kernel = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], np.float32) / 16
    laplacian_kernel = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]])
    prewitt_kernel_horizontal = np.array([[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]])
    prewitt_kernel_vertical = np.array([[-1, -1, -1],
                                    [0, 0, 0],
                                    [1, 1, 1]])
    sobel_kernel_horizontal = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]) 
    sobel_kernel_vertical = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])

    kernels = [box_kernel, gaussian_kernel, laplacian_kernel, prewitt_kernel_horizontal, prewitt_kernel_vertical, sobel_kernel_vertical, sobel_kernel_horizontal]
    kernel_names = ['Box Kernel', 'Gaussian Kernel', 'Laplacian Kernel', 'Prewitt Kernel Horizontal', 'Prewitt Kernel Vertical', 'Sobel Kernel Vertical', 'Sobel Kernel Horizontal']
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    for i in range(len(kernels)):

        filtered_image = convolution_2d(image, kernels[i])
        filtered_image_normalized = normalize_image(filtered_image, max_g)

        plt.subplot(2, 4, i+2)
        plt.imshow(filtered_image_normalized, cmap='gray')
        plt.title(f'{kernel_names[i]}')
        plt.axis('off')
        plt.tight_layout()

    plt.show()
else:
  print("Invalid Input")