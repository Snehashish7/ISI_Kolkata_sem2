import numpy as np
import matplotlib.pyplot as plt
import sys

def read_pgm(filename):
    with open(filename, 'rb') as f:
        if f.readline() == b'P5\n':
          while True:
              l = f.readline()
              if not l.startswith(b'#'):
                  break

          width, height = map(int, l.split())
          max_gray = int(f.readline().strip())
          img_data = bytearray(f.read())
          img = np.array([img_data[i] for i in range(width * height)])

          return img, width, height, max_gray
        return -1, -1, -1, -1
    
def hist_gen(img, max_gray):

    hist = np.zeros(max_gray+1)
    for pixel in img:
        hist[pixel] += 1
    return hist

def otsu_threshold(hist, width, height):
    total_pixels = width * height
    sum_total_pixel_values = np.sum(hist * np.arange(256))
    mu = sum_total_pixel_values / total_pixels
    theta_t = 0
    mu_t = 0
    var_max = 0
    threshold = 0
    
    for t in range(256):
        theta_t += hist[t]
        if theta_t == 0:
            continue
        if total_pixels == theta_t:
            break
        mu_t += t * hist[t]
        var_between = ((mu_t - mu * theta_t) ** 2) / (theta_t * (total_pixels - theta_t))
        if var_between > var_max:
            var_max = var_between
            threshold = t
            
    return threshold

input_filename = sys.argv[1]
img, width, height, max_gray = read_pgm(input_filename)
hist = hist_gen(img, max_gray)

th = otsu_threshold(hist, width, height)

bin_img = []
for i in img:
    if i > th:
        bin_img.append(1)
    else:
        bin_img.append(0)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(np.reshape(img, (width, height)), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.reshape(bin_img, (width, height)), cmap='gray')
plt.title('Otsu Thresholded Image')
plt.axis('off')

plt.show()
