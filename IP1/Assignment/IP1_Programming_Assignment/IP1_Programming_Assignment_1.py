import matplotlib.pyplot as plt
import numpy as np
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
          img = [img_data[i] for i in range(width * height)]

          return img, width, height, max_gray
        return -1, -1, -1, -1
    
def hist_equal(img, width, height, max_gray):

    hist = [0] * (max_gray+1)
    for pixel in img:
        hist[pixel] += 1

    cdf = [0] * (max_gray+1)
    cdf[0] = hist[0]
    for i in range(1, max_gray+1):
        cdf[i] = cdf[i - 1] + hist[i]

    pixels = width * height
    cdf = [cdf[i] / pixels for i in range(max_gray+1)]

    equalized_img = [0] * (width * height)
    for i in range(width * height):
        equalized_img[i] = int(cdf[img[i]] * max_gray)

    return equalized_img

input_filename = sys.argv[1]

input_img, width, height, max_gray = read_pgm(input_filename)

if input_img != -1 and width != -1 and height != -1 and max_gray != -1:

  equalized_img = hist_equal(input_img, width, height, max_gray)

  plt.figure(figsize=(10, 5))

  plt.subplot(1, 2, 1)
  plt.imshow(np.array(input_img).reshape((height, width)), cmap='gray')
  plt.title('Original Image')
  plt.axis('off')

  plt.subplot(1, 2, 2)
  plt.imshow(np.array(equalized_img).reshape((height, width)), cmap='gray')
  plt.title('Histogram Equalized Image')
  plt.axis('off')

  plt.tight_layout()
  plt.show()

else:
  print("Invalid Input")
