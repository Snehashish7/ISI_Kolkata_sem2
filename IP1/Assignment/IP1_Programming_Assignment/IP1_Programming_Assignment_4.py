import numpy as np
import matplotlib.pyplot as plt
import sys

def rgb_to_hsi(rgb_image):
    
    rgb_image_float = rgb_image.astype(float) / 255.0
    r = rgb_image_float[:, :, 0]
    g = rgb_image_float[:, :, 1]
    b = rgb_image_float[:, :, 2]
    
    I = (r + g + b) / 3.0
    
    min_rgb = np.minimum.reduce([r, g, b])
    S = 1 - (3.0 / (r + g + b)) * min_rgb
    
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b) * (g - b))
    hue = np.arccos(num / (den + 1e-5))
    hue[b > g] = 2 * np.pi - hue[b > g]

    H = hue * (180.0 / np.pi)
    
    hsi_image = np.zeros(rgb_image_float.shape)
    hsi_image[:, :, 0] = H
    hsi_image[:, :, 1] = S
    hsi_image[:, :, 2] = I
    
    return hsi_image

def hsi_to_rgb(hsi_image):

    h = hsi_image[:, :, 0] * (np.pi / 180.0)
    s = hsi_image[:, :, 1]
    i = hsi_image[:, :, 2]
    
    r = np.zeros(h.shape)
    g = np.zeros(h.shape)
    b = np.zeros(h.shape)
    
    idx = np.logical_and(0 <= h, h < 120)
    b[idx] = i[idx] * (1 - s[idx])
    r[idx] = i[idx] * (1 + (s[idx] * np.cos(h[idx])) / (np.cos(np.pi / 3 - h[idx])))
    g[idx] = 3 * i[idx] - (r[idx] + b[idx])
    
    idx = np.logical_and(120 <= h, h < 240)
    r[idx] = i[idx] * (1 - s[idx])
    g[idx] = i[idx] * (1 + (s[idx] * np.cos(h[idx] - 120 * np.pi / 180)) / (np.cos(np.pi / 3 - (h[idx] - 120 * np.pi / 180))))
    b[idx] = 3 * i[idx] - (r[idx] + g[idx])
    
    idx = np.logical_and(240 <= h, h < 360)
    g[idx] = i[idx] * (1 - s[idx])
    b[idx] = i[idx] * (1 + (s[idx] * np.cos(h[idx] - 240 * np.pi / 180)) / (np.cos(np.pi / 3 - (h[idx] - 240 * np.pi / 180))))
    r[idx] = 3 * i[idx] - (g[idx] + b[idx])
    
    rgb_image = np.stack([r, g, b], axis=-1)
    rgb_image = np.clip(rgb_image, 0, 1)
    
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    return rgb_image

rgb_image = plt.imread(sys.argv[1])
hsi_image = rgb_to_hsi(rgb_image)
restored_rgb_image = hsi_to_rgb(hsi_image)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.title('Original RGB Image')

plt.subplot(1, 2, 2)
plt.imshow(restored_rgb_image)
plt.title('Restored RGB Image from HSI')

plt.show()
