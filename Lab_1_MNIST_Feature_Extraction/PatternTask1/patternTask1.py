import numpy as np
import ssl
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

ssl._create_default_https_context = ssl._create_unverified_context

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = np.concatenate((x_train, x_test), axis=0)
labels = np.concatenate((y_train, y_test), axis=0)

def binarize_images(imgs, threshold=127):
    return (imgs > threshold).astype(np.uint8)

binary_images = binarize_images(images)

def compute_block_centroids(image, num_blocks_row, num_blocks_col):
    h, w = image.shape
    block_h = h // num_blocks_row
    block_w = w // num_blocks_col

    centroids = []

    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            block = image[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
            coords = np.argwhere(block == 1)

            if len(coords) == 0:
                cy, cx = block_h / 2, block_w / 2
            else:
                cy, cx = coords.mean(axis=0)

            centroids.extend([cx / block_w, cy / block_h])

    return centroids

def extract_features(bin_images, num_blocks_row, num_blocks_col):
    feature_vectors = []
    for img in bin_images:
        fv = compute_block_centroids(img, num_blocks_row, num_blocks_col)
        feature_vectors.append(fv)
    return feature_vectors

num_blocks_row = int(input("Enter number of vertical blocks: "))
num_blocks_col = int(input("Enter number of horizontal blocks: "))

print("Processing images... Please wait.")
features = extract_features(binary_images, num_blocks_row, num_blocks_col)

print("\nNumber of images:", len(features))
print("Feature vector size:", len(features[0]))
print("First feature vector:", features[0])
print("First label:", labels[0])

first_image = binary_images[0]
first_centroids = compute_block_centroids(first_image, num_blocks_row, num_blocks_col)

x_coords_norm = first_centroids[::2]
y_coords_norm = first_centroids[1::2]

h, w = first_image.shape
block_h = h // num_blocks_row
block_w = w // num_blocks_col

x_pixels = []
y_pixels = []

for index in range(len(x_coords_norm)):
    i = index // num_blocks_col
    j = index % num_blocks_col

    global_x = (x_coords_norm[index] * block_w) + (j * block_w)
    global_y = (y_coords_norm[index] * block_h) + (i * block_h)

    x_pixels.append(global_x)
    y_pixels.append(global_y)

plt.figure(figsize=(5, 5))
plt.imshow(first_image, cmap='gray')
plt.scatter(x_pixels, y_pixels, c='red', s=40)
plt.title(f"MNIST Digit: {labels[0]}")
plt.axis('off')
plt.show()