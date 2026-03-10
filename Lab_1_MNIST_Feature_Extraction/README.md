# Lab 1: Feature Extraction utilizing Block Centroids

This directory contains the Python implementation for extracting structural feature vectors from the MNIST dataset of handwritten digits.

## Experimental Objective
The primary objective of this experiment is to transform raw pixel data into meaningful, low-dimensional feature vectors suitable for downstream Machine Learning classification models. The applied methodology encompasses:

1. **Binarization**: Transforming grayscale images into a binary format applying a predefined threshold.
2. **Grid Partitioning**: Systematically dividing the spatial domain of each image into an $N \times M$ grid of blocks.
3. **Centroid Computation**: Calculating the normalized center of mass (centroid) for the foreground pixels localized within each individual block.
4. **Data Visualization**: Graphically representing the original digit superimposed with the computed block centroids for qualitative verification.

*Note: The implementation incorporates a standardized bypass for macOS SSL certificate verification to ensure uninterrupted dataset acquisition.*
