# Image Clustering using K-Means in C++

The Image Clustering project leverages the K-Means clustering algorithm to segment an image into \( K \) distinct clusters. This project is implemented in C++ and focuses on partitioning the pixels of an image into \( K \) clusters based on their color similarity. The primary goal is to group similar colors together, which can be useful for image compression, image segmentation, and feature extraction.

## Project Objectives
- Implement the K-Means clustering algorithm in C++.
- Apply the algorithm to an image to segment it into \( K \) clusters.
- Visualize the clustered image by recoloring each pixel based on its cluster centroid.

## Technologies Used
- **Programming Language:** C++
- **Libraries:** OpenCV for image processing 
- **Algorithm:** K-Means Clustering

## Key Components and Features

### Image Loading and Preprocessing
- Load the image using OpenCV.
- Convert the image from its original format (RGB) to a suitable data structure for clustering (YCbCr).

### K-Means Clustering Implementation
- **Initialization:** Randomly initialize \( K \) centroids from the pixel values.
- **Assignment Step:** Assign each pixel to the nearest centroid based on the Euclidean distance in color space.
- **Update Step:** Recalculate the centroids as the mean of all pixels assigned to each cluster.
- **Iteration:** Repeat the assignment and update steps until convergence.

### Clustered Image Reconstruction
- Replace each pixel's color with the color of its assigned centroid to create the clustered image.
- Save and display the clustered image.

