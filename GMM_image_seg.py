import numpy as np
from sklearn.mixture import GaussianMixture
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Image Preparation
image = Image.open('/Users/davidkim/Desktop/Screen Shot 2022-12-27 at 6.14.06 AM.png')

image_array = np.array(image)

pixels = image_array.reshape(-1, 4)  # Reshape to 2D array


# Step 2: Feature Extraction
# In this case, we're using raw pixel values as features

# Step 3-4: Determine number of components and initialize GMM
n_components = 7  # You can adjust this
gmm = GaussianMixture(n_components=n_components, random_state=42)

# Step 5: Fit the GMM (this performs the EM algorithm)
gmm.fit(pixels)

# Steps 6-8: The GMM is now ready for use
# You can access means, covariances, and weights:
means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_

# Example: Use the GMM to assign each pixel to a component
labels = gmm.predict(pixels)


# Create a color map for visualization
unique_labels = np.unique(labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
color_map = dict(zip(unique_labels, colors))

# Create an RGB image based on the labels
rgb_image = np.array([color_map[label] for label in labels])
rgb_image = rgb_image.reshape(image_array.shape)

# Display the original and segmented images
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(rgb_image[:,:,:3])  # Only use RGB channels, not alpha
plt.title(f'GMM Segmentation ({n_components} components)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Optionally, save the segmented image
segmented_pil_image = Image.fromarray((rgb_image[:,:,:3] * 255).astype(np.uint8))
segmented_pil_image.save('segmented_image.png')
