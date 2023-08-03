import matplotlib.pyplot as plt
import cv2

# Load depth image
depth_image = cv2.imread("rtabmap_pointcloud/imgs/depth/1.png", cv2.IMREAD_ANYDEPTH)

# Display depth image
plt.imshow(depth_image, cmap="gray")
plt.colorbar(label="Depth (mm or m)")
plt.show()
