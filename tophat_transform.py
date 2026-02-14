import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Image (Grayscale)
# Replace with your filename
img_path = "test_img.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found.")
    exit()

# 2. Top-Hat Transform
# We use a kernel roughly the size of a letter (e.g., 25x25)
# This operation effectively subtracts the background.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# 3. Threshold the Top-Hat Result
# Now that the background is gone, the text pops out.
# We can use Otsu's method to automatically find the best cutoff.
_, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. Cleanup (Aggressive Dilation)
# OLD: kernel = np.ones((3,3), np.uint8) -> Too small!
# NEW: Increase size to 7x7 or even 9x9 to cover the "glow"
cleanup_kernel = np.ones((7, 7), np.uint8) 

# We dilate the mask to make the "hole" bigger
mask_clean = cv2.dilate(mask, cleanup_kernel, iterations=2)

# Optional: Add Gaussian Blur to the mask itself to soften the edges
# This helps the inpainting blend more smoothly with the background
mask_clean = cv2.GaussianBlur(mask_clean, (5, 5), 0)

# 5. Inpainting
# Increase the radius slightly to look further for good pixels
result = cv2.inpaint(img, mask_clean, 10, cv2.INPAINT_TELEA)

# --- VISUALIZATION ---
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

ax[0].imshow(img, cmap='gray')
ax[0].set_title("Original")
ax[0].axis('off')

ax[1].imshow(tophat, cmap='gray')
ax[1].set_title("1. Top-Hat (Background Removed)")
ax[1].axis('off')

ax[2].imshow(mask_clean, cmap='gray')
ax[2].set_title("2. Generated Mask")
ax[2].axis('off')

ax[3].imshow(result, cmap='gray')
ax[3].set_title("3. Final Cleaned Image")
ax[3].axis('off')

plt.tight_layout()
plt.show()

# Save for inspection
cv2.imwrite("debug_mask.png", mask_clean)
cv2.imwrite("clean_tophat_result.png", result)
print("Saved debug_mask.png and clean_tophat_result.png")