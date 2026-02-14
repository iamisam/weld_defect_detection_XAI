import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the Image
# Replace with your actual filename
img_path = "test_image.jpg" 
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found.")
    exit()

# 2. Create the Mask (Thresholding)
# "If a pixel is brighter than 220, it is Text."
# "If a pixel is darker than 220, it is Background/Defect."
# We use 220 as a safe cutoff for bright white text.
_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)

# 3. Clean Up the Mask (Morphology)
# Sometimes the mask has tiny noisy dots. This step removes them.
# It also makes the text slightly thicker to ensure we cover the edges.
kernel = np.ones((3,3), np.uint8)
mask_dilated = cv2.dilate(mask, kernel, iterations=2)

# 4. Inpainting (The Magic Eraser)
# "Look at the mask. Replace those pixels by guessing from neighbors."
# radius=5 looks 5 pixels around to guess the texture.
result = cv2.inpaint(img, mask_dilated, 5, cv2.INPAINT_TELEA)

# --- VISUALIZATION ---
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(img, cmap='gray')
ax[0].set_title("Original (With Text)")
ax[0].axis('off')

ax[1].imshow(mask_dilated, cmap='gray')
ax[1].set_title("The Mask (What we are removing)")
ax[1].axis('off')

ax[2].imshow(result, cmap='gray')
ax[2].set_title("Result (Cleaned)")
ax[2].axis('off')

plt.tight_layout()
plt.show()

# Save the result so you can check it closely
cv2.imwrite("cleaned_xray.png", result)
print("Saved cleaned image to cleaned_xray.png")