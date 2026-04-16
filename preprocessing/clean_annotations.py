import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = "test_img.jpg" # Replace with your image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found.")
    exit()

# 1. The Core: Strict Global Threshold
# The text is pure white (255). The ribs are grey. 
# 225 ensures we ONLY grab the text and ignore the background ribs.
_, raw_mask = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)

# 2. Safety Net: Contour Area Filtering
# If the bright curved wall on the left gets caught, this deletes it.
contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clean_mask = np.zeros_like(raw_mask)

for cnt in contours:
    area = cv2.contourArea(cnt)
    # Keep small/medium things (Text & Arrows)
    # Ignore tiny noise (< 10) and massive metal walls (> 5000)
    if 10 < area < 5000:
        cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

# 3. The Halo Killer: Massive Dilation
# This is where your first script failed. We must expand the mask heavily
# so it completely swallows the soft white glow around the letters.
kernel = np.ones((11, 11), np.uint8)
final_mask = cv2.dilate(clean_mask, kernel, iterations=2)

# Soften the edges of the mask so the inpainting blends nicely
final_mask = cv2.GaussianBlur(final_mask, (7, 7), 0)

# 4. Inpaint
# Using a larger radius (15) so it pulls clean metal texture from further away
result = cv2.inpaint(img, final_mask, 15, cv2.INPAINT_TELEA)

# --- VISUALIZATION ---
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Plot on the first subplot (index 0)
ax[0].imshow(img, cmap='gray')
ax[0].set_title("Original")
ax[0].axis('off')

# Plot on the second subplot (index 1)
ax[1].imshow(final_mask, cmap='gray')
ax[1].set_title("Final Mask")
ax[1].axis('off')

# Plot on the third subplot (index 2)
ax[2].imshow(result, cmap='gray')
ax[2].set_title("Result (Cleaned)")
ax[2].axis('off')

plt.tight_layout()
plt.show()
cv2.imwrite("debug_mask_v3.png", final_mask)
cv2.imwrite("clean_result_v3.png", result)