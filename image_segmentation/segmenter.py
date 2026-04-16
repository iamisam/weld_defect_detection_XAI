import cv2
import numpy as np

class DefectSegmenter:
    def __init__(self, heatmap_thresh=0.4):
        # threshold to consider it as a defect zone
        self.heatmap_thresh = heatmap_thresh

    def extract_perimeter(self, roi_gray, gradcam_heatmap):
        """
        Uses Grad-CAM as a guide to extract pixel-perfect perimeters from the raw ROI.
        """
        # Create binary mask from gradcam zone
        hot_zone_mask = (gradcam_heatmap >= self.heatmap_thresh).astype(np.uint8) * 255
        
        # Find local defects within the ROI
        # Ensure the ROI is actually 1-channel grayscale before thresholding
        if len(roi_gray.shape) == 3:
            roi_gray_1ch = cv2.cvtColor(roi_gray, cv2.COLOR_RGB2GRAY)
        else:
            roi_gray_1ch = roi_gray
            
        # Defects (cracks, porosity) are typically darker than the surrounding metal.
        # We use Adaptive Thresholding (INV) to turn dark spots white.
        adaptive_thresh = cv2.adaptiveThreshold(
            roi_gray_1ch, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            21, 5
        )
        
        # Intersect the models' focus (hot zone) with the physical dark spots
        guided_mask = cv2.bitwise_and(adaptive_thresh, adaptive_thresh, mask=hot_zone_mask)
        
        # Clean up stray pixels
        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(guided_mask, cv2.MORPH_OPEN, kernel)
        
        # 5. Extract the exact perimeters (contours)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours, clean_mask