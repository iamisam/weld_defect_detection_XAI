import cv2
import os
import numpy as np

class SegmentationReporter:
    def __init__(self, output_dir="output_segmentations"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def save_segmented_image(self, candidate_id, roi_gray, contours):
        """Draws the perimeter in bright green and saves it to disk."""
        # Ensure 3-channel BGR for visualization
        if len(roi_gray.shape) == 2:
            color_roi = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        elif len(roi_gray.shape) == 3:
            color_roi = roi_gray.copy()
        else:
            raise ValueError(f"Unexpected ROI shape: {roi_gray.shape}")
        
        # Draw perimeter (contours) in green, thickness 1
        cv2.drawContours(color_roi, contours, -1, (0, 255, 0), 1)
        
        filename = os.path.join(self.output_dir, f"candidate_{candidate_id}_segmented.png")
        cv2.imwrite(filename, color_roi)
        return filename

    def generate_report_table(self, segmentation_records):
        """Formats the data into a clean console table."""
        print("\n" + "="*110)
        print(f"{'ID':<5} | {'Defect Type':<20} | {'Global BBox (x,y,w,h)':<25} | {'ROI Center (x,y)':<20} | {'Saved To':<30}")
        print("-" * 110)
        
        for rec in segmentation_records:
            bbox_str = f"{rec['global_bbox']}"
            roi_center = "No Perimeter"
            
            # Calculate the center of the largest contour within the 224x224 ROI
            if rec['contours']:
                largest_contour = max(rec['contours'], key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    roi_center = f"({cx}, {cy})"
                    
            save_path = rec['filepath'].split(os.sep)[-1] if rec['filepath'] else "N/A"
            
            print(f"{rec['candidate_id']:<5} | {rec['class']:<20} | {bbox_str:<25} | {roi_center:<20} | {save_path:<30}")
        print("="*110 + "\n")