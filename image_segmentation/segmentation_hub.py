from .segmenter import DefectSegmenter
from .reporter import SegmentationReporter
import numpy as np

class SegmentationHub:
    def __init__(self):
        self.segmenter = DefectSegmenter(heatmap_thresh=0.5)
        self.reporter = SegmentationReporter()
        self.records = []

    def process_pipeline_results(self, validated_defects):
        """
        Takes the passed-through defect dictionaries from pipeline.py and processes them.
        """
        print("\n[Layer 5] Executing Pixel-Perfect Segmentation...")
        
        for defect in validated_defects:
            # Only process actual defects with Grad-CAM data
            if defect['class'].lower() == 'no_defect' or defect['gradcam'] is None:
                continue
                
            roi_gray = defect['roi']
            
            # The raw heatmap is needed, not the colored RGB image. 
            # We assume pipeline.py passes the raw [224, 224] heatmap array 
            raw_heatmap = defect['raw_heatmap'] 
            
            # 1. Extract perimeters
            contours, binary_mask = self.segmenter.extract_perimeter(roi_gray, raw_heatmap)
            
            # 2. Save the visual result
            filepath = self.reporter.save_segmented_image(defect['candidate_id'], roi_gray, contours)
            
            # 3. Store the record for the final table
            self.records.append({
                'candidate_id': defect['candidate_id'],
                'class': defect['class'],
                'global_bbox': defect['bbox'],
                'contours': contours,
                'filepath': filepath
            })
            
        # 4. Generate the final output table
        if self.records:
            self.reporter.generate_report_table(self.records)
        else:
            print("No actionable defects passed to the segmentation hub.")
            
        return self.records