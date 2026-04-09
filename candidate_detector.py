import cv2
import numpy as np
from typing import List, Tuple, Dict
import os

class CandidateDetector:
    """
    Detects candidate defect regions in radiographic weld images using traditional CV.
    Uses multiple strategies: edge detection, intensity analysis, and morphological operations.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize detector with configurable parameters.
        
        Args:
            config: Dictionary with detection parameters. If None, uses defaults.
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Default configuration optimized for radiographic weld defects."""
        return {
            # Preprocessing
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': (8, 8),
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            
            # Edge detection
            'canny_low': 50,
            'canny_high': 150,
            'canny_aperture': 3,
            
            # Thresholding (for dark defects like cracks, porosity)
            'adaptive_block_size': 21,
            'adaptive_c': 5,
            
            # Morphological operations
            'morph_kernel_size': (3, 3),
            'morph_iterations': 2,
            
            # Contour filtering
            'min_contour_area': 50,        # Ignore tiny noise
            'max_contour_area': 50000,     # Ignore huge artifacts
            'min_aspect_ratio': 0.1,       # Very elongated = possible crack
            'max_aspect_ratio': 10.0,
            'min_solidity': 0.1,           # How "filled" the contour is
            
            # Bounding box expansion (% of box dimensions)
            'bbox_expand_ratio': 0.15,
            
            # NMS (Non-Maximum Suppression)
            'nms_iou_threshold': 0.3,      # Merge overlapping candidates
            
            # Annotation overlap filtering
            'annotation_overlap_threshold': 0.05,  # Reject if >30% overlaps with annotation mask
            
            # Output size for classifier
            'output_size': 224,
            
            # Multi-strategy fusion
            'use_edge_detection': True,
            'use_dark_defects': True,
            'use_bright_defects': True,
        }
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality before detection.
        
        Args:
            image: Grayscale input image
            
        Returns:
            Enhanced image
        """
        # Apply CLAHE to enhance local contrast
        clahe = cv2.createCLAHE(
            clipLimit=self.config['clahe_clip_limit'],
            tileGridSize=self.config['clahe_tile_size']
        )
        enhanced = clahe.apply(image)
        
        # Bilateral filter: reduce noise while preserving edges
        filtered = cv2.bilateralFilter(
            enhanced,
            self.config['bilateral_d'],
            self.config['bilateral_sigma_color'],
            self.config['bilateral_sigma_space']
        )
        
        return filtered
    
    def detect_by_edges(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect candidates using edge detection (good for cracks, sharp boundaries).
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            List of contours
        """
        # Canny edge detection
        edges = cv2.Canny(
            image,
            self.config['canny_low'],
            self.config['canny_high'],
            apertureSize=self.config['canny_aperture']
        )
        
        # Dilate edges to connect nearby fragments
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.config['morph_kernel_size']
        )
        dilated = cv2.dilate(edges, kernel, iterations=self.config['morph_iterations'])
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
    
    def detect_dark_defects(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect dark regions (porosity, cracks - appear darker in X-ray).
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            List of contours
        """
        # Adaptive thresholding to find dark spots
        thresh = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # INV = dark becomes white
            self.config['adaptive_block_size'],
            self.config['adaptive_c']
        )
        
        # Morphological closing to fill small holes
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            self.config['morph_kernel_size']
        )
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
    
    def detect_bright_defects(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect bright regions (some defect types may appear brighter).
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            List of contours
        """
        # Otsu's thresholding for bright spots
        _, thresh = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Morphological opening to remove small noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            self.config['morph_kernel_size']
        )
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            opened,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
    
    def filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filter contours based on geometric properties.
        
        Args:
            contours: List of all detected contours
            
        Returns:
            Filtered list of valid contours
        """
        valid_contours = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Area filter
            if area < self.config['min_contour_area'] or area > self.config['max_contour_area']:
                continue
            
            # Bounding rectangle for aspect ratio
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if aspect_ratio < self.config['min_aspect_ratio'] or aspect_ratio > self.config['max_aspect_ratio']:
                continue
            
            # Solidity: ratio of contour area to convex hull area
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            if solidity < self.config['min_solidity']:
                continue
            
            valid_contours.append(cnt)
        
        return valid_contours
    
    def filter_annotation_overlaps(self, bboxes: List[Tuple[int, int, int, int]], 
                                   annotation_mask: np.ndarray,
                                   overlap_threshold: float = None) -> List[Tuple[int, int, int, int]]:
        """
        Filter out candidate bounding boxes that overlap with annotation regions (smudges).
        
        Args:
            bboxes: List of bounding boxes (x, y, w, h)
            annotation_mask: Binary mask of removed annotations
            overlap_threshold: Fraction of bbox area overlapping with mask to reject (0-1).
                              If None, uses config value.
            
        Returns:
            Filtered list of bounding boxes
        """
        if annotation_mask is None or len(bboxes) == 0:
            return bboxes
        
        if overlap_threshold is None:
            overlap_threshold = self.config.get('annotation_overlap_threshold', 0.3)
        
        filtered_bboxes = []
        
        for (x, y, w, h) in bboxes:
            # Extract the region from annotation mask
            roi_mask = annotation_mask[y:y+h, x:x+w]
            
            # Calculate overlap ratio
            bbox_area = w * h
            overlap_area = np.sum(roi_mask > 0)
            overlap_ratio = overlap_area / bbox_area if bbox_area > 0 else 0
            
            # Keep only if overlap is below threshold
            if overlap_ratio < overlap_threshold:
                filtered_bboxes.append((x, y, w, h))
        
        return filtered_bboxes
    
    def contours_to_bboxes(self, contours: List[np.ndarray], img_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Convert contours to bounding boxes with expansion.
        
        Args:
            contours: List of contours
            img_shape: (height, width) of original image
            
        Returns:
            List of bounding boxes as (x, y, w, h)
        """
        h_img, w_img = img_shape
        bboxes = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Expand bounding box
            expand_w = int(w * self.config['bbox_expand_ratio'])
            expand_h = int(h * self.config['bbox_expand_ratio'])
            
            x = max(0, x - expand_w)
            y = max(0, y - expand_h)
            w = min(w_img - x, w + 2 * expand_w)
            h = min(h_img - y, h + 2 * expand_h)
            
            bboxes.append((x, y, w, h))
        
        return bboxes
    
    def non_max_suppression(self, bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.
        
        Args:
            bboxes: List of bounding boxes (x, y, w, h)
            
        Returns:
            Filtered list of bounding boxes
        """
        if len(bboxes) == 0:
            return []
        
        # Convert to (x1, y1, x2, y2) format
        boxes = []
        for (x, y, w, h) in bboxes:
            boxes.append([x, y, x + w, y + h])
        boxes = np.array(boxes, dtype=np.float32)
        
        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Compute areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by bottom-right y coordinate
        indices = np.argsort(y2)
        
        keep = []
        while len(indices) > 0:
            # Pick the last box
            current = indices[-1]
            keep.append(current)
            
            # Find IoU with remaining boxes
            xx1 = np.maximum(x1[current], x1[indices[:-1]])
            yy1 = np.maximum(y1[current], y1[indices[:-1]])
            xx2 = np.minimum(x2[current], x2[indices[:-1]])
            yy2 = np.minimum(y2[current], y2[indices[:-1]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            overlap = (w * h) / areas[indices[:-1]]
            
            # Keep boxes with IoU less than threshold
            indices = indices[:-1][overlap < self.config['nms_iou_threshold']]
        
        # Convert back to (x, y, w, h)
        filtered_bboxes = []
        for idx in keep:
            x, y, x2_coord, y2_coord = boxes[idx]
            filtered_bboxes.append((int(x), int(y), int(x2_coord - x), int(y2_coord - y)))
        
        return filtered_bboxes
    
    def extract_roi(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract and resize ROI to classifier input size.
        
        Args:
            image: Original image (grayscale or RGB)
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Resized ROI ready for classifier (224x224)
        """
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        
        # Convert to RGB if grayscale
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        
        # Resize to classifier input size
        resized = cv2.resize(roi, (self.config['output_size'], self.config['output_size']))
        
        return resized
    
    def remove_annotations_tophat(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove text annotations using Top-Hat transform (from tophat_transform.py).
        
        Args:
            image: Grayscale input image
            
        Returns:
            Tuple of (cleaned_image, annotation_mask) where annotation_mask marks removed regions
        """
        # Top-Hat Transform to detect bright annotations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Threshold to create mask
        _, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Cleanup and expand mask
        cleanup_kernel = np.ones((7, 7), np.uint8)
        mask_clean = cv2.dilate(mask, cleanup_kernel, iterations=2)
        mask_clean = cv2.GaussianBlur(mask_clean, (5, 5), 0)
        
        # Inpaint using the mask
        cleaned = cv2.inpaint(image, mask_clean, 10, cv2.INPAINT_TELEA)
        
        return cleaned, mask_clean
    
    def detect(self, image: np.ndarray, visualize: bool = False, remove_annotations: bool = True) -> Dict:
        """
        Main detection pipeline: find all candidate defect regions.
        
        Args:
            image: Input image (grayscale or RGB)
            visualize: If True, returns visualization image
            remove_annotations: If True, apply top-hat transform to remove annotations first
            
        Returns:
            Dictionary containing:
                - 'bboxes': List of bounding boxes (x, y, w, h)
                - 'rois': List of extracted ROIs (224x224)
                - 'vis_image': Visualization (if visualize=True)
                - 'cleaned_image': Annotation-removed image (if remove_annotations=True)
                - 'annotation_mask': Mask of removed annotations
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Remove annotations using top-hat transform
        annotation_mask = None
        if remove_annotations:
            gray, annotation_mask = self.remove_annotations_tophat(gray)
        
        # Preprocess
        processed = self.preprocess(gray)
        
        # Multi-strategy detection
        all_contours = []
        
        if self.config['use_edge_detection']:
            edge_contours = self.detect_by_edges(processed)
            all_contours.extend(edge_contours)
        
        if self.config['use_dark_defects']:
            dark_contours = self.detect_dark_defects(processed)
            all_contours.extend(dark_contours)
        
        if self.config['use_bright_defects']:
            bright_contours = self.detect_bright_defects(processed)
            all_contours.extend(bright_contours)
        
        # Filter contours
        valid_contours = self.filter_contours(all_contours)
        
        # Convert to bounding boxes
        bboxes = self.contours_to_bboxes(valid_contours, gray.shape)
        
        # Apply NMS
        bboxes = self.non_max_suppression(bboxes)
        
        # IMPORTANT: Filter out candidates that overlap with annotation smudges
        if annotation_mask is not None:
            bboxes = self.filter_annotation_overlaps(bboxes, annotation_mask)
        
        # Extract ROIs
        rois = [self.extract_roi(image, bbox) for bbox in bboxes]
        
        result = {
            'bboxes': bboxes,
            'rois': rois,
            'num_candidates': len(bboxes),
            'cleaned_image': gray,  # Store the cleaned image (after annotation removal)
            'annotation_mask': annotation_mask  # Store the annotation mask
        }
        
        # Create visualization if requested
        if visualize:
            # Use cleaned image for visualization
            vis_image = gray.copy()
            if len(vis_image.shape) == 2:
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
            else:
                vis_image = image.copy()
            
            for idx, (x, y, w, h) in enumerate(bboxes):
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    vis_image,
                    f"C{idx+1}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
            
            result['vis_image'] = vis_image
        
        return result


def test_detector():
    """Test the detector on a sample image."""
    import matplotlib.pyplot as plt
    
    # Load a test image (replace with your actual image path)
    test_image_path = "test_img3.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        print("Please provide a radiographic weld image for testing.")
        return
    
    image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Initialize detector
    detector = CandidateDetector()
    
    # Run detection with annotation removal
    results = detector.detect(image, visualize=True, remove_annotations=True)
    
    print(f"Found {results['num_candidates']} candidate regions (after filtering annotation smudges)")
    print(f"Bounding boxes: {results['bboxes']}")
    
    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Original Image (with annotations)")
    axes[0, 0].axis('off')
    
    # Annotation mask (debug mask)
    if results['annotation_mask'] is not None:
        axes[0, 1].imshow(results['annotation_mask'], cmap='gray')
        axes[0, 1].set_title("Annotation Mask (filtered regions)")
        axes[0, 1].axis('off')
    else:
        axes[0, 1].axis('off')
    
    # Cleaned image (annotations removed)
    axes[0, 2].imshow(results['cleaned_image'], cmap='gray')
    axes[0, 2].set_title("Cleaned Image (annotations removed)")
    axes[0, 2].axis('off')
    
    # Detection result
    axes[0, 3].imshow(cv2.cvtColor(results['vis_image'], cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title(f"Detected Candidates ({results['num_candidates']})")
    axes[0, 3].axis('off')
    
    # Show first few ROIs
    for idx in range(min(4, len(results['rois']))):
        if idx < len(results['rois']):
            axes[1, idx].imshow(results['rois'][idx])
            axes[1, idx].set_title(f"ROI {idx+1} (224x224)")
            axes[1, idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(results['rois']), 4):
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig("candidate_detection_test.png", dpi=150, bbox_inches='tight')
    print("Saved visualization to: candidate_detection_test.png")
    plt.show()


if __name__ == "__main__":
    test_detector()