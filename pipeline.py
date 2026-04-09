import torch
import cv2
import numpy as np
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import os

from candidate_detector import CandidateDetector


def run_full_pipeline(image_path, model_path="models/best_optimized_resnet18.pth"):
    """
    Complete pipeline: Image -> Candidates -> Classification -> GradCAM
    
    Args:
        image_path: Path to radiographic image
        model_path: Path to trained ResNet model
    """
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Class names (update if yours are different)
    class_names = ['Crack', 'LOP', 'No_Defect', 'Porosity']
    
    # Load model
    print("Loading model...")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_ftrs, len(class_names))
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    # Setup GradCAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load image
    print(f"\nProcessing: {image_path}")
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 1: Candidate Detection
    print("\n[1/3] Detecting candidates...")
    detector = CandidateDetector()
    detection_results = detector.detect(original_image, visualize=False, remove_annotations=True)
    
    num_candidates = detection_results['num_candidates']
    print(f"      Found {num_candidates} candidates")
    
    # Step 2 & 3: Classification + GradCAM for each candidate
    print(f"\n[2/3] Classifying candidates...")
    
    results = []
    
    for idx, roi in enumerate(detection_results['rois']):
        print(f"      Candidate {idx+1}/{num_candidates}...", end=" ")
        
        # Prepare image
        pil_img = Image.fromarray(roi)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        rgb_float = np.float32(roi) / 255
        
        # Classification
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)
        
        predicted_class = class_names[pred_idx.item()]
        confidence_score = confidence.item()
        
        print(f"{predicted_class} ({confidence_score:.1%})")
        
        # GradCAM (only for actual defects)
        gradcam_img = None
        if predicted_class.lower() != 'no_defect':
            heatmap = cam(input_tensor=input_tensor, targets=None)[0, :]
            gradcam_img = show_cam_on_image(rgb_float, heatmap, use_rgb=True)
        
        results.append({
            'candidate_id': idx + 1,
            'bbox': detection_results['bboxes'][idx],
            'class': predicted_class,
            'confidence': confidence_score,
            'roi': roi,
            'gradcam': gradcam_img,
            'probabilities': {
                class_names[i]: probabilities[0][i].item() 
                for i in range(len(class_names))
            }
        })
    
    # Step 3: Summarize defects
    print(f"\n[3/3] Summary:")
    defect_summary = {}
    for r in results:
        class_name = r['class']
        if class_name not in defect_summary:
            defect_summary[class_name] = 0
        defect_summary[class_name] += 1
    
    print("\n" + "="*60)
    print("DEFECT SUMMARY")
    print("="*60)
    for defect_type, count in defect_summary.items():
        print(f"  {defect_type}: {count}")
    print("="*60)
    
    # List each defect
    print("\nDETAILED RESULTS:")
    for r in results:
        if r['class'].lower() != 'no_defect':
            bbox = r['bbox']
            print(f"  • {r['class']} at ({bbox[0]}, {bbox[1]}) - {r['confidence']:.1%} confidence")
    
    return {
        'image_path': image_path,
        'num_candidates': num_candidates,
        'defect_summary': defect_summary,
        'results': results,
        'cleaned_image': detection_results['cleaned_image']
    }


if __name__ == "__main__":
    # Run pipeline
    image_path = "test_img3.jpg"  # Change to your image
    model_path = "models/best_optimized_resnet18.pth"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
    elif not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
    else:
        results = run_full_pipeline(image_path, model_path)