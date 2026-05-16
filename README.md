# Segmentation Model Training Pipeline

## Module: `segmentation_model/dataset.py`

### Significance

This module defines the core dataset abstraction used across all segmentation training stages. It enforces a consistent directory structure and ensures that all datasets—regardless of source—are transformed into a standardized format suitable for model training. It also introduces lightweight augmentations used primarily during finetuning.

### Key Logic and Workflow

1. **Directory Structure Assumption**

```text
root/
 ├── images/
 └── masks/
```

2. **Image Processing**

- Images are read using OpenCV.
- Converted from BGR to RGB.
- Normalized to `[0, 1]`.
- Transposed into PyTorch format `(C, H, W)`.

```python
img = cv2.imread(...)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0
img = np.transpose(img, (2, 0, 1))
```

3. **Mask Processing**

- Masks are loaded in grayscale.
- Binarized to create a single-channel segmentation target.
- Expanded to shape `(1, H, W)`.

```python
mask = cv2.imread(..., 0)
mask = (mask > 0).astype(np.float32)
mask = mask[None, ...]
```

4. **Augmentation (Optional)**

- Gaussian noise injection
- Brightness scaling

```python
if np.random.rand() < 0.3:
    noise = np.random.randn(*img.shape) * 10

if np.random.rand() < 0.3:
    factor = np.random.uniform(0.9, 1.1)
```

### Customization and Extensibility

- Integrate geometric augmentations (flip, rotate)
- Replace with Albumentations for stronger pipelines
- Extend for multi-class segmentation

---

## Module: `segmentation_model/ema.py`

### Significance

Implements Exponential Moving Average (EMA) of model weights. EMA is used for validation and checkpointing to improve stability and generalization.

### Key Logic and Workflow

**Initialization**

```python
self.ema = copy.deepcopy(model).eval()
```

**Update Rule**

```python
v.copy_(v * self.decay + msd[k] * (1 - self.decay))
```

### Why EMA is Used

- Reduces variance in parameter updates
- Produces smoother convergence
- Typically yields better validation performance

### Customization and Extensibility

- Increase decay (e.g., `0.9999`) for smoother updates
- Decrease decay for faster adaptation to new data

---

## Module: `segmentation_model/metrics.py`

### Significance

Provides both training loss functions and evaluation metrics. Designed specifically to handle class imbalance and the frequent occurrence of empty masks in defect datasets.

### Loss Functions

```python
loss = (
    0.5 * BCE +
    0.3 * Dice +
    0.2 * Focal
)
```

- **Binary Cross Entropy (BCE)**: Stable baseline loss
- **Dice Loss**: Optimizes spatial overlap
- **Focal Loss**: Focuses on hard-to-classify regions

### Metrics

- `dice`, `iou`: segmentation quality
- `precision`, `recall`: error characterization
- `dice_split`: separates performance into:
  - empty masks
  - non-empty masks

```python
both_empty = (union == 0)
```

- `pred_empty_ratio`: detects prediction collapse

### Key Design Insight

Empty masks are common in defect detection. Standard metrics can be misleading, so split metrics are used to separately evaluate empty and non-empty cases.

### Customization and Extensibility

- Adjust threshold (default `0.5`)
- Rebalance loss weights
- Extend to multi-class segmentation metrics

---

## Module: Segmentation Training Pipeline (Curriculum Learning)

### Significance

Implements a staged training strategy where the model progressively learns from simpler to more complex datasets.

### Training Order

```text
NEU → DAGM → SEV → custom → GDX (final)
```

### Rationale

- Start with simpler datasets (NEU)
- Introduce synthetic variability (DAGM)
- Move to real-world industrial data (SEV)
- Finetune on project-specific data (custom)
- Final specialization on target dataset (GDXray)

This approach improves generalization and stabilizes convergence.

---

## Module: `train_neu.py`

### Role

Initial pretraining stage on the NEU dataset.

### Configuration

```python
model = smp.Unet("efficientnet-b2", encoder_weights="imagenet")
lr = 1e-3
epochs = 20
```

### Behavior

- Learns foundational defect features
- No augmentation applied during training

### Outputs

```text
models/neu.pt
EMA/neu.pt
```

---

## Module: `train_dagm.py`

### Role

Second stage of training using the DAGM dataset.

### Key Changes

```python
model.load_state_dict(torch.load("EMA/neu.pt"))
lr = 5e-4
epochs = 8
```

### Behavior

- Adapts model to synthetic defect patterns

### Outputs

```text
models/dagm.pt
EMA/dagm.pt
```

---

## Module: `train_sev.py`

### Role

Third stage using the Severstal dataset.

### Key Changes

```python
model.load_state_dict(torch.load("EMA/dagm.pt"))
lr = 3e-4
epochs = 20
```

### Behavior

- Learns real-world industrial defect complexity

### Outputs

```text
models/sev.pt
EMA/sev.pt
```

---

## Module: `train_custom.py`

### Role

Finetuning stage on project-specific dataset.

### Key Changes

```python
model.load_state_dict(torch.load("EMA/sev.pt"))
lr = 1e-4
augment = True
epochs = 12
```

### Behavior

- Adapts model to domain-specific weld defects
- Introduces augmentation to improve robustness

### Outputs

```text
models/custom.pt
EMA/custom.pt
```

---

## Module: `train_final.py`

### Role

Final training stage on the GDXray dataset.

### Key Changes

```python
model.load_state_dict(torch.load("EMA/custom.pt"))
lr = 5e-5
augment = True
epochs = 20
```

### Behavior

- Final specialization for deployment environment

### Outputs

```text
models/final.pt
EMA/final.pt
```

---

## Training Design Summary

### Curriculum Strategy

```text
generic → synthetic → real → custom → target
```

### Core Mechanics

- EMA weights propagated across stages
- Learning rate reduced progressively
- Augmentation applied only during finetuning

### Benefits

- Improved convergence stability
- Better cross-domain generalization
- Reduced overfitting

---

## Module: `unified_test_script.py`

### Significance

Provides a standardized evaluation framework across datasets and training stages.

### Key Workflow

1. **Model Loading**

```python
model = smp.Unet(..., encoder_weights=None)
model.load_state_dict(...)
```

2. **Inference**

```python
pred = torch.sigmoid(model(x))
```

3. **Metric Computation**

```python
dice, iou, precision, recall
dice_split
pred_empty_ratio
```

4. **F1 Score**

```python
f1 = 2*(P*R)/(P+R)
```

5. **Logging**

```text
logs/{model}_{dataset}_{mode}.csv
```

### Evaluation Modes

- **Own Dataset Testing**
  Evaluates model performance on its native dataset

- **Cross Dataset Testing**
  Evaluates generalization across other datasets

### Purpose

- Detect overfitting
- Measure transfer learning effectiveness
- Validate curriculum learning strategy

### Customization and Extensibility

- Add threshold sweeps
- Log per-image metrics
- Extend to ensemble evaluations

---

## Overall Segmentation Training Flow

```text
Dataset → SegDataset → Model (Unet)
→ Loss (BCE + Dice + Focal)
→ Optimizer (Adam)
→ EMA update
→ Validation (EMA)
→ Metrics logging
→ Best checkpoint saving
```

### Key Strengths

- EMA-based stabilization
- Curriculum learning pipeline
- Robust handling of empty masks
- Multi-dataset generalization capability

# Segmentation Training Commands

```bash
# Step 1: Pretraining on NEU
python segmentation_model/training/train_neu.py

# Step 2: Continue training on DAGM (loads NEU EMA)
python segmentation_model/training/train_dagm.py

# Step 3: Continue training on SEV (loads DAGM EMA)
python segmentation_model/training/train_sev.py

# Step 4: Finetune on custom dataset (loads SEV EMA)
python segmentation_model/training/train_custom.py

# Step 5: Final training on GDXray (loads custom EMA)
python segmentation_model/training/train_final.py

# Step 6: Unified evaluation (own + cross dataset testing)
python segmentation_model/training/unified_test_script.py


```

# Classification Model Training and Evaluation Pipeline

This section documents the classification subsystem of the weld defect detection pipeline. It is responsible for categorizing segmented defect regions into predefined defect classes using a deep learning model enhanced with spatial mask guidance and explainability modules.

---

## System Overview

The classification pipeline follows a structured workflow:

1. Dataset Preparation (image + mask fusion)
2. Model Training (custom → final)
3. Evaluation (metrics + confusion matrix + saved outputs)
4. Explainability (GradCAM++ / HiResCAM visualizations)

---

## Module: classification_model/train_classifier_custom.py

### Significance

Initial training stage for classification using project-specific (custom) dataset. Introduces mask-guided classification and handles class imbalance.

### Key Logic and Workflow

#### 1. Dataset Construction

Each sample combines grayscale image + segmentation mask:

```python
img3 = np.stack([img,img,img],0)
x = np.concatenate([img3, mask[None,...]],0)
```

→ Results in **4-channel input (RGB + mask)**

#### 2. Mask Regularization

```python
if random.random() < 0.5:
    mask = np.zeros_like(mask)
mask = mask * 0.3
```

- Prevents over-reliance on mask
- Forces model to learn visual features

#### 3. Data Augmentation

- Horizontal flip
- Rotation
- Brightness scaling
- Noise injection

#### 4. Class Imbalance Handling

```python
weights = [1.0/counts[y] for y in labels]
sampler = WeightedRandomSampler(weights, ...)
```

#### 5. Additional Class Weighting

```python
if class == "crack":
    weight *= 2.5
```

→ prioritizes critical defect type

#### 6. Model Architecture

```python
resnet18 → modified input (4 channels)
```

#### 7. Loss Function

```python
CrossEntropyLoss(weight=cw, label_smoothing=0.05)
```

#### 8. Optimization

- Optimizer: AdamW
- Scheduler: CosineAnnealingLR

#### 9. Model Selection

```python
score = f1 - 0.2 * val_loss
```

→ balances accuracy and calibration

### Outputs

- `models/custom_classifier.pt`
- `logs/train_log_custom.csv`

---

## Module: classification_model/train_classifier_final.py

### Significance

Final training stage using full standardized dataset.

### Differences from Custom Training

- Uses **final mask set** instead of custom masks
- Same architecture and pipeline
- Trained on broader dataset → improved generalization

### Key Behavior

- Continues learning with same strategy
- Produces final deployable classifier

### Outputs

- `models/final_classifier.pt`
- `logs/train_log_final.csv`

---

## Module: classification_model/test_classifier_custom.py

### Significance

Evaluation of custom-trained classifier.

### Key Logic

#### 1. Dataset (No Augmentation)

Uses same preprocessing as training but deterministic.

#### 2. Forward Pass

```python
probs = torch.softmax(out, dim=1)
pred = torch.argmax(out,1)
```

#### 3. Metrics Computed

- Accuracy
- Precision (macro)
- Recall (macro)
- F1 score

#### 4. Confusion Matrix

```python
confusion_matrix(gts, preds)
```

#### 5. Output Saving

```python
np.save(... probs)
np.save(... preds)
np.save(... gts)
```

### Outputs

- CSV metrics log
- NumPy arrays for ROC / visualization
- printed confusion matrix

---

## Module: classification_model/test_classifier_final.py

### Significance

Evaluation of final production classifier.

### Differences from Custom Test

- Uses final model
- Same evaluation pipeline
- Saves outputs with `final_*` naming

### Outputs

- `final_probs.npy`
- `final_preds.npy`
- `final_gts.npy`
- `test_log_final.csv`

---

## Module: classification_model/gradcampp_all.py

### Significance

Generates GradCAM++ heatmaps for interpretability.

### Key Logic

#### 1. Multiple Evaluation Modes

```text
final / custom × with_mask / no_mask
```

→ allows comparison of mask influence

#### 2. CAM Generation

```python
cam = GradCAMPlusPlus(...)
heat = cam(input_tensor=x)[0]
```

#### 3. Thresholding

```python
heat = heat * (heat > 0.2)
```

#### 4. Overlay

```python
show_cam_on_image(rgb, heat)
```

### Outputs

- Heatmap overlays saved per class

---

## Module: classification_model/hirescam_all.py

### Significance

Alternative explainability method with higher spatial fidelity.

### Differences from GradCAM++

- Uses `HiResCAM`
- Slightly stricter threshold:

```python
heat = heat * (heat > 0.3)
```

### Purpose

- Compare interpretability methods
- Validate localization behavior

---

## Module: classification_model/run_all.py

### Significance

Automates full classification workflow.

### Execution Pipeline

```bash
train final
test final
train custom
test custom
gradcam++
hirescam
```

### Key Logic

```python
r = os.system(command)
torch.cuda.empty_cache()
time.sleep(5)
```

- sequential execution
- GPU memory cleanup between stages
- stops on failure

### Purpose

- reproducibility
- one-command pipeline execution

---

## Classification Model Design Summary

### Input Representation

```text
3-channel grayscale image + 1-channel mask
```

→ enables **guided classification**

### Core Techniques

- Mask-guided learning
- Mask dropout (regularization)
- Class balancing (sampler + weights)
- Label smoothing
- Curriculum (custom → final)

### Training Strategy

```text
custom dataset → final dataset
```

### Explainability

- GradCAM++ → smoother maps
- HiResCAM → sharper localization

### Strengths

- robust to class imbalance
- avoids mask over-dependence
- interpretable predictions
- strong generalization across datasets

---

## Overall Classification Pipeline

```text
Dataset → ClsDataset → ResNet18 (4-channel input)
→ CrossEntropyLoss (weighted + smoothed)
→ Optimizer (AdamW) + Scheduler
→ Validation + Best checkpoint
→ Testing (metrics + confusion matrix)
→ XAI (GradCAM++, HiResCAM)
```

# Classification Pipeline Commands

```bash
# Step 1 — Train custom classifier
python classification_model/training/train_classifier_custom.py

# Step 2 — Test custom classifier
python classification_model/training/test_classifier_custom.py

# Step 3 — Train final classifier
python classification_model/training/train_classifier_final.py

# Step 4 — Test final classifier
python classification_model/training/test_classifier_final.py

# Step 5 — Generate GradCAM++ visualizations
python classification_model/gradcampp_all.py

# Step 6 — Generate HiResCAM visualizations
python classification_model/hirescam_all.py

# Optional — Run everything sequentially
python classification_model/run_all.py
```

# Dataset Preparation and Preprocessing

### Overview

The training pipeline relies on multiple datasets collected from different domains, each contributing distinct characteristics to the learning process. These datasets vary in terms of defect distribution, annotation style, and real-world complexity. To ensure consistent training and effective generalization, extensive preprocessing and dataset engineering were performed.

The segmentation pipeline uses the following datasets:

- **NEU** – controlled industrial dataset with labeled surface defects
- **DAGM** – synthetic dataset with algorithmically generated anomalies
- **SeverstalSteel** – real-world steel defect dataset with high variability
- **custom** – project-specific dataset aligned with weld inspection requirements
- **GDXray** – final target dataset representing deployment conditions

The classification pipeline uses:

- **custom** – curated dataset aligned with segmentation outputs
- **RIAWELC** – additional dataset used to enhance classification diversity and robustness

A key challenge across these datasets is the imbalance between defective and non-defective samples, especially for segmentation tasks where many datasets are heavily biased toward positive (defect-present) samples. Addressing this imbalance was critical to prevent model bias and prediction collapse.

---

## NEU Dataset Processing

### Initial Observation

Upon inspection, the NEU dataset was found to be 100% positive, meaning:

- Every image contained at least one defect
- Every corresponding mask had non-zero regions

This posed a significant issue:

- The model would never learn to predict "no defect" cases\*\*
- High risk of false positives during inference
- Poor calibration for real-world scenarios where defects are sparse

---

### Strategy: Synthetic Negative Sample Generation

To address this, negative (empty-mask) samples were artificially created using spatial decomposition.

#### Method

- Each image was split into left and right halves
- Regions without visible defects were identified
- Corresponding mask regions (which were empty) were extracted
- These clean halves were used to construct new samples with no defects

This effectively transformed defect-containing images into valid background-only samples while preserving realistic texture and noise patterns.

---

### Dataset Balancing

After generating negative samples:

- The dataset was rebalanced to include both:
  - defect-present samples
  - defect-absent samples

The target distribution was:

```text
~30–35% empty masks
~65–70% defect masks
```

This ratio was chosen to:

- Avoid overwhelming the model with negatives
- Still provide sufficient exposure to "no defect" cases

---

### Augmentation Pipeline

To further increase diversity and robustness, augmentation was applied.

#### Transformations Used

- Random rotations
- Horizontal and vertical flips
- Additive Gaussian noise
- Blurring operations

#### Consistency Constraint

All spatial transformations were applied jointly to images and masks to preserve alignment:

```text
image transform == mask transform
```

This is critical for segmentation tasks, as any misalignment would corrupt supervision.

---

### Outcome

After preprocessing, the NEU dataset was transformed from a fully positive dataset into a balanced and diverse training set with:

- Realistic negative samples
- Improved variability
- Better representation of real-world conditions

This significantly improved:

- Model calibration
- False positive control
- Generalization to downstream datasets

---

## DAGM Dataset Preprocessing

### Initial Observation

Inspection of the DAGM dataset revealed the opposite issue compared to NEU:

- A large majority of images contained empty masks
- Very few samples had actual defect regions

This created a strong class imbalance skewed toward negative samples:

- Risk of the model learning to predict “no defect” for most inputs
- Poor sensitivity to actual defect regions
- Reduced effectiveness of segmentation learning

---

### Strategy: Positive Sample Amplification

To correct this imbalance, the focus was on increasing the representation of defect-containing samples.

#### Method

- Images with non-empty masks (i.e., containing defect boundaries) were identified
- Only these positive samples were selected for augmentation
- Augmentation was applied repeatedly to expand their presence in the dataset

---

### Augmentation Pipeline

The following transformations were applied to defect-containing images:

- Random rotations
- Horizontal and vertical flips
- Additive Gaussian noise
- Blur transformations

#### Mask Consistency

To maintain spatial alignment:

- Rotations and flips were applied to both image and mask
- Noise and blur were applied only to the image, not the mask

```text
spatial transforms → image + mask
intensity transforms → image only
```

This ensured that:

- Mask boundaries remained accurate
- Visual realism of images was preserved

---

### Dataset Rebalancing

After augmentation, the dataset distribution was adjusted to match a target balance:

```text
~30–35% empty masks
~65–70% defect masks
```

This mirrors the balancing strategy used in the NEU dataset, ensuring consistency across pretraining stages.

---

### Outcome

The DAGM dataset was transformed from a negative-dominant dataset into a balanced training resource with:

- Increased representation of defect patterns
- Improved diversity in defect appearance
- Reduced bias toward empty predictions

This step was critical in enabling the model to:

- Learn meaningful defect features from synthetic data
- Avoid collapse into trivial “no defect” predictions
- Transition effectively into more complex real-world datasets (Severstal)

---

## Severstal Steel Dataset Processing

### Initial Observation

The Severstal Steel dataset differed from previous datasets in two key ways:

- Images were not in the required 256×256 resolution
- Dataset was already reasonably balanced between defect and non-defect samples

This meant that unlike NEU and DAGM, no aggressive rebalancing was required. The primary challenge was ensuring spatial compatibility with the training pipeline.

---

### Strategy: Tiling for Resolution Standardization

Since the model expects fixed-size inputs (256×256), the dataset was processed using a tiling approach.

#### Method

- Large images were divided into smaller 256×256 patches
- Corresponding segmentation masks were tiled in the same manner
- Only aligned image-mask pairs were retained

This ensured:

- No distortion of defect structures (unlike aggressive resizing)
- Preservation of spatial fidelity
- Increased number of training samples

---

### Dataset Rebalancing

Severstal dataset was balanced according to the target balance after tiling so no balancing was done for this dataset.

### Outcome

The Severstal dataset contributed:

- Real-world defect variability
- Balanced class distribution
- High-quality segmentation signals without requiring heavy augmentation

---

## GDXray Dataset Processing

### Initial Observation

The GDXray dataset had the following characteristics:

- Balanced distribution between defect and non-defect samples
- Limited number of total samples
- Non-uniform image sizes

---

### Strategy: Tiling + Augmentation

#### Step 1: Tiling

As with Severstal:

- Images were divided into 256×256 patches
- Masks were tiled accordingly

#### Step 2: Data Augmentation

To address low sample count, augmentation was applied to all samples:

- Rotations
- Flips
- Gaussian noise
- Blur transformations

Mask handling followed the same rule:

```text
geometric transforms → image + mask
intensity transforms → image only
```

### Dataset Rebalancing

After augmentation, the dataset distribution was adjusted to match a target balance:

```text
~30–35% empty masks
~65–70% defect masks
```

### Outcome

- Increased dataset size significantly
- Maintained balanced distribution
- Improved robustness to noise and variation

This dataset serves as the final training stage and closely represents deployment conditions.

---

## Custom Dataset Processing (Segmentation)

### Initial Observation

The custom dataset showed:

- Strong bias toward defect-present samples
- Very few negative (empty mask) examples
- Limited dataset size

---

### Strategy

Two parallel objectives were addressed:

#### 1. Negative Sample Expansion

- Additional empty-mask samples were created
- Ensured the model learns "no defect" scenarios

#### 2. Dataset Augmentation

To increase diversity and size:

- Rotations
- Flips
- Noise
- Blur

Mask consistency rules were maintained:

```text
spatial transforms → image + mask
intensity transforms → image only
```

### Dataset Rebalancing

After augmentation, the dataset distribution was adjusted to match a target balance:

```text
~30–35% empty masks
~65–70% defect masks
```

### Outcome

- Balanced distribution between defect and non-defect
- Increased dataset size
- Improved generalization for project-specific patterns

---

## Classification Dataset Processing

### Datasets Used

- custom dataset
- RIAWELC dataset

---

### Class Structure Alignment

The two datasets originally had different class definitions:

RIAWELC classes:

```text
crack
lack of penetration
porosity
no defect
```

Custom dataset classes:

```text
air hole
bite edge
broken arc
crack
lack of fusion
overlap
slag inclusion
```

#### Unification Strategy

- All classes were merged into a unified folder-based structure
- Common classes (e.g., crack) were combined across datasets
- Each class mapped to a single directory

This ensured consistent label space for training.

---

### Image Representation Alignment

A critical discrepancy existed between datasets:

- Custom dataset used inverted X-ray images (defects appear bright)
- RIAWELC used standard X-ray representation (defects appear dark)

#### Solution

- Entire RIAWELC dataset was intensity-inverted
- This aligned visual representation across datasets

Additionally:

- The inference pipeline (Streamlit) performs:
  - input inversion before model inference
  - output re-inversion for visualization

This ensures consistency between training and deployment.

---

### Resolution Standardization

- Custom dataset: already 256×256
- RIAWELC dataset: originally 224×224 → upscaled to 256×256

This ensured compatibility with segmentation outputs and unified processing.

---

## Final Data Standardization Summary

### Segmentation

- All images standardized to 256×256
- Achieved via:
  - tiling (Severstal, GDXray)
  - resizing (where required)

### Classification

- All images standardized to 256×256
- Mask-guided inputs aligned across datasets
- Visual domain unified via inversion

---

## Overall Impact

These preprocessing steps ensured:

- Consistent input dimensions across all models
- Balanced representation of defect and non-defect samples
- Robustness to noise, variation, and dataset bias
- Alignment between training and inference pipelines

This unified preprocessing strategy is a key factor enabling the multi-stage training pipeline to function effectively across diverse datasets.

## Data Analysis, Metrics Visualization, and Model Diagnostics

This section covers the analytical and visualization utilities used to understand dataset distributions, validate preprocessing quality, and evaluate model performance across both segmentation and classification pipelines. These scripts transform raw logs and dataset statistics into interpretable plots, tables, and diagnostic artifacts.

---

## Dataset Distribution Analysis

### Classification Dataset Distribution

A utility script iterates through the classification dataset splits (`train`, `val`, `test`) and computes the number of samples per class.

### Key Output

```text
classification_distribution_all_splits.csv
```

This file contains:

- split (train / val / test)
- class name
- sample count

### Purpose

- Detect class imbalance
- Verify dataset merging correctness (custom + RIAWELC)
- Ensure uniformity across splits

---

### Segmentation Empty vs Non-Empty Analysis

A separate analysis computes the ratio of:

- empty masks (no defect)
- non-empty masks (defect present)

across all datasets and splits.

### Key Output

```text
segmentation_empty_stats.csv
```

Columns:

- dataset
- split
- empty count
- non-empty count
- empty ratio
- non-empty ratio

### Purpose

- Validate preprocessing balancing strategy
- Ensure target ratio (~30–35% empty) is achieved
- Detect dataset bias before training

---

## Classification Distribution Visualization

### Bar Plots

Generated for each split:

- Standard count distribution
- Log-scale distribution (for skew detection)
- Percentage-based distribution

### Outputs

```text
class_dist_train_normal.png
class_dist_train_log.png
class_dist_train_pct.png
... (same for val and test)
```

### Purpose

- Visual inspection of imbalance
- Detect rare classes
- Compare distribution consistency across splits

---

### Distribution Table

A pivot table is generated:

```text
classification_table.csv
classification_table.png
```

Structure:

- rows → classes
- columns → train / val / test / total

### Purpose

- Compact overview of dataset composition
- Useful for reporting and verification

---

## Segmentation Ratio Visualization

### Empty vs Non-Empty Ratio Plots

Stacked bar plots are generated for each split:

```text
empty_ratios_train.png
empty_ratios_val.png
empty_ratios_test.png
```

### Structure

- x-axis → dataset (neu, dagm, sev, custom, gdx)
- y-axis → ratio (0–1)
- stacked bars → empty vs non-empty

### Purpose

- Confirm balancing across datasets
- Ensure consistency between train/val/test
- Detect anomalies in preprocessing

---

## Classification Confusion Matrix Visualization

Two confusion matrices are generated.

### Clean Annotated Matrix

- Minimal styling
- Focus on numeric counts

```text
final_cm_clean.png
custom_cm_clean.png
```

### Purpose

- Identify misclassification patterns
- Analyze class confusion
- Evaluate class separability

---

## Classification Training Metrics Visualization

### Loss Curves

Plots training vs validation loss:

```text
final_loss.png
custom_loss.png
```

### Purpose

- Detect convergence behavior
- Identify overfitting

---

### Precision–Recall Curve

Plots recall vs precision across epochs:

```text
final_pr.png
custom_pr.png
```

### Purpose

- Evaluate trade-off between precision and recall
- Understand model bias

---

### Accuracy Progression

```text
accuracy_epoch.png
```

### Purpose

- Compare learning speed between models
- Track convergence

---

### Accuracy Comparison

```text
accuracy_compare.png
```

### Purpose

- Compare best-performing models
- Provide quick benchmark

---

## Advanced Classification Evaluation

A comprehensive analysis suite is implemented for deeper model diagnostics

### 1. Loss Curve Comparison

Side-by-side comparison of training and validation loss for custom and final models.

### 2. Accuracy Progression

Tracks validation accuracy across epochs for both models.

### 3. Precision / Recall / F1 Evolution

Monitors how each metric evolves during training.

### 4. Learning Stability

Uses rolling standard deviation of validation loss to measure training smoothness.

### 5. Overfitting Analysis

Plots generalization gap:

```text
val_loss - train_loss
```

### 6. Convergence Speed

Measures epochs required to reach a target accuracy threshold (e.g., 95%).

### 7. Final Loss Breakdown

Compares final training and validation losses.

### 8. Best Epoch Identification

Marks epoch with lowest validation loss.

### 9. Metric Correlation Heatmap

Shows relationships between:

- accuracy
- precision
- recall
- F1

### 10. Training Efficiency

F1 score per epoch comparison.

### 11. Test Set Comparison

Bar chart comparing final metrics between models.

### 12. Generalization Analysis

Compares validation performance vs test performance.

### 13. Improvement Analysis

```text
delta = final - custom
```

Shows gains achieved by final model.

### 14. Radar Chart

Multi-metric comparison in polar format.

### 15. Metric Table

Structured comparison of all metrics.

### 16. Training Trajectory

Scatter plot of:

```text
validation loss vs accuracy (colored by epoch)
```

### 17. Model Maturity

Compares final epoch vs test performance.

### 18. Consistency Check

Compares mean training metrics vs test metrics with variance.

### 19. Summary Table

Compact representation of:

- best epoch
- losses
- metrics

### 20. Complete Overview Dashboard

Combines multiple plots into a single visualization for quick inspection.

---

## ROC Curve Analysis

For each model:

```text
final_roc.png
custom_roc.png
```

### Features

- One-vs-rest ROC curves per class
- AUC scores displayed

### Purpose

- Evaluate separability per class
- Identify weak classes

---

## Overall Impact

These visualization tools provide:

- full visibility into dataset quality
- validation of preprocessing assumptions
- detailed insight into model training dynamics
- strong diagnostic capability for debugging and optimization

They form a critical layer between raw experimentation and final model deployment.
