# Plant Disease Vision Challenge

This project started from a simple challenge: can a computer look at a leaf image and say something useful about plant health?

The brief in [docs/Challenge.md](docs/Challenge.md) asked for a solution that could detect and quantify plant characteristics, especially leaf area and disease symptoms. Instead of trying to solve everything at once, this repository takes a practical path:

- First, learn to classify the disease or healthy condition from the image.
- Then, complement that prediction with a lightweight segmentation step that estimates healthy leaf area and diseased lesion area.

The result is a compact computer vision pipeline for the PlantVillage dataset that combines deep learning classification with classical image processing.

## The Story

The dataset looks friendly at first glance: clean backgrounds, labeled folders, thousands of images. But the real challenge appears when we inspect the class distribution.

- The dataset contains 15 classes from tomato, potato, and bell pepper leaves.
- It includes 20,638 images in total.
- The largest class is about 21 times bigger than the smallest one.

That imbalance matters. A model can look accurate overall while still doing a poor job on rare diseases. Because of that, the project was designed around two goals:

1. Learn a strong visual representation with transfer learning.
2. Reduce imbalance bias with weighted sampling and class-weighted loss.

## What Was Built

### 1. Disease Classification

An EfficientNet-B0 model was fine-tuned on PlantVillage using PyTorch and `timm`.

- Transfer learning from ImageNet
- Stratified train/validation/test split
- `WeightedRandomSampler` for class imbalance
- Class-weighted cross-entropy loss
- Two-stage training: frozen backbone first, then full fine-tuning

During the quick-run experiment documented in the notebooks:

- Train split: 2,800 images
- Validation split: 600 images
- Test split: 600 images
- Best validation accuracy: `0.8533`

### 2. Leaf Area and Lesion Segmentation

To address the challenge requirement beyond classification, notebook 4 adds an HSV-based segmentation step:

- Green pixels approximate healthy leaf area
- Brown/yellow pixels approximate lesion area

This is intentionally simple. It works well on PlantVillage because the images are captured under controlled conditions, and it gives an interpretable estimate of visible damage without requiring a full pixel-level annotated segmentation dataset.

### 3. Combined Inference

The final demo function, `predict_and_segment()`, returns:

- top-1 predicted class
- confidence score
- top-3 predictions
- estimated healthy leaf area percentage
- estimated lesion area percentage

So the output is not just "what disease is this?" but also "how much of the visible leaf seems affected?"

## Results

The most important outcome is that the model does more than memorize the dominant classes.

On the held-out test set from the current quick-run configuration saved in the notebooks:

- Test accuracy: `86.0%`
- Macro F1: `0.8341`
- Weighted F1: `0.8606`

Some of the strongest class-level F1 scores were:

- `Pepper__bell___healthy`: `0.9773`
- `Pepper__bell___Bacterial_spot`: `0.9643`
- `Tomato__Tomato_YellowLeaf__Curl_Virus`: `0.9570`
- `Potato___Early_blight`: `0.9180`
- `Tomato_healthy`: `0.9011`

The harder classes were mostly the visually similar ones or classes with low support:

- `Potato___healthy`: `0.5714`
- `Tomato_Early_blight`: `0.7123`
- `Tomato_Late_blight`: `0.7097`

That tells a believable story:

- The model is already useful.
- It is strongest on classes with distinctive visual patterns.
- It still struggles when symptoms overlap or when data is scarce.

## Why This Addresses the Challenge

The original challenge asked for detection and quantification of plant characteristics. This repository answers that in two layers:

- Classification identifies the likely disease category or healthy state.
- Segmentation estimates visible healthy area and lesion coverage.

In other words, the project does not treat the challenge as only a labeling problem. It also tries to measure the visual footprint of the disease.

This is still a prototype, not a field-ready agronomic system. The segmentation is rule-based, and the dataset comes from controlled lab-style images rather than noisy real farm scenes. But for a challenge solution, it demonstrates:

- practical implementation
- technical reasoning
- awareness of limitations
- a path toward improvement

## Visual Outputs

Generated artifacts are saved in `docs/`:

- `training_curves.png`
- `confusion_matrix.png`
- `segmentation_results.png`
- `inference_demo.png`

These figures help tell the story from training behavior to final predictions and visual lesion estimation.

## Repository Guide

- `notebooks/notebook1.ipynb`: dataset inspection and first checks
- `notebooks/notebook2_eda.ipynb`: exploratory data analysis and normalization statistics
- `notebooks/notebook3_train.ipynb`: model training
- `notebooks/notebook4_eval.ipynb`: evaluation, segmentation, and inference demo
- `docs/theoretical_basis.md`: conceptual explanation of the approach
- `models/`: saved model artifacts

## How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the training notebook:

```text
notebooks/notebook3_train.ipynb
```

3. Run the evaluation notebook:

```text
notebooks/notebook4_eval.ipynb
```

The evaluation notebook expects the checkpoint generated by notebook 3 in `models/best_model.pth`.

## Next Steps

If this project were extended beyond the challenge, the most valuable improvements would be:

- train on the full dataset instead of only the quick-run subset
- test on real-world field images with cluttered backgrounds
- replace HSV rules with a learned segmentation model
- add calibration and uncertainty reporting for safer predictions

## Conclusion

This repository tells a straightforward story: start from a real agricultural vision challenge, build a classifier that respects class imbalance, and go one step further by estimating leaf and lesion coverage.

The current system already reaches solid test performance on PlantVillage and produces interpretable outputs. More importantly, it shows how to connect machine learning results back to the original problem statement instead of stopping at a single accuracy number.
