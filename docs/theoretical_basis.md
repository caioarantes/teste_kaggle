# Theoretical Basis — Plant Disease Detection and Quantification

## 1. Problem Framing

The challenge consists of two coupled tasks applied to images of plant leaves:

1. **Leaf area quantification** — determine the proportion of the image covered by healthy leaf tissue.
2. **Disease classification** — identify the plant species and the disease (or healthy state) depicted.

The solution addresses both tasks with a hybrid pipeline: a deep learning classifier for task 2 and a classical computer vision segmenter for task 1.

---

## 2. Dataset Characteristics

### 2.1 PlantVillage

The dataset originates from the PlantVillage project, a large-scale effort to create a labeled image corpus for automated plant pathology. Images are collected under controlled conditions (uniform background, consistent lighting) and span tomato, potato, and bell pepper plants in both healthy and diseased states.

| Property | Value |
|---|---|
| Total images | 41,272 |
| Classes | 15 (3 healthy, 12 diseased) |
| Plants covered | Tomato, Potato, Pepper (Bell) |
| Image source | Laboratory / controlled field conditions |

### 2.2 Class Imbalance

The distribution is severely skewed: the largest class (*Tomato Yellow Leaf Curl Virus*, 6,416 images) contains approximately 21× more samples than the smallest (*Potato Healthy*, 304 images). Ignoring this imbalance biases training toward majority classes and inflates raw accuracy while masking poor recall on minority classes.

**Imbalance ratio** is formally:

$$\rho = \frac{N_{\max}}{N_{\min}} = \frac{6416}{304} \approx 21$$

---

## 3. Deep Learning Classification

### 3.1 Convolutional Neural Networks

CNNs learn hierarchical spatial representations by applying learned filter banks (convolutions) across the spatial dimensions of an image. Each layer captures increasingly abstract features:

- **Early layers** — edges, textures, color gradients.
- **Mid layers** — shapes, spots, lesion patterns.
- **Deep layers** — class-discriminative semantics.

This hierarchy makes CNNs naturally suited for plant disease recognition, where distinguishing features range from low-level color changes to high-level lesion morphology.

### 3.2 Transfer Learning

Training a large CNN from scratch requires millions of labeled examples and significant compute. Transfer learning reuses representations learned on a large general dataset (ImageNet, ~1.28 M images, 1,000 classes) and adapts them to the target task.

The strategy relies on two empirical observations:

1. **Feature generality** — early and mid convolutional filters (edges, textures) are domain-agnostic and transfer well to plant images.
2. **Data efficiency** — fine-tuning a pre-trained model converges faster and generalizes better when labeled target data is limited.

### 3.3 EfficientNet-B0

EfficientNet is a family of CNNs designed via *compound scaling*: width (channels), depth (layers), and resolution (input size) are scaled simultaneously using a fixed ratio determined by a neural architecture search. B0 is the base model in the family.

**Key properties:**

| Property | Value |
|---|---|
| Parameters | ~5.3 M |
| Input size | 224 × 224 |
| Top-1 accuracy (ImageNet) | ~77.1 % |
| MBConv block | Mobile inverted bottleneck + Squeeze-and-Excitation |

The Squeeze-and-Excitation (SE) mechanism performs channel-wise attention: it learns to re-weight feature maps so that disease-relevant channels are amplified and background channels suppressed.

### 3.4 Two-Phase Fine-Tuning

Directly unfreezing all layers from the start risks destabilizing the pre-trained representations (the gradient signal from the new classification head has high variance early in training).

**Phase 1 — Head-only training (epochs 1–3):**
The backbone is frozen; only the classification head is updated. This anchors the head's weights to a sensible initialisation before the backbone is exposed to gradients.

**Phase 2 — Full fine-tuning (epochs 4–7):**
All layers are unfrozen. The learning rate is reduced by 10× relative to phase 1 to apply small corrections to the backbone without catastrophic forgetting of ImageNet features.

### 3.5 Optimization and Regularization

**AdamW optimizer** combines the adaptive per-parameter learning rates of Adam with a decoupled weight decay term. Unlike standard Adam, which absorbs weight decay into the gradient update, AdamW applies it directly to the weights, yielding better generalization:

$$\theta_{t+1} = \theta_t - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) - \alpha \lambda \theta_t$$

**Cosine Annealing LR Scheduler** smoothly decays the learning rate following a cosine curve, avoiding abrupt drops and allowing the model to escape sharp minima late in training:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

---

## 4. Handling Class Imbalance

Two complementary mechanisms are applied simultaneously.

### 4.1 WeightedRandomSampler

At every training epoch, samples are drawn with replacement according to per-class weights:

$$w_i = \frac{1}{N_{c(i)}}$$

where $N_{c(i)}$ is the total count of the class of sample $i$. This makes each mini-batch approximately class-balanced regardless of the original distribution.

### 4.2 Class-Weighted Cross-Entropy Loss

Standard cross-entropy treats all classes equally. Weighted cross-entropy scales the loss contribution of each sample by the inverse class frequency:

$$\mathcal{L} = -\sum_{i=1}^{N} w_{c_i} \log p(c_i \mid x_i), \quad w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$

where $K$ is the number of classes. Combining both techniques at the sampling and loss levels provides redundant signal that strongly counteracts the imbalance.

---

## 5. Data Augmentation

Augmentation artificially expands the effective training set by applying label-preserving transformations. For leaf images the following transforms are used:

| Transform | Motivation |
|---|---|
| Random horizontal / vertical flip | Leaves have no canonical orientation |
| Random rotation ±15° | Rotational variation in field images |
| Color jitter (brightness, contrast, saturation ±0.2) | Lighting variability across images |

All transforms are applied only at training time; validation and test images are only resized and normalized.

### 5.1 Dataset-Specific Normalization

Normalizing pixel values to zero mean and unit variance with dataset-specific statistics (rather than ImageNet defaults) ensures the input distribution more closely matches what the model expects and can marginally improve convergence:

$$\hat{x}_c = \frac{x_c - \mu_c}{\sigma_c}, \quad \mu = [0.470, 0.481, 0.424], \quad \sigma = [0.198, 0.174, 0.211]$$

These statistics were estimated from a random sample of 225 images from the PlantVillage dataset.

---

## 6. Classical Computer Vision — HSV Segmentation

### 6.1 HSV Color Space

RGB encodes color as a mixture of red, green, and blue primaries. HSV separates chromatic information into three perceptually meaningful components:

- **Hue (H)** — the color angle (0–180° in OpenCV's 8-bit convention).
- **Saturation (S)** — color purity; low saturation means gray/white.
- **Value (V)** — brightness.

HSV is preferred over RGB for color-based segmentation because thresholds on hue are more robust to lighting changes than RGB thresholds.

### 6.2 Leaf Area Mask (Green Mask)

Healthy leaf tissue is characterized by chlorophyll, which absorbs red and blue light while reflecting green. In HSV space this corresponds to hues in the yellow-green range:

| Channel | Range |
|---|---|
| H | 25° – 90° |
| S | 40 – 255 |
| V | 40 – 255 |

The lower saturation/value bounds exclude near-white backgrounds and deep shadows.

**Green coverage** is the fraction of image pixels falling inside this mask:

$$\text{Green coverage} = \frac{\sum \mathbf{1}[\text{pixel} \in \text{green mask}]}{W \times H}$$

### 6.3 Disease Lesion Mask

Diseased tissue typically turns yellow, brown, or necrotic — hues outside the healthy green range. The disease mask targets brown/yellow and reddish tones:

| Channel | Range |
|---|---|
| H | 0° – 25° or 160° – 180° |
| S | 40 – 255 |
| V | 40 – 200 |

The upper value bound (200) excludes highly reflective specular highlights.

**Disease coverage** is computed analogously to green coverage.

### 6.4 Limitations

HSV thresholding is sensitive to imaging conditions. Backgrounds that share hue ranges with leaf tissue (e.g., brown soil, yellow light sources) can cause false positives. In the PlantVillage dataset, images have a uniform controlled background, which reduces this risk.

---

## 7. Evaluation Metrics

### 7.1 Classification

Because the dataset is imbalanced, raw accuracy is not a reliable indicator of model quality. The following per-class and aggregate metrics are used:

**Precision:** fraction of predicted positives that are truly positive.

$$P_c = \frac{TP_c}{TP_c + FP_c}$$

**Recall:** fraction of true positives that are correctly predicted.

$$R_c = \frac{TP_c}{TP_c + FN_c}$$

**F1-score:** harmonic mean of precision and recall, penalizing large discrepancies between the two.

$$F1_c = 2 \cdot \frac{P_c \cdot R_c}{P_c + R_c}$$

**Macro F1** averages F1 across all classes with equal weight, giving each class the same influence regardless of size — this is the most informative single metric under imbalance.

**Confusion matrix** provides a complete 15×15 view of which classes are confused, enabling targeted analysis of hard pairs (e.g., *Tomato Early Blight* vs. *Tomato Late Blight*).

### 7.2 Segmentation

Segmentation quality is assessed visually and by comparing mean green/disease coverage statistics across healthy vs. diseased categories (expected to diverge). Formal IoU metrics are not applicable because the dataset provides no pixel-level ground truth.

---

## 8. Combined Inference Pipeline

The final inference function merges both sub-systems:

```
Input image
     │
     ├──► EfficientNet-B0 → Top-3 classes + softmax probabilities
     │
     └──► HSV segmentation → Green coverage % + Disease coverage %
```

This dual output addresses both challenge requirements in a single call: disease identity from the classifier and leaf/lesion area from the segmenter.

---

## 9. Architecture Diagram

```
PlantVillage Images (224×224×3)
           │
    ┌──────▼───────┐
    │  Data Augment │  (flip, rotate, color jitter — train only)
    └──────┬───────┘
           │
    ┌──────▼───────────────────────┐
    │        EfficientNet-B0        │
    │  ┌─────────────────────────┐  │
    │  │  Backbone (MBConv + SE) │  │  ← frozen in Phase 1
    │  └────────────┬────────────┘  │
    │               │               │
    │  ┌────────────▼────────────┐  │
    │  │  Global Avg Pool         │  │
    │  └────────────┬────────────┘  │
    │               │               │
    │  ┌────────────▼────────────┐  │
    │  │  Dropout + Linear(15)   │  │  ← always trained
    │  └─────────────────────────┘  │
    └──────┬───────────────────────┘
           │
     Softmax → Top-3 predictions
           │
    ┌──────▼───────┐
    │ HSV Segment.  │  (parallel branch)
    └──────┬───────┘
           │
  Green % / Disease %

Loss: WeightedCrossEntropy
Optimizer: AdamW + CosineAnnealingLR
Sampler: WeightedRandomSampler
```

---

## 10. References

- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019.
- Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. arXiv:1511.08060.
- Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks*. CVPR 2018.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019.
- Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR 2017.
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR 16, 321–357.
