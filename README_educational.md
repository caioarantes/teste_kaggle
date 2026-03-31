# Plant Disease Vision Challenge

This project started from a simple challenge: can a computer look at a leaf image and say something useful about plant health?

The brief in [docs/Challenge.md](docs/Challenge.md) asked for a solution that could detect and quantify plant characteristics, especially leaf area and disease symptoms. Instead of trying to solve everything at once, this repository takes a practical path:

- First, learn to classify the disease or healthy condition from the image.
- Then, complement that prediction with a lightweight segmentation step that estimates healthy leaf area and diseased lesion area.

The result is a compact computer vision pipeline for the PlantVillage dataset that combines deep learning classification with classical image processing.

> **Background: What is a Computer Vision Pipeline?**
>
> A "pipeline" in machine learning refers to a sequence of processing steps where the output of one stage feeds into the next. In computer vision, a typical pipeline starts with raw image pixels and produces some structured output — a class label, a bounding box, a mask, or in this case both a label and a coverage estimate. The term "classical image processing" refers to algorithms that do not learn from data, but instead apply hand-crafted mathematical rules (like color thresholding) directly to pixel values. Deep learning, by contrast, learns those rules automatically from labeled examples.

---

## The Story

The dataset looks friendly at first glance: clean backgrounds, labeled folders, thousands of images. But the real challenge appears when we inspect the class distribution.

- The dataset contains 15 classes from tomato, potato, and bell pepper leaves.
- It includes 20,638 images in total.
- The largest class is about 21 times bigger than the smallest one.

![Class distribution](docs/class_distribution.png)

This plot makes the imbalance visible. A few classes dominate the dataset, while others appear only rarely. That is why the training pipeline uses weighted sampling and class-weighted loss instead of relying on raw accuracy alone.

That imbalance matters. A model can look accurate overall while still doing a poor job on rare diseases. Because of that, the project was designed around two goals:

1. Learn a strong visual representation with transfer learning.
2. Reduce imbalance bias with weighted sampling and class-weighted loss.

> **Background: Class Imbalance in Datasets**
>
> Class imbalance occurs when the number of training examples is not evenly distributed across the possible output categories. In this dataset, the ratio between the most frequent and least frequent class is approximately 21:1. This is considered severe imbalance.
>
> Why does this matter for training a model? During training, each image contributes a small error signal called the "loss gradient" that nudges the model's parameters in a better direction. When one class supplies 21 times more gradient updates than another, the model receives far more pressure to correctly classify majority-class examples. Over time, it learns a decision boundary that is well-suited for common classes but poorly calibrated for rare ones.
>
> The classic symptom is a model that achieves, say, 90% overall accuracy on a dataset where the majority class occupies 90% of examples — it could achieve that score by simply predicting the majority class every time, learning nothing useful about the minority classes.
>
> **Stratified Splitting** is a closely related concept. When dividing a dataset into train, validation, and test subsets, a purely random split on an imbalanced dataset can accidentally place all examples of a rare class into the training set (leaving none for evaluation) or, worse, place none in training at all. Stratified splitting preserves the class proportions in each subset, guaranteeing that every class is represented in every split according to its original frequency. This makes evaluation statistics meaningful and prevents the accidental exclusion of rare classes from any partition.

---

## What Was Built

### 1. Disease Classification

An EfficientNet-B0 model was fine-tuned on PlantVillage using PyTorch and `timm`.

- Transfer learning from ImageNet
- Stratified train/validation/test split
- `WeightedRandomSampler` for class imbalance
- Class-weighted cross-entropy loss
- Two-stage training: frozen backbone first, then full fine-tuning

> **Background: Key Terms at a Glance**
>
> - **PyTorch**: An open-source deep learning library that provides tensor computation (similar to NumPy but with GPU acceleration) and automatic differentiation for building and training neural networks.
> - **timm (Torch Image Models)**: A community-maintained library of pre-trained image model architectures for PyTorch. It provides ready-to-use implementations of modern architectures like EfficientNet with pretrained weights.
> - **Fine-tuning**: Adapting a pre-trained model to a new, related task by continuing training on the new dataset. This is the central technique of transfer learning, discussed in depth in the "Why Transfer Learning" section below.
> - **Frozen backbone**: During the first training phase, the parameters of the feature-extraction part of the network are held fixed (frozen) and only the new classification head is trained. This is explained further below.
> - **WeightedRandomSampler**: A PyTorch utility that draws training batches by sampling images with probabilities proportional to assigned weights, causing rare classes to be seen more often per epoch.

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

> **Background: What is Image Segmentation?**
>
> Image segmentation is the task of assigning a label to every pixel in an image, rather than assigning a single label to the entire image (which is classification). There are two main flavors:
>
> - **Semantic segmentation**: Every pixel is labeled with a class (e.g., "leaf", "lesion", "background"), but pixels of the same class are not distinguished from each other.
> - **Instance segmentation**: Individual objects of the same class are also separated from each other.
>
> Training a segmentation model requires pixel-level annotations — a mask image where each pixel is manually labeled. Creating these annotations is expensive and time-consuming. This project sidesteps that requirement entirely by using color-based thresholding in the HSV color space, which does not need any annotated masks and works directly from the raw pixels.

### 3. Combined Inference

The final demo function, `predict_and_segment()`, returns:

- top-1 predicted class
- confidence score
- top-3 predictions
- estimated healthy leaf area percentage
- estimated lesion area percentage

So the output is not just "what disease is this?" but also "how much of the visible leaf seems affected?"

---

## Why This Approach Makes Sense

The challenge looks simple when written in one sentence, but the image actually contains several layers of information. A healthy leaf is not defined only by its species. Disease appears through color shifts, texture changes, spots, mold patterns, necrotic regions, and the relative proportion of damaged tissue. That is why the project uses a hybrid pipeline instead of a single rule.

### From Pixels to Symptoms

A convolutional neural network is useful here because it learns visual patterns hierarchically:

- early layers detect edges, color gradients, and fine textures
- middle layers capture spots, veins, and lesion shapes
- deeper layers combine those clues into class-level disease signatures

This fits plant pathology well, because the difference between two classes may begin as a small color or texture cue and end as a recognizable disease pattern.

#### Concept: Convolutional Neural Networks (CNNs)

> A **Convolutional Neural Network** is a class of deep neural network designed specifically for grid-structured data like images. Its defining operation is the **convolution**: a small matrix of learnable numbers, called a **filter** or **kernel**, is slid across the image. At each position, the filter performs an element-wise multiplication with the underlying pixel values and sums the result into a single number. Repeating this across the full image produces a new 2D grid called a **feature map**, where each value represents "how much of the pattern this filter encodes was found at this location."
>
> Why does this work for images? Three properties make convolution well-suited to visual data:
>
> 1. **Local connectivity**: Pixels near each other are highly correlated (a leaf edge looks the same regardless of where in the image it appears). Convolution exploits this by connecting each output to only a small local region, not to every pixel simultaneously.
> 2. **Parameter sharing**: The same filter is applied at every spatial position. This means a filter that detects horizontal edges works everywhere in the image, dramatically reducing the number of parameters compared to a fully connected layer.
> 3. **Translational equivariance**: If a disease spot moves five pixels to the right, the feature map also shifts by five pixels, preserving spatial information through the network's depth.
>
> A deep CNN stacks many convolutional layers. Early layers learn generic low-level features (edges, color gradients). As depth increases, these are composed into mid-level features (textures, spots, shapes) and then high-level semantic concepts (disease categories). This hierarchical composition is precisely why CNNs are so effective at plant pathology: the visual signature of a disease is built from low-level color and texture cues combined at increasingly abstract levels.
>
> **Pooling layers** are interspersed between convolutional layers to progressively reduce spatial resolution, making the representation more compact and increasingly invariant to small positional shifts. **ReLU activation functions** introduce nonlinearity after each convolution, which is mathematically necessary for the network to approximate complex, non-linear mappings from pixels to class labels.

### Why Transfer Learning

Training a vision model from scratch would ask too much from a relatively modest labeled dataset. Transfer learning is the practical answer. The model begins with ImageNet-trained visual features and then adapts them to plant images.

The intuition is simple:

- general features like edges and textures transfer well across image domains
- fine-tuning converges faster than training from zero
- pretraining reduces the amount of data needed to reach useful performance

That is also why the training is split into two phases:

- first, train the classification head while the backbone stays frozen
- then, unfreeze the full network and fine-tune with a smaller learning rate

This stabilizes training early and avoids damaging the pretrained representation too aggressively.

#### Concept: Transfer Learning

> **Transfer learning** is the practice of taking a model trained on one task and reusing its learned representations as a starting point for a different but related task.
>
> The theoretical justification comes from observing that CNNs trained on large image datasets like ImageNet (which contains 1.2 million images across 1,000 classes) develop feature detectors in their early layers that are remarkably universal. Filters for edges, color blobs, Gabor-like textures, and other fundamental visual primitives emerge spontaneously regardless of the specific dataset. These low-level features are useful for virtually any image recognition task, including plant disease identification.
>
> Transfer learning exists on a spectrum between two extremes:
>
> - **Feature extraction**: The pretrained backbone is frozen entirely. Only a new classification head attached to the top is trained. The backbone acts as a fixed feature extractor, transforming images into rich representations that the head then maps to new class labels. This is fast and requires very little data, but the backbone cannot adapt to domain-specific visual patterns.
> - **Full fine-tuning**: All layers, including the backbone, are updated during training on the new dataset. This allows the network to adapt its representations to the new domain, but requires more data and careful control of the learning rate to avoid "catastrophic forgetting" — overwriting the useful pretrained features with noise.
>
> This project uses a **two-phase hybrid** approach, which is the most common practical strategy:
>
> **Phase 1 — Frozen backbone (feature extraction phase):** Only the new classification head is trained. This is done first because the head is randomly initialized and would produce large, noisy gradients if the whole network were updated simultaneously. Training only the head first lets it reach a reasonable starting point without disturbing the pretrained backbone.
>
> **Phase 2 — Full fine-tuning:** The entire network is unfrozen and trained with a much smaller learning rate. Small learning rates are critical here: if the learning rate is too large, the gradient updates will overwrite the carefully pre-learned features with task-specific noise, a phenomenon known as catastrophic forgetting. With a small learning rate, the backbone gently shifts its representations toward the plant domain while retaining the general visual knowledge from ImageNet.
>
> **Why does domain similarity matter?** ImageNet contains photographs of animals, vehicles, plants, furniture, and many other everyday objects. Plant leaves are natural images with textures, color gradients, and structured shapes that overlap substantially with the ImageNet distribution. This makes EfficientNet-B0's pretrained features a strong starting point. If the target domain were radically different — say, medical X-rays or satellite imagery — the transfer benefit would be smaller and more layers would need fine-tuning from scratch.

### Why EfficientNet-B0

EfficientNet-B0 was chosen because it offers a strong tradeoff between model capacity and practicality.

- it is compact enough to train comfortably in this project setup
- it still captures rich visual detail
- it is a strong transfer-learning baseline for 224x224 natural images

In storytelling terms, it is the right kind of tool for this stage of the project: strong enough to be credible, light enough to stay practical.

There is also a useful architectural idea behind EfficientNet. Instead of scaling only one thing, such as depth or width, EfficientNet scales three dimensions together:

- depth: how many layers the network has
- width: how many channels each layer uses
- resolution: how much visual detail enters the model

That balance matters because leaf disease cues live at different scales. Some are global color shifts, while others are local spots or edge irregularities. EfficientNet is designed to preserve that balance efficiently.

It also uses Squeeze-and-Excitation blocks, which act like channel-wise attention. In plain terms, the network learns which feature maps matter most and boosts them. For plant images, that can help amplify disease-relevant color and texture channels while suppressing less useful background information.

#### Concept: EfficientNet-B0 Architecture

> **EfficientNet** was introduced by Tan and Le (2019) with a key insight: previous methods for making CNNs more accurate simply increased one architectural dimension at a time — adding more layers (depth), making layers wider (more channels), or feeding higher-resolution inputs. Each of these gives diminishing returns when pushed too far alone.
>
> EfficientNet proposes **compound scaling**: simultaneously scaling depth, width, and resolution according to a fixed ratio, determined by a neural architecture search. The result is a family of models (B0 through B7) where each step up provides a consistent improvement in accuracy per unit of compute. EfficientNet-B0 is the baseline — the smallest and fastest of the family.
>
> **MBConv (Mobile Inverted Bottleneck Convolution)** is the core building block of EfficientNet. It was originally developed for MobileNetV2 and combines several efficiency tricks:
>
> 1. **Depthwise separable convolution**: Instead of applying a full convolution (which mixes spatial information and channel information simultaneously), the operation is factored into two cheaper steps: a depthwise convolution (spatial filtering, applied independently per channel) followed by a pointwise (1×1) convolution (channel mixing). This dramatically reduces the number of parameters and floating-point operations compared to a standard convolution of the same effective receptive field.
> 2. **Inverted residual structure**: In standard ResNet bottlenecks, the channel count is compressed and then expanded. MBConv does the opposite — it first expands the channel count (using a 1×1 convolution with an expansion factor, typically 6×), applies the depthwise convolution in the wider space, and then projects back to a smaller channel count. The intuition is that useful features are better separated and processed in a higher-dimensional space.
> 3. **Residual skip connection**: Like ResNet, MBConv adds the input directly to the output, creating a skip connection. This allows gradients to flow directly backward through many layers without vanishing, making deep networks much easier to train.
>
> **Squeeze-and-Excitation (SE) blocks** are a form of channel-wise attention. After the main convolution, SE blocks:
>
> 1. **Squeeze**: Global average pool the spatial dimensions of each feature map down to a single scalar per channel. This summarizes "how active" each channel is across the entire image.
> 2. **Excitation**: Pass those scalars through a small two-layer fully connected network (with a bottleneck) that outputs a weight between 0 and 1 for each channel.
> 3. **Scale**: Multiply each channel of the feature map by its learned weight.
>
> The effect is that the network dynamically recalibrates its feature maps based on global image content. For plant disease classification, this is practically useful: if the input is a tomato with yellow mosaic patterns, the SE block can boost channels that encode yellow-green texture patterns and suppress channels that respond to background clutter, without explicit programming of this logic.
>
> **Why B0 specifically?** B0 has approximately 5.3 million parameters and achieves around 77% top-1 accuracy on ImageNet at 224×224 resolution. For this project's scale (a few thousand training images, limited compute), B0 is the right tradeoff: large enough to learn discriminative disease features, small enough to train in reasonable time on a single GPU or even CPU, and well-supported by the `timm` library with high-quality pretrained weights.

### Why Class Imbalance Needed Special Treatment

The biggest trap in this dataset is imbalance. Some classes appear often enough that a model can learn them easily, while rare classes can be drowned out during training. If we only optimized for overall accuracy, the model could look better than it really is.

Two mechanisms were used together:

- `WeightedRandomSampler` makes minority classes appear more often during training
- class-weighted cross-entropy increases the penalty for mistakes on underrepresented classes

This combination matters because it shifts learning pressure toward the rare classes instead of letting the majority classes dominate every epoch.

The idea can be written simply. If a class appears less often, it receives a larger weight:

```text
sample weight ~ 1 / class count
```

So a rare class contributes more often during sampling and more strongly during loss computation. That does not magically solve imbalance, but it does stop the model from learning the easy majority classes first and forgetting the rest.

#### Concept: WeightedRandomSampler and Class-Weighted Cross-Entropy

> **WeightedRandomSampler** operates at the data loading level, before training begins. Every image in the training set is assigned a scalar weight, typically computed as:
>
> ```text
> weight_i = 1 / count(class of image_i)
> ```
>
> During each epoch, the data loader draws images by sampling without replacement according to these weights. An image from a class with 100 examples gets a weight of 1/100; an image from a class with 2,100 examples gets a weight of 1/2100. The sampler therefore draws the rare-class image roughly 21 times more often per epoch than it would under uniform sampling. From the model's perspective, each epoch feels like a more balanced dataset.
>
> An important subtlety: because the sampler oversamples minority classes, the "epoch" no longer corresponds to a single pass over the dataset. Some images are seen multiple times; others are never seen in a given epoch. The total number of samples per epoch is a fixed constant (usually set equal to the dataset size), but the distribution is reweighted.
>
> **Class-weighted cross-entropy loss** operates at the loss function level. Standard cross-entropy for multiclass classification is:
>
> ```text
> L = -log(p_true_class)
> ```
>
> where `p_true_class` is the predicted probability for the correct class. Class-weighted cross-entropy multiplies this by a per-class weight:
>
> ```text
> L_weighted = -w_c * log(p_true_class)
> ```
>
> where `w_c` is the weight assigned to class `c`. These weights are typically set to the inverse class frequency, normalized so they sum to the number of classes:
>
> ```text
> w_c = total_samples / (num_classes * count_of_class_c)
> ```
>
> A misclassification on a rare class now contributes a proportionally larger penalty to the total loss, generating a larger gradient update. This is complementary to weighted sampling: sampling changes which images are seen, while weighted loss changes how much the model is penalized for errors on those images. Using both together provides two independent mechanisms working in the same direction.
>
> **Why not just oversample by duplicating rare-class images?** Simple duplication (naive oversampling) causes the model to see the exact same images multiple times. It memorizes those specific examples rather than learning generalizable patterns. `WeightedRandomSampler` at least resamples from the available pool, though it still sees the same finite set of images. This is one reason data augmentation is applied simultaneously — to make each resampled view of a rare-class image look slightly different.

### Why Data Augmentation and Normalization Matter

Even though PlantVillage images are controlled, the model still benefits from seeing small variations during training.

- random flips help because leaves do not have a fixed orientation
- small rotations make the model less sensitive to pose
- color jitter helps simulate moderate lighting variation

Normalization was also computed from the dataset itself so the model sees a more consistent input distribution. None of these steps are flashy, but together they improve stability and generalization.

Another useful detail is that augmentation is label-preserving. Rotating or flipping a leaf does not change its disease label, so those transformations increase diversity without changing the meaning of the image. This is a practical way to make a modest dataset behave like a slightly larger one.

The same logic applies to normalization. Pixel values are centered and scaled channel by channel so that the network receives more stable inputs:

```text
normalized pixel = (pixel - mean) / std
```

That makes optimization smoother and helps the pretrained backbone adapt to the new dataset.

#### Concept: Data Augmentation

> **Data augmentation** refers to the practice of applying random, label-preserving transformations to training images during each epoch, so that the model sees slightly different versions of each image on every pass.
>
> The core insight is: for natural images, many transformations do not change semantic meaning. A tomato leaf with early blight, flipped horizontally, is still a tomato leaf with early blight. Rotating it 15 degrees does not change its diagnosis. Slightly shifting its brightness does not make it a different disease. These transformations are **label-preserving** by construction, but they expand the effective diversity of the training set.
>
> Common augmentations and their rationale:
>
> - **Random horizontal and vertical flips**: Leaves have no canonical orientation (they grow at arbitrary angles relative to the camera). Flipping creates a mirror image that is visually distinct but equally valid as a training example.
> - **Random rotation**: Similarly, slight rotation simulates the random orientation of leaves in images and prevents the model from using orientation as a spurious feature.
> - **Color jitter** (random changes to brightness, contrast, saturation, hue): Simulates natural variation in lighting conditions. A disease spot should be recognizable whether the image was taken in bright sunlight or under shade.
> - **Random crops and resizing**: Forces the model to recognize disease features at different scales and positions within the frame.
>
> Augmentation reduces **overfitting** — the failure mode where a model memorizes the training set rather than learning generalizable patterns. By making each epoch's data look slightly different, the model cannot memorize pixel-level details. It must instead learn structural features that persist across transformations.
>
> An important constraint: augmentation should only be applied to the **training** split. Validation and test images should be processed consistently (typically just resized and normalized) so that evaluation reflects true held-out performance rather than random variation introduced by augmentation.

#### Concept: Normalization

> **Input normalization** scales pixel values so that the input distribution to the network has zero mean and unit variance per channel. The formula is:
>
> ```text
> normalized pixel = (pixel_value - channel_mean) / channel_std
> ```
>
> This is applied independently to each of the three RGB channels. The mean and standard deviation are computed from the training dataset (or taken from the ImageNet statistics when using pretrained weights, since the network was trained on those statistics).
>
> Why does this help gradient descent? Neural network optimization relies on gradients — partial derivatives of the loss with respect to every weight. When input features have very different scales (e.g., one feature ranges from 0 to 255 while another ranges from 0 to 1), gradients in directions corresponding to large-scale inputs are proportionally larger. This creates a poorly conditioned loss landscape with elongated contours, where gradient descent oscillates inefficiently. Normalizing inputs brings all features to a comparable scale, making the loss landscape more isotropic and gradient descent much more stable and efficient.
>
> For transfer learning specifically, normalization to the same statistics used during pretraining is crucial. EfficientNet-B0 was pretrained on ImageNet, and its batch normalization layers (which appear throughout the architecture) have accumulated running mean and variance statistics calibrated to ImageNet-normalized inputs. If the new dataset is presented with a very different input distribution, those batch normalization statistics are invalid and the model's early layers produce activations outside the expected range. Using the same normalization statistics maintains the integrity of the pretrained representations.

### Why AdamW and Cosine Scheduling

Optimization is part of the theory too. This project uses AdamW because it combines adaptive learning rates with explicit weight decay, which tends to generalize better than plain Adam in many vision tasks.

The learning rate is then reduced with cosine annealing. Instead of dropping in a sudden step, it decreases smoothly over time. Conceptually, that means:

- early training can move quickly
- later training makes smaller corrections
- fine-tuning becomes less destructive to pretrained features

This pairs naturally with the two-phase training strategy.

#### Concept: AdamW Optimizer

> **Gradient descent** is the fundamental optimization algorithm for training neural networks. At each step, the gradient of the loss with respect to every parameter is computed (via **backpropagation**, which applies the chain rule of calculus through the computational graph), and each parameter is moved a small distance in the direction that reduces the loss:
>
> ```text
> parameter = parameter - learning_rate * gradient
> ```
>
> The **learning rate** controls step size. A rate too large causes divergence; a rate too small causes very slow convergence.
>
> **Adam (Adaptive Moment Estimation)** improves on basic gradient descent by maintaining two running averages per parameter:
>
> 1. **First moment (m)**: An exponential moving average of the gradient itself. This acts like momentum — smoothing out noisy gradients and accelerating movement in consistent directions.
> 2. **Second moment (v)**: An exponential moving average of the squared gradient. This measures the variance of gradients for each parameter. Parameters with consistently large gradients receive smaller updates; parameters with small gradients receive larger updates. This is the "adaptive" part — each parameter effectively has its own learning rate.
>
> The update rule (simplified) is:
>
> ```text
> m = beta1 * m + (1 - beta1) * gradient
> v = beta2 * v + (1 - beta2) * gradient^2
> parameter = parameter - learning_rate * m / (sqrt(v) + epsilon)
> ```
>
> **AdamW** (Adam with decoupled Weight Decay) fixes a subtle bug in standard Adam. L2 regularization is a technique that adds a penalty proportional to the squared magnitude of each weight to the loss, discouraging the network from assigning excessively large values to any parameter. In standard Adam, this L2 penalty interacts with the adaptive scaling (the `1/sqrt(v)` term), weakening the regularization effect for parameters with high gradient variance. AdamW decouples weight decay from the gradient-based update:
>
> ```text
> parameter = parameter - learning_rate * m / (sqrt(v) + epsilon) - weight_decay * parameter
> ```
>
> The second term applies weight decay directly and uniformly, independent of the adaptive scaling. This produces more consistent regularization and tends to improve generalization in practice.

#### Concept: Cosine Annealing Learning Rate Schedule

> A **learning rate schedule** changes the learning rate over the course of training. Starting with a high rate and reducing it over time is almost universally beneficial: high rates allow fast initial progress while the model is far from a good solution, and lower rates allow fine-grained convergence near the end of training without overshooting the minimum.
>
> **Step decay** is the simplest schedule: multiply the learning rate by a factor (e.g., 0.1) at fixed epochs. The problem is that the loss landscape changes abruptly at each step, which can be destabilizing.
>
> **Cosine annealing** uses a smooth cosine curve to reduce the learning rate from its maximum value to near zero over T epochs:
>
> ```text
> lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
> ```
>
> where `t` is the current epoch and `T` is the total number of epochs. The cosine shape starts high (fast progress early), decreases most rapidly in the middle epochs (efficient traversal of the loss landscape), and slows as it approaches zero (gentle refinement near convergence). This smooth decay avoids the jarring loss spikes that step schedules can produce.
>
> For two-phase training, this is especially valuable: in Phase 2, when the backbone is unfrozen with a small learning rate, cosine annealing ensures that the learning rate decreases progressively rather than remaining constant, providing a natural curriculum from moderately aggressive fine-tuning at the start to very conservative adjustments at the end. This minimizes the risk of catastrophic forgetting in the late stages of fine-tuning.

### Why HSV Segmentation Was Added

Classification answers "what is this?" but the challenge also asks for quantification. That is where the second branch comes in.

Instead of training a full segmentation network without pixel-level annotations, the project uses HSV color thresholding:

- green ranges estimate healthy leaf tissue
- yellow/brown ranges estimate diseased lesion regions

This choice is not the most sophisticated possible, but it is aligned with the data. PlantVillage images have controlled backgrounds and lighting, which makes a color-space method surprisingly effective as a first quantification baseline.

This is also a good place to explain why HSV is preferred over RGB. In RGB, color and brightness are entangled. In HSV, they are separated into:

- hue: the actual color family
- saturation: how pure or vivid the color is
- value: brightness

That makes thresholding more intuitive. Healthy tissue usually stays in the green hue range, while lesions drift toward yellow, brown, or reddish tones. Because hue is separated from brightness, the segmentation is easier to reason about than a direct RGB rule.

In simplified form, the two coverage measures are:

```text
green coverage   = green-mask pixels   / total image pixels
disease coverage = lesion-mask pixels  / total image pixels
```

These are not clinical measurements, but they are useful interpretable proxies.

#### Concept: HSV Color Space

> An image stored in **RGB (Red, Green, Blue)** format represents each pixel as a triplet of values indicating how much red, green, and blue light is mixed to produce the observed color. RGB is convenient for display hardware, but it has a significant drawback for color-based analysis: the three channels are highly correlated with brightness. A green leaf in shadow has different R, G, B values than the same green leaf in bright sunlight, even though the "color" — in the perceptual sense — is the same. Writing a robust rule like "find all green pixels" in RGB requires handling this brightness variation explicitly, which complicates thresholding.
>
> **HSV (Hue, Saturation, Value)** is a cylindrical color model that separates these concerns:
>
> - **Hue (H)**: Encodes the "type" of color as an angle on a color wheel (0-360 degrees, or 0-180 in OpenCV's 8-bit representation). Red is near 0/360 degrees, yellow near 60, green near 120, cyan near 180, blue near 240, magenta near 300. Hue is the quantity you are referring to when you say a color is "green" or "yellow-orange."
>
> - **Saturation (S)**: Measures the purity or vividness of the color. A highly saturated color is vivid; a low saturation means the color is washed out or grayish. This lets you distinguish a vibrant green leaf from a pale, dull green.
>
> - **Value (V)**: Measures brightness — how much light the pixel reflects overall. A pixel with high value and low saturation is white; with low value and low saturation is black.
>
> The cylindrical model can be visualized as a cone or cylinder where the vertical axis is value (brightness), the radial distance from the center is saturation, and the angle around the axis is hue.
>
> **Why HSV is better for leaf segmentation:** Since hue is independent of value, a green leaf pixel has approximately the same hue whether the image was taken in bright or dim lighting. A simple threshold on the H channel (e.g., H between 35 and 85 degrees in the standard scale) captures most shades of green across different lighting conditions. In RGB, an equivalent rule would require a more complex three-dimensional inequality that implicitly handles brightness, and it would be harder to generalize. HSV makes the domain knowledge ("green is healthy, yellow-brown is diseased") directly expressible as simple interval conditions on a single channel.
>
> **Color conversion** from RGB to HSV is a deterministic mathematical transformation. In OpenCV (used in this project), it is a single function call: `cv2.cvtColor(image, cv2.COLOR_RGB2HSV)`. No learning is involved; it is purely a change of coordinate system for the same color information.

### How the Two Parts Work Together

The project can be read as a small decision pipeline:

```text
Input leaf image
    |
    +--> EfficientNet-B0 classifier
    |      -> top-1 prediction
    |      -> top-3 classes
    |      -> confidence scores
    |
    +--> HSV segmentation
           -> healthy leaf coverage
           -> lesion coverage
```

That is the key theoretical idea behind the repository: classification and quantification are related, but they do not need to be solved by exactly the same mechanism.

The broader architecture can also be read like this:

```text
Leaf image
    -> augmentation and normalization
    -> EfficientNet-B0 backbone
    -> classification head
    -> softmax probabilities

Leaf image
    -> HSV conversion
    -> green and lesion masks
    -> coverage percentages
```

One branch learns from data. The other encodes explicit color rules. Together they provide a more educational and interpretable solution than either branch alone.

> **Background: Hybrid (Learned + Rule-Based) Pipelines**
>
> Machine learning and classical algorithms are not mutually exclusive. Many production systems combine learned components (neural networks) with hand-crafted, deterministic rules (geometric algorithms, color thresholds, physics-based models). The learned component handles the complexity that is hard to specify by hand (recognizing the visual signature of 15 plant diseases). The rule-based component handles aspects where domain knowledge is directly applicable and where labeled training data is absent (estimating the proportion of green versus brown pixels does not require a single labeled image).
>
> This hybrid design philosophy also improves interpretability. The classification result comes from a neural network that is not easily inspectable, but the coverage percentages come from explicit color rules that a human can verify and adjust. If the segmentation gives a surprising result, one can inspect the thresholds directly. This kind of transparency is valuable both for debugging and for building trust with end users.

### How Success Was Measured

Because the classes are imbalanced, accuracy alone would hide important failures. The evaluation therefore emphasizes:

- precision and recall at the class level
- F1-score as the balance between the two
- macro F1 because it gives each class equal importance
- confusion matrices to reveal which diseases look similar to the model

For segmentation, the project uses visual inspection and coverage percentages rather than IoU-style metrics, because the dataset does not provide pixel-level ground-truth masks.

It helps to read the classification metrics this way:

- precision asks: when the model predicts a class, how often is it correct?
- recall asks: when that class is truly present, how often does the model find it?
- F1-score balances the two, so a model cannot look good by optimizing only one side

Macro F1 is especially important here because it treats each class equally, even when one class has far fewer examples than another. In an imbalanced dataset, that makes it more informative than accuracy alone.

#### Concept: Evaluation Metrics for Classification

> **Accuracy** is the simplest metric: the fraction of all predictions that are correct. While intuitive, it is misleading for imbalanced datasets. A model that always predicts the majority class achieves accuracy equal to the majority class proportion, while learning nothing about any other class.
>
> To understand the richer metrics, consider the predictions for a single class C treated as a binary problem (C vs. not-C):
>
> - **True Positives (TP)**: Images that belong to class C and are predicted as C. Correct positive detections.
> - **False Positives (FP)**: Images that do not belong to class C but are predicted as C. False alarms.
> - **False Negatives (FN)**: Images that belong to class C but are predicted as something else. Missed detections.
> - **True Negatives (TN)**: Images that do not belong to class C and are not predicted as C. Correct rejections.
>
> From these four counts:
>
> ```text
> Precision = TP / (TP + FP)
> Recall    = TP / (TP + FN)
> ```
>
> **Precision** answers: "Of all the times the model said 'this is class C', how often was it right?" A high-precision model is cautious — it only claims a disease is present when it is quite sure.
>
> **Recall** (also called sensitivity or true positive rate) answers: "Of all the actual class C images, how many did the model correctly identify?" A high-recall model is thorough — it rarely misses true positives, though it may produce false alarms.
>
> There is an inherent tradeoff: increasing one often decreases the other (by adjusting the decision threshold). The **F1-score** is the harmonic mean of precision and recall:
>
> ```text
> F1 = 2 * (Precision * Recall) / (Precision + Recall)
> ```
>
> F1 is high only when both precision and recall are high. It is zero if either is zero.
>
> For multi-class problems, we compute per-class precision, recall, and F1 and then average them. Two common averaging strategies:
>
> - **Macro averaging**: Compute the metric for each class independently and take the unweighted average. Every class contributes equally, regardless of how many examples it has. This is the right choice when all classes are equally important, which is the case here — a rare disease is no less serious than a common one.
>
> - **Weighted averaging**: Compute the metric per class and take a weighted average, where the weight of each class is proportional to its support (number of true examples in the test set). This gives more influence to frequent classes. It produces an overall metric that reflects performance on the data distribution, but can hide poor performance on rare classes.
>
> In this project, **macro F1** is the primary metric because it penalizes poor performance on rare classes equally to poor performance on common ones. **Weighted F1** is reported alongside it as a secondary indicator.
>
> **Confusion Matrix** is a square matrix of size (num_classes × num_classes). Each row represents the true class and each column represents the predicted class. The diagonal entries show correct predictions; off-diagonal entries show the specific errors. A confusion matrix reveals which classes are being confused with which — for example, if "Tomato Early Blight" is frequently predicted as "Tomato Late Blight", that indicates those two diseases have visually overlapping features. This diagnostic information is not available from aggregate metrics alone.

---

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

### Training Behavior

![Training curves](docs/training_curves.png)

The training curves show a healthy learning pattern. Accuracy rises sharply after the backbone is unfrozen, while validation loss continues to fall instead of diverging. That suggests the model is learning useful visual features rather than simply overfitting the small quick-run subset.

### Confusion Matrix

![Confusion matrix](docs/confusion_matrix.png)

The confusion matrix shows that many classes are predicted cleanly, especially the healthier and more visually distinctive categories. The main errors are concentrated in disease pairs with similar texture and color patterns, which is exactly where we would expect a compact baseline model to struggle.

---

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

---

## Methodology

All metrics, plots, and visual outputs reported in this README — including test accuracy, F1 scores, training curves, confusion matrices, segmentation examples, and the inference demo — are the actual outputs produced by running the notebooks in this repository. They were not simulated, manually constructed, or estimated post-hoc. Any reader can reproduce them by following the steps in the [How To Run](#how-to-run) section.

This solution was developed with support from a combination of proprietary LLM models used as engineering assistants during the workflow. In practice, those models helped accelerate tasks such as structuring notebooks, refining explanations, debugging implementation details, and improving documentation quality.

That said, the final pipeline is still grounded in standard, testable computer vision practice:

- transfer learning with EfficientNet-B0
- imbalance handling with weighted sampling and class-weighted loss
- reproducible notebook-based training and evaluation
- classical HSV segmentation for interpretable leaf and lesion estimates

It is also important to be transparent about originality. The PlantVillage dataset has been publicly available for years, and many open repositories, notebooks, and tutorials already address plant disease classification on it. Because of that, this specific solution is unlikely to be fully original in the sense of proposing a brand-new modeling idea.

Its value is instead in being technically sound, coherent with the challenge goals, and clearly documented. The contribution here is not claiming novelty, but showing a credible end-to-end implementation that:

- trains successfully
- produces measurable results
- connects predictions back to the original challenge requirements
- acknowledges both strengths and limitations

---

## Workflow Pros and Cons

This section provides a thorough analysis of every major design decision in this project. It is intended for students and practitioners who want to understand not just what was built, but why specific choices were made and what tradeoffs they entail.

### Overall Hybrid Workflow: CNN Classification + HSV Segmentation

**Pros**

- **Complementary information**: The CNN answers a categorical question ("which disease is this?") while the HSV step answers a quantitative question ("how much of the leaf is affected?"). Neither answers the other's question well on its own, so the combination provides genuinely richer output.
- **No pixel-level annotation required**: Training a semantic segmentation network (e.g., U-Net, DeepLabV3) would require dense pixel-level masks for every training image, which are expensive to produce. The HSV approach needs zero labeled masks.
- **Interpretability**: The segmentation result is fully transparent. A user or reviewer can understand exactly how the green and lesion coverage numbers are computed, and can verify them visually. The CNN's output, by contrast, is harder to explain.
- **Modularity**: The two branches are independent. The classifier can be improved or replaced without touching the segmentation branch, and vice versa. This reduces coupling and simplifies debugging.
- **Low additional compute cost**: HSV thresholding is essentially free in terms of compute time compared to running the CNN forward pass. Adding it to the pipeline adds negligible latency.

**Cons and Limitations**

- **Consistency gap**: The two branches can produce conflicting outputs. A leaf classified as "healthy" by the CNN might still show significant lesion coverage according to the HSV branch (or vice versa). There is no mechanism to reconcile these outputs, and the combined inference result may confuse users.
- **Rule-based branch fragility**: The HSV thresholds are manually tuned for PlantVillage's controlled imaging conditions. Under different lighting, cameras, or backgrounds, the segmentation quality can degrade sharply without warning. This fragility is invisible from the classification metrics.
- **Not a clinical measurement**: The coverage percentages are estimates based on color alone. They do not account for leaf occlusion, perspective distortion, overlapping leaves, or the actual three-dimensional structure of the leaf surface. They should not be interpreted as agronomically precise severity scores.
- **No joint training signal**: Because the two branches are independent, there is no end-to-end supervision that encourages them to agree. A fully integrated multi-task model trained with both classification and segmentation losses simultaneously would be more coherent.

**Alternatives to Consider**

- **Multi-task CNN with segmentation decoder**: Architectures like U-Net or DeepLabV3 can output both a class label and a pixel-level mask from a single forward pass, trained jointly. This requires pixel-level annotations but produces more consistent and accurate segmentations.
- **Weakly supervised segmentation with Class Activation Maps (CAMs)**: Techniques like Grad-CAM can localize the regions most responsible for the CNN's classification decision, providing a rough segmentation without pixel-level labels. This would be more tightly coupled to the classification branch than HSV thresholding.
- **Pretrained segmentation model**: A model pretrained on a large agricultural segmentation dataset could be fine-tuned here, leveraging existing annotated data from other sources.

---

### Transfer Learning Choice

**Pros**

- **Data efficiency**: The most important advantage. With only a few thousand training images, training a deep CNN from scratch is unlikely to converge to useful representations. Transfer learning provides a head start from 1.2 million ImageNet images.
- **Faster convergence**: Fine-tuning typically converges in far fewer epochs than training from scratch, since the early layers already contain useful visual features and do not need to be learned.
- **Lower risk of overfitting**: A model trained from scratch on a small dataset tends to memorize training examples. A pretrained model's backbone already generalizes across millions of images; fine-tuning adapts it rather than replacing it, preserving the generalization.
- **Well-established practice**: Transfer learning on ImageNet-pretrained models is the dominant paradigm for applied computer vision tasks with limited data. The technique is well-understood, well-supported by libraries, and consistently delivers strong results.
- **Accessible pretrained weights**: Libraries like `timm` provide high-quality pretrained weights for dozens of architectures, eliminating the need to train large models from scratch.

**Cons**

- **Domain shift risk**: ImageNet is dominated by photographs of everyday objects, not close-up botanical images. The low-level features (edges, textures) transfer well, but higher-level features may be tuned for object categories irrelevant to plant pathology. Some fine-tuning is always required.
- **Pretrained biases**: The pretrained model may have learned statistical biases from ImageNet that are irrelevant or even counterproductive for plant images (e.g., features that distinguish dog breeds do not help distinguish disease types).
- **Dependency on backbone quality**: The performance ceiling is partly constrained by the quality and breadth of the pretrained features. If the target domain is sufficiently different from ImageNet, a domain-specific pretrained model (e.g., one pretrained on a large plant image dataset) would be a better starting point.
- **Licensing and version sensitivity**: Pretrained models from public libraries can change between versions (different training procedures, data, or hyperparameters), which can affect reproducibility if the model version is not pinned.

**Alternatives to Consider**

- **Domain-specific pretraining**: Models pretrained on iNaturalist (a large natural history image dataset), PlantCLEF (plant classification challenge data), or similar botanical datasets might provide a more relevant starting point than ImageNet.
- **Self-supervised pretraining on the target dataset**: Methods like SimCLR, MoCo, or DINO can pretrain a model on the unlabeled portion of the target dataset using contrastive or self-prediction objectives, potentially producing better representations than ImageNet transfer for this specific domain.
- **Training from scratch with a very small architecture**: If the dataset were much larger (hundreds of thousands of images), training a compact custom architecture from scratch could be competitive, eliminating the domain shift concern entirely.

---

### EfficientNet-B0 Specifically

**Pros**

- **Parameter efficiency**: EfficientNet-B0 achieves strong accuracy at around 5.3 million parameters and 0.39 billion FLOPs. This means it trains quickly on modest hardware and is fast at inference time, which matters for practical deployment.
- **Strong baseline accuracy**: Despite its small size, B0 achieves ~77% top-1 on ImageNet, outperforming much larger models like VGG-16. This accuracy advantage comes from the compound scaling principle, which optimizes the tradeoff between model size and capacity more carefully than prior designs.
- **Squeeze-and-Excitation attention**: SE blocks add channel-wise attention at low additional cost. For plant images with heterogeneous disease patterns, the ability to focus on the most relevant feature channels can help the model distinguish visually similar diseases.
- **Excellent `timm` support**: EfficientNet-B0 is one of the best-supported architectures in `timm`, with multiple sets of high-quality pretrained weights and extensive community validation.
- **Appropriate for 224×224 inputs**: The model is designed for this resolution, which is standard across most transfer learning benchmarks and well-matched to PlantVillage image dimensions.

**Cons**

- **Not the highest accuracy option**: Larger EfficientNet variants (B4, B7) or transformer-based models (Vision Transformer, Swin Transformer) achieve substantially higher accuracy on ImageNet and competitive benchmarks. If maximizing accuracy is the priority and compute is available, B0 may not be the best choice.
- **MBConv design optimized for mobile deployment**: While MBConv blocks are computationally efficient, they were originally designed for resource-constrained mobile devices. Modern GPU hardware may extract more performance from architectures like ResNet or ConvNeXt that are better optimized for GPU parallelism.
- **Compound scaling diminishing returns at B0**: The full benefit of compound scaling is most visible when comparing larger EfficientNet variants to single-dimension-scaled baselines. At B0, the advantage over simpler baselines like ResNet-18 is less dramatic.
- **Sensitivity to implementation details**: EfficientNet's training recipe (learning rate, weight decay, augmentation policy) is known to be somewhat sensitive, and reproducing the published numbers requires careful hyperparameter matching.

**Alternatives to Consider**

- **ResNet-18 or ResNet-50**: Older but extremely well-understood architectures with predictable behavior. ResNet-50 has more parameters than B0 but is a highly reliable baseline with excellent pretrained weights.
- **MobileNetV3 or ShuffleNet**: Even smaller than EfficientNet-B0, suitable if inference speed is the primary concern (e.g., for deployment on edge devices in the field).
- **Vision Transformer (ViT-S or ViT-B)**: Transformer-based architectures have overtaken CNNs on many benchmarks. ViT models pretrained with DINO or CLIP provide especially rich representations, but require larger datasets or more sophisticated training procedures to fine-tune effectively on small datasets.
- **ConvNeXt-Tiny**: A modern CNN design that matches or exceeds transformer performance on many tasks while remaining fully convolutional and efficient. A strong contemporary alternative to EfficientNet-B0 for this scale of problem.

---

### Class Imbalance Strategy: WeightedRandomSampler + Weighted Loss

**Pros**

- **Dual intervention at complementary stages**: Sampling reweighting affects which data the model sees; loss reweighting affects how much each example matters for gradient updates. Using both together provides more consistent pressure toward rare classes than either mechanism alone.
- **Simple to implement and understand**: Both techniques are available as straightforward options in PyTorch. The weighting formula (`1 / class_count`) is intuitive and easy to audit.
- **No architectural changes required**: These techniques are applied at the training loop level without modifying the model architecture, making them easy to add to any existing pipeline.
- **Directly addresses the most dangerous failure mode**: Without imbalance handling, the model could achieve high accuracy by ignoring rare classes. Both techniques explicitly and measurably push against this tendency.

**Cons**

- **WeightedRandomSampler increases overfitting risk on rare classes**: By repeatedly showing the same rare-class images (since oversampling draws from a finite pool), the model may memorize those specific examples rather than learning general patterns for that class. Data augmentation partially mitigates this, but does not fully resolve it.
- **Optimal class weights are not obvious**: The standard `1 / class_count` formula is a sensible default, but it is not theoretically guaranteed to be optimal for any given dataset. Other formulations (square root of inverse frequency, effective number of samples weighting) have been proposed and may work better in specific cases.
- **Weighted sampling can be slow**: PyTorch's `WeightedRandomSampler` computes a weighted probability distribution over a large index, which can add overhead to data loading, particularly for very large datasets.
- **Weighted loss alone may not fully compensate for extreme imbalance**: With a 21:1 ratio, even a large loss weight cannot fully compensate for the fact that the model receives much less gradient information about the structure of rare-class decision boundaries.

**Alternatives to Consider**

- **SMOTE (Synthetic Minority Over-sampling Technique)**: Instead of simply resampling existing images, SMOTE generates synthetic training examples by interpolating in feature space between existing minority-class examples. For image data, this is less straightforward than for tabular data, but image-space SMOTE variants exist.
- **Focal Loss**: A modification of cross-entropy that automatically down-weights easy examples (those the model already predicts correctly with high confidence) and focuses gradient updates on hard, uncertain examples. Originally developed for object detection, it handles imbalance without explicit class counting.
- **Class-balanced batch sampling**: Rather than using inverse-frequency weights, enforce that each training batch contains exactly the same number of examples from each class. This is stronger than weighted sampling but reduces the diversity within each batch.
- **Ensemble methods**: Training multiple models on different resampled subsets of the data and combining their predictions can be more robust than a single model with weighted sampling.

---

### HSV Segmentation for This Problem

**Pros**

- **No labeled data required**: The most significant advantage. Pixel-level annotation is the bottleneck for supervised segmentation. This approach completely bypasses that requirement.
- **Fully interpretable and adjustable**: Every decision — which hue ranges map to healthy tissue versus lesion — is an explicit, human-readable threshold. Domain experts (agronomists) can inspect and adjust the thresholds without machine learning knowledge.
- **Effective for controlled imaging conditions**: PlantVillage images have clean, consistent backgrounds and relatively uniform lighting. Under these conditions, color-based thresholding performs surprisingly well as a first-pass estimate.
- **Computationally trivial**: The entire segmentation runs in milliseconds per image, compared to the hundreds of milliseconds (or more) required for a forward pass through a segmentation network.
- **Provides a natural baseline**: Even a sophisticated learned segmentation model should be evaluated against this simple baseline to confirm that the added complexity is actually necessary.

**Cons**

- **Fragile to lighting variation**: The relationship between a physical color and its HSV representation depends on consistent imaging conditions. Outdoor farm photography under varying sunlight, cloud cover, and camera settings will shift hue values significantly, causing the fixed thresholds to fail.
- **Background and non-leaf pixel contamination**: If background objects (soil, stakes, other leaves) have similar hue to the thresholded ranges, they will be incorrectly included in the coverage estimates. PlantVillage's controlled backgrounds minimize this, but it is a fundamental limitation of the approach.
- **Cannot distinguish disease type from color alone**: HSV thresholding identifies "yellowing" or "browning" but cannot distinguish which disease is causing it. Some diseases (e.g., nitrogen deficiency) cause yellowing that is not a disease at all. The segmentation output is not clinically meaningful in isolation.
- **Hue ranges require manual tuning**: The thresholds that work well for tomato leaves may not work for potato or bell pepper leaves, which have somewhat different natural coloration. Systematic tuning requires domain expertise.
- **No uncertainty estimate**: The method produces a binary mask with no confidence or uncertainty. A pixel is labeled "lesion" or "not lesion" with no gradation, even in ambiguous regions like the boundary between healthy and diseased tissue.

**Alternatives to Consider**

- **Learned segmentation (U-Net, DeepLabV3+)**: If pixel-level annotations could be obtained for even a small subset of images, a learned segmentation model would generalize far better to new imaging conditions and disease types.
- **Weakly supervised segmentation with Grad-CAM**: Gradient-weighted Class Activation Maps generate rough localization maps from the classification network without additional annotation. These maps highlight the image regions most responsible for the predicted class, providing a rough segmentation proxy that is informed by disease identity rather than just color.
- **Adaptive thresholding**: Instead of fixed global thresholds, compute thresholds adaptively based on the color distribution of each individual image. This can partially compensate for lighting variation, though it introduces other challenges.
- **Normalized vegetation indices**: In remote sensing and agricultural monitoring, indices like the Normalized Difference Vegetation Index (NDVI) quantify plant health using ratios of spectral bands. For RGB cameras, simpler proxies like the Excess Green Index (ExG = 2G - R - B) can partially separate vegetation from non-vegetation without the fragility of HSV thresholds.

---

## Alignment With Evaluation Criteria

The challenge defines five evaluation criteria. This project addresses each one explicitly.

### 1. Code Quality

The implementation is organized as a small, reproducible workflow rather than a single monolithic script.

- separate notebooks for inspection, exploratory analysis, training, and evaluation
- saved artifacts in `models/` and `docs/`
- clear dependency list in `requirements.txt`
- reusable inference function combining classification and segmentation

The goal is not just to make the model work once, but to make the pipeline understandable and repeatable.

### 2. Technical Foundation

The core modeling choices are technically standard and well justified for this problem:

- transfer learning with EfficientNet-B0 for image classification
- stratified data splitting
- `WeightedRandomSampler` and class-weighted loss to address heavy imbalance
- multiple evaluation metrics, including macro F1, instead of raw accuracy alone
- HSV-based segmentation as a lightweight and interpretable quantification method

This makes the project technically sound even if it is not methodologically novel.

### 3. Solution Creativity

The project is not presented as a novel research contribution, since PlantVillage has been used extensively in public repositories and tutorials. The creative part is in how the challenge was framed and combined:

- disease classification for diagnosis
- rule-based lesion/leaf-area estimation for quantification
- a single end-to-end demo that reports both category and visible damage estimate

That combination makes the output more useful than a classifier alone while staying realistic for the available data.

### 4. Communication Skills

The repository is designed to communicate the work clearly to a reviewer, not only to run code.

- the README explains the problem, approach, results, and limitations in plain language
- visual outputs are embedded and interpreted, not just attached
- notebooks show the progression from exploration to training to evaluation
- the methodology section is transparent about both tooling and originality

This matters because a challenge solution should be understandable by someone who did not build it.

### 5. Practical Implementation Considerations

This is a prototype with a credible implementation path, but it is not yet a production deployment.

Practical strengths:

- runs in Python with common open-source libraries
- produces model outputs that are easy to interpret
- stores checkpoints and figures as deliverable artifacts
- can be extended into a small app or upload-based tool

Practical constraints:

- current reported metrics come from the notebook quick-run setup, not a full training cycle
- PlantVillage images are controlled and cleaner than real field conditions
- HSV segmentation is sensitive to lighting, background, and color variation
- real deployment would require validation on farm images, better calibration, and likely a learned segmentation model

In other words, the solution is practical as a challenge prototype and educational baseline, but more engineering and field validation would be needed before real agricultural use.

### Segmentation Examples

![Segmentation results](docs/segmentation_results.png)

These examples illustrate the second half of the challenge: quantification. The green mask estimates visible healthy leaf tissue, while the red/brown mask estimates lesion area. On healthy leaves, most pixels stay in the green region; on diseased leaves, the lesion mask expands over damaged areas. The method is simple, but it gives an interpretable visual estimate of symptom coverage.

### End-to-End Inference Demo

![Inference demo](docs/inference_demo.png)

This is the most practical output of the project. Each panel combines classification and quantification in one step: true label, predicted label, confidence, estimated leaf area, and estimated lesion area. Green titles indicate correct predictions, while red titles highlight mistakes. This makes it easy to see both what the model gets right and how it behaves when it is uncertain or confused.

---

## Repository Guide

- `notebooks/notebook1.ipynb`: dataset inspection and first checks
- `notebooks/notebook2_eda.ipynb`: exploratory data analysis and normalization statistics
- `notebooks/notebook3_train.ipynb`: model training
- `notebooks/notebook4_eval.ipynb`: evaluation, segmentation, and inference demo
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
- test different model architectures and compare their tradeoffs in accuracy, speed, and robustness
- perform systematic hyperparameter tuning for learning rate, batch size, augmentation, and regularization
- test on real-world field images with cluttered backgrounds
- replace HSV rules with a learned segmentation model
- add calibration and uncertainty reporting for safer predictions

## Conclusion

This repository tells a straightforward story: start from a real agricultural vision challenge, build a classifier that respects class imbalance, and go one step further by estimating leaf and lesion coverage.

The current system already reaches solid test performance on PlantVillage and produces interpretable outputs. More importantly, it shows how to connect machine learning results back to the original problem statement instead of stopping at a single accuracy number.
