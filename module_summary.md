# Module Summary Report — APA 7
## CNN Image Classification on Fashion-MNIST: A Controlled Dropout Experiment

---

**Title:** Convolutional Neural Networks for Clothing Image Classification: A Dropout Regularization Experiment on Fashion-MNIST

**Author:** Sébastien Bodrero

**Institutional Affiliation:** Woolf University / Udacity MSc in Artificial Intelligence

**Course:** AI Mastery — Module 4: Deep Learning Systems

**Date:** April 2026

---

## Overview

This report documents a controlled deep learning experiment for 10-class image classification using the Fashion-MNIST dataset (70,000 grayscale 28×28 images of clothing items). Two Convolutional Neural Network (CNN) configurations were implemented in PyTorch: a baseline SimpleCNN with no regularization, and an experimental variant adding a single Dropout(p=0.5) layer to the classifier. As Adhikari (2022) notes, a reproducible workflow "ensures that others can verify, build on, and extend [your] analysis" — fixed random seeds, pinned dependencies via `requirements.txt`, and automated dataset download are all implemented to this end. The experimental CNN + Dropout achieved 92.85% test accuracy versus 91.55% for the baseline, a +1.30 percentage point improvement that confirms dropout's effectiveness in reducing mild overfitting.

---

## Dataset and Task Description

The **Fashion-MNIST** dataset (Xiao et al., 2017) contains 70,000 grayscale 28×28 images across 10 clothing classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). The training set contains 60,000 images and the test set 10,000, with exactly 6,000 and 1,000 images per class respectively — a perfectly balanced distribution. The dataset is publicly available via `torchvision.datasets.FashionMNIST` and is downloaded automatically on first notebook run. Images are normalized using the dataset's population statistics (mean=0.2860, std=0.3530). The task is 10-class supervised image classification, appropriate for CNNs because they exploit spatial structure through learned local filters and weight sharing (LeCun et al., 1998). The dataset was not used in Modules 1–3 and is not synthetic.

---

## Model Architecture and Design Decisions

The **baseline SimpleCNN** follows a standard two-block feature extractor + linear classifier pattern:

- **Conv Block 1:** Conv2d(1→32, 3×3, padding=1) + ReLU + MaxPool(2×2)
- **Conv Block 2:** Conv2d(32→64, 3×3, padding=1) + ReLU + MaxPool(2×2)
- **Classifier:** Flatten → Linear(3136→256) + ReLU → Linear(256→10)
- **Total trainable parameters: ~824,458**

**Design rationale:**

| Decision | Justification |
|---|---|
| Two conv blocks | 28×28 → 14×14 → 7×7 spatial reduction; a third pooling would yield 3×3 — too small |
| Filter counts 32→64 | Standard doubling pattern: increases representational capacity as spatial size shrinks |
| ReLU activations | Avoids vanishing gradients; computationally efficient |
| Adam lr=0.001 | Adaptive learning rates per parameter; robust default for deep learning tasks |
| CrossEntropyLoss | Standard multi-class loss; combines log-softmax with NLL for numerical stability |
| Batch size 64 | Balances gradient variance (smaller) vs. throughput (larger) on CPU |

---

## Experimental Comparison

**One change from the baseline:** `nn.Dropout(p=0.5)` added after the ReLU activation of FC1 in the classifier.

**What changed:** A Dropout layer with probability 0.5 is inserted between FC1 and FC2.

**What stayed the same:** All convolutional layers, FC layer sizes (3,136→256→10), optimizer (Adam, lr=0.001), loss function (CrossEntropyLoss), batch size (64), epochs (10), random seed (42).

**Why dropout was selected:** The baseline training logs revealed a diagnostic pattern of mild overfitting: training loss fell steeply (0.4009 → 0.0535 over 10 epochs) while test accuracy plateaued after epoch 6 (~91.5%). This divergence is a classic overfitting signature. Dropout is the most direct intervention: by randomly zeroing 50% of FC1 activations per forward pass during training, it prevents neuron co-adaptation and forces distributed representations (Srivastava et al., 2014). At inference, dropout is disabled and outputs are scaled by (1 − p) to preserve expected magnitude.

---

## Results and Interpretation

| Metric | Baseline SimpleCNN | CNN + Dropout |
|---|---|---|
| Final train loss | 0.0535 | 0.1358 |
| Test accuracy | 91.55% | **92.85%** |
| Difference | — | **+1.30 pp** |

The CNN + Dropout model outperforms the baseline by +1.30 percentage points. The higher training loss (0.1358 vs. 0.0535) is expected — dropout introduces noise that prevents the model from achieving a near-zero training loss. But test accuracy is higher, confirming better generalization. The most dramatic per-class improvement is on T-shirt/top (+10.1 pp: 79.9% → 90.0%), with Pullover (+4.8 pp) and Sneaker (+3.4 pp) also improving substantially. The Shirt class remained the hardest category for both models (~78%), primarily confused with T-shirt/top and Coat — classes that share similar low-resolution visual structure and are inherently ambiguous without colour or texture information at 28×28 resolution.

### Interpretation for a Non-Technical Audience

Imagine trying to sort 10,000 photos of clothing items into 10 categories — without colour, at very low resolution. The first model (baseline) is like a student who has memorized the textbook answers too closely and occasionally fails on questions phrased slightly differently. The second model (CNN + Dropout) introduces a training trick: randomly "switching off" half the student's knowledge during each practice session, forcing them to learn more flexible strategies. The result: the second model classifies 92.85% of clothing items correctly versus 91.55% for the first — roughly 130 more correct predictions out of every 10,000. The hardest items to classify are Shirts and T-shirts, which look almost identical in a tiny black-and-white image.

---

## Limitations and Potential Bias

1. **No validation split:** The test set was used for per-epoch accuracy monitoring, violating strict train/val/test discipline. Results may be optimistic due to implicit selection.

2. **No data augmentation:** Standard augmentations (random flips, crops) were not applied. These would increase training diversity and likely reduce overfitting further.

3. **Single seed:** Results are from one training run. The +1.30 pp advantage of dropout may vary across random initializations; multiple seeds are needed for statistical confidence.

4. **Low-resolution grayscale limitation:** The Shirt class (~78% accuracy for both models) is fundamentally ambiguous at 28×28 grayscale. This is a dataset limitation, not a model failure.

5. **Surveillance misuse risk:** The CNN architecture generalizes directly to real-world clothing recognition from images. Deployment without consent in surveillance contexts would violate privacy rights. A bias audit on culturally diverse data is required before any real-world use.

---

## Reproducibility

- **Automated dataset download:** `torchvision.datasets.FashionMNIST(download=True)` — no manual steps required.
- **Fixed random seeds:** `torch.manual_seed(42)` and `np.random.seed(42)` set before all stochastic operations.
- **Pinned dependencies:** `requirements.txt` generated via `pip freeze`.
- **Top-to-bottom execution:** Validated via `jupyter nbconvert --to notebook --execute`.

Adhikari (2022) identifies these as foundational requirements for reproducible data science — all are implemented here.

---

## References

Adhikari, N. K. J. (2022). *Reproducible data science with Python: An open learning resource*. ResearchGate. https://doi.org/10.13140/RG.2.2.22099.04641

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, *86*(11), 2278–2324. https://doi.org/10.1109/5.726791

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, *15*(56), 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html

Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms*. arXiv. https://arxiv.org/abs/1708.07747
