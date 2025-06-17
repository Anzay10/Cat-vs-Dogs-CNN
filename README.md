# cat_vs_dogs_classifier.ipynb

Image classification with CNN, performance metrics and uncertainty estimation
# Cat-vs-Dogs CNN Classifier

A complete deep learning pipeline for binary image classification on the **Kaggle Cats vs Dogs dataset**. This project implements a custom **Convolutional Neural Network (CNN)** and includes **uncertainty estimation using Monte Carlo Dropout**, making it suitable for research or real-world applications where confidence matters.

---

## Dataset

- **Source**: [Kaggle - Dogs vs. Cats Competition](https://www.kaggle.com/c/dogs-vs-cats/data)
- 25,000 labeled images of cats and dogs in JPG format
- Train/Validation split: 80/20 using `ImageDataGenerator`

---

## Model Architecture

| Layer            | Details                          |
|------------------|----------------------------------|
| Input            | (128, 128, 3) RGB images         |
| Conv2D + ReLU    | 32 filters, 3x3, same padding    |
| MaxPooling2D     | 2x2                              |
| Conv2D + ReLU    | 64 filters, 3x3, same padding    |
| MaxPooling2D     | 2x2                              |
| Conv2D + ReLU    | 128 filters, 3x3, same padding   |
| MaxPooling2D     | 2x2                              |
| Flatten          | -                                |
| Dense + ReLU     | 128 units                        |
| Dropout          | 0.5                              |
| Output (Sigmoid) | 1 unit — binary classification   |

**Optimizer**: Adam (lr = 0.001)  
**Loss**: Binary Crossentropy  
**Metrics**: Accuracy, Precision, Recall, AUC

---

## Final Performance (Validation Set)

| Metric      | Value     |
|-------------|-----------|
| Accuracy    | **86.0%** |
| AUC         | **0.938** |
| Precision   | 88.6%     |
| Recall      | 82.7%     |
| Loss        | 0.337     |

The model converged well by epoch 6, with a consistent increase in validation AUC and balanced performance.

---

## Uncertainty Estimation

Using **Monte Carlo Dropout (MC Dropout)** with 50 inference passes to capture confidence.

### Example:
- **0.77** → Mean predicted probability
- **± 0.14** → Standard deviation across 50 forward passes

High uncertainty values signal ambiguous images — useful for safety-critical or human-in-the-loop workflows.

---

## Results Visualization

- Training vs Validation Accuracy
- Training vs Validation Loss
- Precision & Recall across epochs
- Top 6 most uncertain test images plotted

---

## How to Run

1. Clone the repository or open the notebook in Google Colab
2. Upload the `dogs-vs-cats.zip` dataset from Kaggle
3. Run all cells in order — includes preprocessing, training, evaluation, and uncertainty estimation

---

## Requirements

```bash
pip install tensorflow numpy matplotlib pillow scikit-learn






