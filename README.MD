# Bean Leaf Disease Classification

## Overview

This project implements a deep learning system for classifying bean leaf images into three categories:

- **Angular leaf spot**
- **Bean rust**
- **Healthy**

The project uses transfer learning with EfficientNetB0 and a FastAPI-ready model workflow. It was developed as part of a computer vision assignment focused on image classification, data augmentation, model fine-tuning, and evaluation.

## Dataset

This project uses the [Beans Leaf Disease Classification dataset](https://huggingface.co/datasets/AI-Lab-Makerere/beans), provided by AI Lab Makerere. The dataset contains images of bean leaves belonging to three classes: angular leaf spot, bean rust, and healthy leaves.

The dataset is divided into three subsets:

- **Training:** 1,034 images
- **Validation:** 133 images
- **Testing:** 128 images

### Download the Dataset

The dataset archives are hosted on Hugging Face:

- [Training images — train.zip](https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/train.zip)
- [Validation images — validation.zip](https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/validation.zip)
- [Test images — test.zip](https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/test.zip)

The image archives are intentionally excluded from this Git repository because the training archive exceeds GitHub's 100 MB individual file limit. Download and extract the archives locally before running the notebook.

### Installation and Extraction

From the project root, run the following commands in a Linux or WSL terminal:

```bash
mkdir -p data/beans
cd data/beans

wget -O train.zip \
  https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/train.zip

wget -O validation.zip \
  https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/validation.zip

wget -O test.zip \
  https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/test.zip

unzip train.zip
unzip validation.zip
unzip test.zip
```

After extraction, verify that the directory structure matches the following layout:

```text
data/beans/
├── train/
│   ├── angular_leaf_spot/
│   ├── bean_rust/
│   └── healthy/
├── validation/
│   ├── angular_leaf_spot/
│   ├── bean_rust/
│   └── healthy/
└── test/
    ├── angular_leaf_spot/
    ├── bean_rust/
    └── healthy/
```

The dataset loader expects the three class folders to be present inside each split directory. If extraction creates an additional nested directory, move the split folders so that they match the structure shown above.

### Dataset Exclusion

The downloaded dataset and ZIP archives are excluded from version control through `.gitignore`. Users must obtain the images from the source linked above before running the training or evaluation notebook.


## Model Architecture

The classifier uses EfficientNetB0 pretrained on ImageNet as its feature-extraction backbone. A custom classification head contains:

1. Global average pooling
2. A dense layer with 256 ReLU units
3. Dropout with a rate of 0.3
4. A three-unit softmax output layer

Images are resized to 224 × 224 pixels.

## Data Augmentation

The training pipeline uses four augmentation techniques:

- Random horizontal flipping
- Random rotation
- Random zoom
- Random contrast adjustment

Augmentation is applied to training images to improve generalization. Validation and test images are not randomly augmented.

## Training Strategy

Training uses two phases:

**Phase I — Frozen backbone**

The pretrained EfficientNetB0 feature extractor is frozen while the custom classification head is trained.

**Phase II — Fine-tuning**

The backbone is unfrozen selectively, with only the final layers eligible for training. Batch normalization layers remain frozen, and a lower learning rate is used to reduce the risk of disrupting pretrained features.

The selected Phase I checkpoint achieved the same reported validation accuracy as Phase II with a slightly lower validation loss.

## Evaluation Results

The selected Phase I model was evaluated on 128 held-out test images.

| Metric | Result |
|---|---:|
| Test accuracy | 95.31% |
| Test loss | 0.1617 |
| Correct predictions | 122 / 128 |

### Per-Class Accuracy

| Class | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Angular leaf spot | 39 | 43 | 90.70% |
| Bean rust | 42 | 43 | 97.67% |
| Healthy | 41 | 42 | 97.62% |

Angular leaf spot had the lowest class recall and is a useful target for further error analysis.

### Validation Results

The checkpoint comparison produced the following results:

| Metric | Phase I | Phase II |
|---|---:|---:|
| Validation accuracy | 92.48% | 92.48% |
| Validation loss | 0.1842 | 0.1853 |

These validation values are from the checkpoint comparison. The originally recorded Phase I epoch logs also contain a higher per-epoch validation accuracy of 93.23%; differences should be investigated if the evaluation datasets or checkpoint selection procedures changed.

## Evaluation Artifacts

The `results/` directory contains evaluation outputs, including:

- `confusion_matrix.png`
- `per_class_accuracy.csv`
- `phase1_accuracy.png`
- `phase1_loss.png`
- `phase1_history_reconstructed.csv`
- `sample_predictions.png`

The Phase I curves were reconstructed from the rounded metrics printed in the training logs because the original in-memory history was unavailable. They are approximate visualizations of the recorded epoch results.

## Technology Stack

- Python
- TensorFlow / Keras
- EfficientNetB0
- NumPy and Pandas
- Matplotlib and Seaborn
- scikit-learn
- FastAPI (for the planned inference API)

## Running the API

The FastAPI application can expose endpoints for service health checks and image classification once the application files and a valid trained model artifact are in place.

The expected model location is:

```text
models/best_model.keras
```

The existing selected checkpoint is stored at:

```text
notebooks/models/phase1_best.keras
```

Copy it to the expected model location only after confirming the application configuration and checkpoint compatibility.

## Limitations

- The test set contains 128 images, so reported performance may vary on a larger or different dataset.
- Angular leaf spot has lower recall than the other two classes.
- High classification confidence does not guarantee that a prediction is correct.
- The FastAPI inference workflow should be tested separately before the application is described as production-ready.

## Future Improvements

- Investigate misclassified angular leaf spot images.
- Evaluate on a larger, independent dataset.
- Add image upload validation and structured API error responses.
- Test model inference through FastAPI.
- Consider model optimization for deployment on resource-constrained hardware.
