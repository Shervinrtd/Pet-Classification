# 🐾 Pet Classification with PyTorch

This repository contains the solution for **Assignment Module 2: Pet Classification**. The goal of this project is to build and train deep learning models to classify images of **37 different breeds of cats and dogs** using the [Oxford-IIIT-Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

The project is implemented in **PyTorch** and is divided into two main parts: designing a Convolutional Neural Network (CNN) from scratch and fine-tuning a pre-trained ResNet-18 model.

## 👥 Authors

* **Omid Nejati**
* **Alireza Shahidiani**

## Project Structure

### Part 1: Design Your Own Network
In this section, we implemented a custom CNN architecture (`MyCNN_v2`) to train on the dataset from scratch.
* **Architecture:** A 5-block VGG-style network using `Conv2d`, `BatchNorm2d`, `ReLU`, and `MaxPool2d`.
* **Regularization:** Included `Dropout` in the fully connected classifier layers.
* **Ablation Study:** We performed ablation experiments to validate our design choices:
    * *Baseline Model*: Included both Batch Normalization and Dropout.
    * *Ablation 1*: Removed Batch Normalization (Resulted in the best performance for the custom model ~56%).
    * *Ablation 2*: Removed Dropout (Led to overfitting).

### Part 2: Fine-tune an Existing Network
In this section, we leveraged **Transfer Learning** using a pre-trained **ResNet-18**.
* **Part 2A:** Fine-tuned the model using the same hyperparameters from Part 1 (freezing the backbone, training only the head).
* **Part 2B:** Optimized the training strategy to surpass 90% accuracy. This involved:
    * Unfreezing all layers.
    * Using a 2-phase training approach (Warmup -> Fine-tuning).
    * Lowering the learning rate significantly (`1e-5`) to prevent destroying pre-trained weights.

## Dataset

The project automatically downloads the dataset from the course repository.
* **Source:** Oxford-IIIT-Pet Dataset.
* **Classes:** 37 breeds.
* **Split:** Custom Train/Val/Test split provided by the `ipcv-assignment-2` repository.
* **Preprocessing:** Images are resized to 256x256 and center-cropped to 224x224. Data augmentation (RandomResizedCrop, RandomHorizontalFlip) is applied to the training set.

##  Requirements

To run this notebook, you need the following Python libraries:
* `torch` & `torchvision` (PyTorch)
* `numpy`
* `matplotlib` (for visualization)
* `Pillow` (PIL)
* `seaborn` & `scikit-learn` (for confusion matrix plotting)

##  Usage

1.  **Open the Notebook:**
    Launch Jupyter Notebook or Google Colab and open `IPCV_assignment2.ipynb`.
2.  **Run the cells:**
    The notebook is self-contained. The first cell will verify your environment (GPU usage is highly recommended).
    * The dataset will be downloaded automatically via `!git clone ...`.
    * The training loops for Part 1 and Part 2 will execute sequentially.

##  Results Summary

| Model Variant | Description | Accuracy |
| :--- | :--- | :--- |
| **Custom CNN** | No Batch Norm (Best Custom) | **~56.2%** |
| **ResNet-18** | Frozen Backbone (Part 2A) | **~89.0%** |
| **ResNet-18** | Full Fine-Tuning (Part 2B) | **~89.8%** |

## 📝 License

This project is created for academic purposes as part of the Image Processing and Computer Vision course.
