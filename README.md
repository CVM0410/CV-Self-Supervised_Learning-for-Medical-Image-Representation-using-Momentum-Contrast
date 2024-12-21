# Self Supervised Learning for Medical Image Representation using Momentum Contrast

This project explores the application of **Self-Supervised Learning (SSL)** techniques for medical image representation using the **CheXpert** dataset. The project leverages SSL frameworks such as **MoCo (Momentum Contrast)**, **SimCLR**, **BYOL**, and **SwAV** to classify multiple thoracic diseases. The primary goal is to minimize reliance on labeled data while addressing challenges such as class imbalance and uncertainty in medical imaging datasets.

---

## Repository Overview

This repository contains the following files:

- **`CV_Final_Project.ipynb`**: Jupyter notebook that implements the complete pipeline for training SSL models and fine-tuning them for multi-label classification on the CheXpert dataset.
- **`train_split.csv`**: Training split of the CheXpert dataset with associated labels.
- **`test_split.csv`**: Test split of the CheXpert dataset with associated labels.
- **`requirements.txt`**: List of dependencies required to run the project.

---

## Dataset

The project uses the **CheXpert** dataset, which contains chest X-rays annotated for 14 conditions. Due to data-sharing restrictions, the dataset must be downloaded separately. The dataset contains the following classes:

1. No Finding  
2. Enlarged Cardiomediastinum  
3. Cardiomegaly  
4. Lung Opacity  
5. Lung Lesion  
6. Edema  
7. Consolidation  
8. Pneumonia  
9. Atelectasis  
10. Pneumothorax  
11. Pleural Effusion  
12. Pleural Other  
13. Fracture  
14. Support Devices  

---

## Key Features of the Project

### 1. Preprocessing Pipeline
- **Image Processing**: Includes resizing, denoising, histogram equalization, and normalization.
- **Handling Missing Data**: Uncertain labels are treated as negative for simplified training.
- **Data Augmentation**: Model-specific augmentations such as random cropping, color jitter, Gaussian blur, and multi-crop for SwAV are used to improve robustness.

### 2. Self-Supervised Learning Methods
- **MoCo (Momentum Contrast)**: Utilizes a memory bank to generate negative pairs and a momentum encoder for stable updates.
- **SimCLR**: Relies on strong augmentations for contrastive learning with in-batch negative samples.
- **BYOL (Bootstrap Your Own Latent)**: Does not use negative pairs but enforces consistency between two augmented views of an image.
- **SwAV (Swapping Assignments between Views)**: Employs clustering-based learning with multi-crop augmentations.

### 3. Class Imbalance Mitigation
- **WeightedRandomSampler**: Adjusts sampling probabilities for imbalanced classes.
- **Focal Loss**: Focuses on hard-to-classify samples to improve minority class recall.

### 4. Optimization Techniques
- **Mixed Precision Training (AMP)**: Reduces memory usage and accelerates training.
- **AdamW Optimizer**: Improves weight regularization.
- **OneCycleLR**: Dynamically adjusts the learning rate to stabilize training.
- **Gradient Clipping**: Prevents exploding gradients during backpropagation.

---

## Results

### Evaluation Metrics
The models were evaluated using the following metrics:
- **AUC (Area Under the Curve)**: Measures the overall performance across all thresholds.
- **F1 Score**: Evaluates the balance between precision and recall.
- **Cohen's Kappa**: Assesses agreement between model predictions and ground truth.
- **Average Precision**: Emphasizes the precision of minority classes.

### Key Findings
- BYOL and MoCo showed strong performance across various conditions, particularly in low-data scenarios.
- Grad-CAM visualizations provided insights into model interpretability, ensuring predictions aligned with clinically relevant features.

---

## Requirements

Install the dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

### Usage

- Run the Notebook
- Place the train_split.csv and test_split.csv files in the same directory as the notebook.
- Launch the Jupyter notebook and follow the step-by-step instructions in CV_Final_Project.ipynb.

---

### Future Scope

- Segmentation Tasks: Incorporating segmentation during preprocessing could isolate clinically relevant regions (e.g., lungs, heart) and potentially improve model accuracy.
- Multi-Modality Analysis: Combining chest X-rays with other imaging modalities (e.g., CT scans) could enhance diagnostic accuracy.
- Real-Time Deployment: Developing frameworks for real-world clinical implementation to validate robustness and utility.

---

### Limitations

Due to limited computational resources and cost constraints:

- State-of-the-art results were not achieved.
- The models were not trained with extensive hyperparameter tuning or larger batch sizes.
- With advanced computational resources, this project could achieve significant improvements and become an ideal solution for medical image representation.

---

### Authors

This project was developed as part of NYU Computer Vision CS-GY 6643.
- Chirag Mahajan
- Mohammed Basheeruddin
- Shubham Goel
- Nirbhaya Reddy G

---

## References

1. **CheXpert Dataset**  
   Stanford Machine Learning Group. *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.*  
   [https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)

2. **Momentum Contrast (MoCo)**  
   He, Kaiming, et al. *Momentum Contrast for Unsupervised Visual Representation Learning.*  
   [https://arxiv.org/abs/1911.05722](https://arxiv.org/abs/1911.05722)

3. **SimCLR**  
   Chen, Ting, et al. *A Simple Framework for Contrastive Learning of Visual Representations.*  
   [https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)

4. **BYOL (Bootstrap Your Own Latent)**  
   Grill, Jean-Bastien, et al. *Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning.*  
   [https://arxiv.org/abs/2006.07733](https://arxiv.org/abs/2006.07733)

5. **SwAV (Swapping Assignments between Views)**  
   Caron, Mathilde, et al. *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.*  
   [https://arxiv.org/abs/2006.09882](https://arxiv.org/abs/2006.09882)

