# Breast Cancer Diagnosis Classification with Balanced Dataset

A breast cancer classification project addressing class imbalance via under- and oversampling. Uses Random Forest and stratified k-fold cross-validation for robust evaluation. Includes data preprocessing, model training, evaluation, and saving balanced datasets for reproducibility.

## Overview

This project focuses on building a machine learning model to classify breast cancer diagnosis (Benign vs Malignant) using the Breast Cancer Wisconsin dataset. The main challenge addressed is **class imbalance** between the two diagnosis classes.

Machine learning algorithms generally perform poorly when training data is imbalanced, because they tend to be biased towards the majority class. Therefore, this project demonstrates techniques for balancing the dataset and evaluating model performance robustly.

---

## Why Data Balancing?

- **Class Imbalance Problem:**  
  The dataset contains more malignant samples than benign, causing models to favor the majority class.  
- **Impact:**  
  A biased model may have high overall accuracy but fail to detect the minority class accurately (which is often more critical in medical diagnosis).  
- **Solution:**  
  Apply **undersampling** and **oversampling** to balance class distributions before training models.

---

## Methods Applied

### 1. Data Exploration
- Loaded the dataset and examined class distribution.
- Split data into minority (Malignant) and majority (Benign) classes.

### 2. Undersampling
- Reduced the majority class size to match the minority class by random sampling.
- Benefits: Faster training, avoids majority class bias.
- Drawbacks: May lose valuable data.

### 3. Oversampling
- Increased the minority class size by duplicating samples randomly until it matches the majority class.
- Benefits: No data loss, better sensitivity for minority class.
- Drawbacks: Potential overfitting due to duplicate data.

### 4. Model Training and Evaluation
- Used **Random Forest Classifier** with controlled max depth and number of trees.
- Employed **Stratified K-Fold Cross-Validation** (10 splits) to ensure balanced train/test splits preserving class proportions.
- Evaluated model performance with:
  - Accuracy score
  - Confusion matrix visualization
  - Classification report (Precision, Recall, F1-score)

---

## Results

- The balanced dataset (both undersampled and oversampled) allows the Random Forest classifier to perform better on both classes.
- Cross-validation provides a reliable estimate of model generalization.
- Confusion matrices show improved detection of minority class samples.

---

## How to Use This Code

1. **Load the Dataset:**  
   Replace the file path in `pd.read_csv()` with your dataset location.

2. **Run Data Balancing Steps:**  
   Choose either undersampling or oversampling methods to balance classes.

3. **Train and Evaluate Model:**  
   The code automatically runs 10-fold stratified cross-validation and displays performance metrics.

4. **Analyze Results:**  
   View confusion matrices and classification reports for each fold to understand model behavior.

5. **Save Balanced Dataset (Optional):**  
   Oversampled dataset can be saved for future use.

---

## Why This Matters

Balancing classes is crucial in sensitive applications like medical diagnosis to avoid missing critical cases (false negatives). This project provides a practical approach to handling imbalanced datasets and validating classification models effectively.

---

## Dependencies

- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  

Install required packages using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
