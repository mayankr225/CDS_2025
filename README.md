# CDS_2025

A collection of Jupyter Notebooks from the Cybersecurity Data Science course lab sessions, accompanied by a comprehensive final report outlining our analysis, findings, and methodologies.

---

## Authors

- Mayank Rawat (640192)
- Astha Gupta (640204)
- Akshita Badola (642493)


---

## **Cyber Security Project: Vulnerability Prediction in Source Code**

This repository contains lab session materials for the Cyber Security class at Hamburg University of Technology (SoSe 2025), focusing on the detection and prediction of security vulnerabilities in source code using machine learning techniques.

The project is divided into three main parts, guiding you through the complete pipeline of vulnerability prediction: dataset creation, data preprocessing, model training, and evaluation.

---

### **Project Structure**

- **CDS_Part1.ipynb**  
  Focuses on the basics of an ML pipeline, data loading, and model evaluation.

- **CDS_Part2.ipynb**  
  Covers the creation of a custom vulnerability dataset using ProjectKB.

- **CDS_Part3.ipynb**  
  Deals with choosing, training, and evaluating a machine learning model for vulnerability prediction.

---

### **Learning Objectives**

#### **Machine Learning Fundamentals**
- Utilize a basic ML pipeline with pre-trained models
- Build custom data loaders (e.g., for HDF5 binary data)
- Load and run pre-trained ML models
- Evaluate the performance of ML models using appropriate metrics
- Calculate and interpret performance metrics (accuracy, precision, recall, F1 score, PR-AUC, ROC-AUC)

#### **Vulnerability Dataset Creation**
- Identify and extract vulnerable/non-vulnerable code from software repositories
- Apply suitable preprocessing techniques for code data
- Create a dataset of security vulnerabilities (Java methods, CVEs from ProjectKB)

#### **Model Development for Vulnerability Prediction**
- Choose an appropriate ML model architecture 
- Preprocess the dataset for the chosen model
- Split datasets for cross-validation (train, test, validation)
- Develop and implement a model training pipeline
- Train and optimize the model
- Visualize learning behavior (loss graphs)
- Generate and evaluate predictions for validation sets
- Address challenges like overfitting and model optimization

---

### **Project Description Overview**

#### **Part 1: Introduction to ML Pipeline and Evaluation**
Utilize a pre-trained ML model for vulnerability detection.  
Tasks include building a data loader for HDF5 files and generating a table of random samples with their labels.

#### **Part 2: Vulnerability Dataset Creation**
- Clone the ProjectKB repository and locate CVE data
- Develop a script to extract repository URLs and fixing commit IDs from CVE statements
- Identify and extract both "fixed" and "vulnerable" method versions from commits

#### **Part 3: ML Model Training and Evaluation**
- Choose a suitable ML model architecture (CNN, BiLSTM)
- Preprocess the dataset (tokenization, padding)
- Split the dataset into training, validation, and test sets
- Implement a training pipeline (loss function monitoring, early stopping)
- Evaluate performance (accuracy, precision, recall, F1-score)
- Experiment with model parameters and analyze their impact

---

### **Materials and Resources**

- Lecture Slides (specific slides mentioned in each part: 1, 2, 3, 5, 6, 9)
- [PyTorch Documentation: Datasets and Data Loaders](https://pytorch.org/docs/stable/data.html)
- [ProjectKB GitHub Repository](https://github.com/SAP/project-kb)
- [PyDriller Documentation](https://pydriller.readthedocs.io/en/latest/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [matplotlib Documentation](https://matplotlib.org/stable/contents.html)

---

### **Setup and Usage**

To run these notebooks, you will need a Python environment with the following libraries installed:

- pandas
- numpy
- h5py
- torch
- tensorflow / keras
- scikit-learn
- matplotlib
- PyDriller
- PyYAML
- git (for cloning ProjectKB)

**Recommended Environment:**  
Google Colab or similar Jupyter environment (for computational requirements and ease of dependency management).  
If your datasets are stored on Google Drive, ensure you mount your Drive.

**Important Note:**  
Before running the notebooks, update all dataset file paths and locations to match your local or your mounted Google Drive setup.

---
