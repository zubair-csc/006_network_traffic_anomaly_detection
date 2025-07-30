# 🛡️ Network Traffic Anomaly Detection System
Python License

## 📋 Project Overview
This project implements a network traffic anomaly detection pipeline using Isolation Forest, DBSCAN, and statistical methods (IQR and Z-Score) on the NSL-KDD dataset. Built in Python for Google Colab, it includes data preprocessing, anomaly detection, model evaluation, and visualizations to identify potential intrusions in network traffic.

## 🎯 Objectives
- Detect anomalies in network traffic indicating potential intrusions
- Implement and compare Isolation Forest, DBSCAN, and statistical methods
- Evaluate model performance using precision, recall, F1-score, and accuracy
- Visualize anomalies using PCA scatter plots and feature distributions
- Provide a synthetic data fallback for testing without the dataset

## 📊 Dataset Information
**Dataset**: NSL-KDD Network Intrusion Detection Dataset (Kaggle)
- **Size**: ~125,973 training samples, ~22,544 test samples
- **Files**:
  - `KDDTrain+.txt`: Training data (43 features including duration, protocol_type, label)
  - `KDDTest+.txt`: Test data (same structure)
- **Target Output**: Binary classification (normal vs. anomaly)
- **Techniques**: Isolation Forest, DBSCAN, IQR, Z-Score, Ensemble

## 🔧 Technical Implementation
### 📌 Analysis Techniques
- **Isolation Forest**: Detects anomalies by isolating data points
- **DBSCAN**: Identifies anomalies as noise points in clustering
- **Statistical Methods**: Uses IQR and Z-Score for outlier detection
- **Ensemble**: Combines predictions via majority voting

### 🧹 Data Preprocessing
- **NSL-KDD Data**:
  - Encode categorical features (protocol_type, service, flag)
  - Standardize numerical features
  - Handle missing values with median imputation
  - Convert labels to binary (normal=0, anomaly=1)
- **Synthetic Data** (fallback):
  - Generate 10,000 samples with 20 features
  - 90% normal, 10% anomalous data

### ⚙️ Modeling
- **Isolation Forest**: Trained with contamination set to anomaly ratio
- **DBSCAN**: Uses PCA (95% variance) and eps=0.5, min_samples=5
- **Statistical Methods**: IQR and Z-Score thresholds for outliers
- **Ensemble**: Majority vote from all methods

### 📏 Evaluation Metrics
- **Anomaly Detection**:
  - Precision, Recall, F1-Score, Accuracy
  - Confusion Matrix
- **Visualizations**:
  - PCA scatter plots for ground truth and predictions
  - Histograms of top 10 features by variance

### 📊 Visualizations
- PCA scatter plots (Annual_Income vs. Spending_Score, colored by cluster)
- Feature distribution histograms with mean, median, and IQR bounds
- Anomaly score plot for Isolation Forest

## 🚀 Getting Started
### Prerequisites
- Python 3.8+
- Google Colab or Jupyter Notebook

### Installation
Clone the repository:
```bash
git clone https://github.com/zubair-csc/006_network_traffic_anomaly_detection.git
cd 006_network_traffic_anomaly_detection
```
Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Dataset Setup
- Download the NSL-KDD dataset from [Kaggle: NSL-KDD Dataset](https://www.kaggle.com/datasets/hassan06/nslkdd)
- Upload `KDDTrain+.txt` and `KDDTest+.txt` to Google Colab or place them in the project directory
- Update file paths in the script if necessary

### Running the Script
- Open `006_network_traffic_anomaly_detection.ipynb` in Google Colab or a Python environment
- Run the notebook to perform anomaly detection and generate visualizations
- View outputs: model metrics, visualizations, and anomaly predictions
- If dataset files are missing, the script uses synthetic data automatically

## 📈 Results
- **Isolation Forest**: Detects anomalies with high precision and recall
- **DBSCAN**: Identifies noise points as anomalies
- **Statistical Methods**: Detects outliers in multiple features
- **Ensemble**: Improves overall performance via majority voting
- **Visualizations**: Clear scatter plots and histograms for analysis

## 🙌 Acknowledgments
- Kaggle for providing the NSL-KDD dataset
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
