 Mini-Project for Fundamentals of Machine Learning Course
![background](./materials/ai_wp.jpg)
This repository contains the code and data for a mini-project on facial expression recognition using machine learning algorithms.

## 📑 Project Policy
- Team: group should consist of 3-4 students.

   ## 👥 Team Members

| No. | Student Name        | Student ID |
|-----|---------------------|------------|
| 1   | Nguyễn Hoàng Lịch   | 21280069   |
| 2   | Phạm Duy Sơn        | 21280107   |
| 3   | Trần Ngọc Tuấn      | 21280058   |
| 4   | Nguyễn Minh Duy     | 21280010   |

- The submission deadline is strict: *11:59 PM* on *June 22nd, 2024*. Commits pushed after this deadline will not be considered.

## 📦 Project Structure

The repository is organized into the following directories:

- */data*: This directory contains the facial expression dataset. You'll need to download the dataset and place it here before running the notebooks. (Download link provided below)
- */notebooks*: This directory contains the Jupyter notebook ```EDA.ipynb```. This notebook guides you through exploratory data analysis (EDA) and classification tasks.

## ⚙️ Usage

This project is designed to be completed in the following steps:

1. *Fork the Project*: Click on the ```Fork``` button on the top right corner of this repository, this will create a copy of the repository in your own GitHub account. Complete the table at the top by entering your team member names.

2. *Download the Dataset*: Download the facial expression dataset from the following [link](https://mega.nz/file/foM2wDaa#GPGyspdUB2WV-fATL-ZvYj3i4FqgbVKyct413gxg3rE) and place it in the */data* directory:

3. *Complete the Tasks*: Open the ```notebooks/EDA.ipynb``` notebook in your Jupyter Notebook environment. The notebook is designed to guide you through various tasks, including:
    
    1. Prerequisite
    2. Principle Component Analysis
    3. Image Classification
    4. Evaluating Classification Performance 
    5. *Commit and Push Your Changes*: Once you've completed the tasks outlined in the notebook, commit your changes to your local repository and push them to your forked repository on GitHub.


Feel free to modify and extend the notebook to explore further aspects of the data and experiment with different algorithms. Good luck.

    Make sure to run all the code cells in the ```EDA.ipynb``` notebook and ensure they produce output before committing and pushing your changes.
## 📝 Results

| Algorithm               | Original Data Accuracy | Processed Data Accuracy |
|-------------------------|------------------------|-------------------------|
| XGBoost                 | 34.42%                 | 44.54%                  |
| SVM                     | 44.62%                 | 56.98%                  |
| Random Forest           | 48.69%                 | 47.88%                  |
| MLP                     | 26.83%                 | 46.50%                  |


## 🔍 Interesting Findings

Throughout our project, we implemented several advanced techniques to enhance the performance of our models:

1. **Feature Engineering with HOG**: We applied the Histogram of Oriented Gradients (HOG) algorithm to extract features from images. HOG effectively captures the main contours of facial expressions by analyzing the direction and magnitude of gradients within each cell, leading to improved model performance.

2. **Addressing Imbalanced Datasets with SMOTE and Tomek Links**: Given the imbalanced nature of our dataset, we adopted a combined approach using SMOTE and Tomek Links to ensure better model training:
- ```SMOTE (Synthetic Minority Over-sampling Technique)```: This technique generates new synthetic samples for the minority class, balancing the class distribution and allowing the model to learn more effectively from minority class examples.
- ```Tomek Links```: We utilized Tomek Links to identify and remove ambiguous data points near the class boundaries. 

3. **Performance Comparison:**
- Processed Data: Across all models, the dataset with processed features consistently outperformed the original data. This enhancement suggests that feature engineering techniques such as ``HOG`` (Histogram of Oriented Gradients), ``PCA`` (Principal Component Analysis), and scaling play a crucial role in improving model performance for image classification tasks.
- ```XGBoost, SVC, RandomForest```, and ``MLP`` demonstrated significant improvements in predictive accuracy, as evidenced by higher evaluation metrics (e.g., accuracy, precision, recall, F1-score) on the processed dataset compared to the original dataset.

4. **Model Suitability for Image Classification:**
- ```XGBoost``` and ```RandomForest``` particularly benefited from the processed data, leveraging feature transformations and dimensionality reduction provided by PCA to better capture intricate facial features and variations.
- ```SVC``` and ```MLP```: Both models also showed marked improvement on the processed dataset, indicating that feature preprocessing facilitated better generalization and discrimination between facial features and identities.

5. **Impact of Feature Engineering:**
- ```HOG``` and ```PCA```: The utilization of HOG for feature extraction and PCA for dimensionality reduction proved beneficial in enhancing the models' ability to discern subtle facial differences and improve classification accuracy.
- ```Scaling```: Standardizing the features through scaling likely contributed to enhanced convergence speed and overall model stability, further boosting performance across all tested algorithms.

## 🔮 Future Works


To enhance the performance of our image classification models on the ```ICML Face Data dataset```, we plan to integrate Convolutional Neural Networks (CNNs) into our framework. We will explore the following CNN architectures:

1. **Baseline CNN**
- Objective: Develop a custom ```CNN``` for feature extraction.
- Integration: Use these features with XGBoost, SVC, RandomForest, and MLP classifiers.

2. **VGG16**
- Objective: Leverage the ```VGG16``` architecture for improved feature extraction.
- Integration: Use pre-trained ```VGG16``` features to train the existing classifiers.

3. **ResNet50**
- Objective: Utilize ```ResNet50``` to capture intricate details in images.
- Integration: Feed ```ResNet50``` features into the traditional classifiers for enhanced performance.

4. **Combined Approach**
- Objective: Experiment with ensemble methods combining CNN outputs with traditional classifiers.
- Integration: Aggregate predictions from CNN-based models and traditional classifiers for superior accuracy.

5. **Implementation Plan**
- Feature Extraction: Initially use CNNs for feature extraction, followed by fine-tuning on the ```ICML Face Data```.
- Evaluation: Conduct comprehensive evaluations using ROC/AUC, accuracy, and confusion matrix metrics.

6. **Anticipated Benefits**
- Improved Accuracy: Superior feature extraction from CNNs is expected to enhance classification accuracy.
- Robustness: Better pattern recognition will lead to more robust models.
- State-of-the-Art Performance: Combining CNNs with traditional classifiers aims to achieve state-of-the-art results.
Integrating CNN models such as ```Baseline CNN```, ```VGG16```, and ```ResNet50``` with XGBoost, SVC, RandomForest, and MLP classifiers represents a promising direction to improve accuracy and robustness for image classification on the ICML Face Data dataset.



