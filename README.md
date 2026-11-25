# Predictive Software Defect Detection Using Machine Learning

## Project Overview
This project uses machine learning algorithms to predict whether a software module is defective or non-defective based on software metrics from the NASA JM1 dataset.

The project also includes a Streamlit web application that allows users to enter software metric values and obtain predictions.

## Objectives
- Analyze software metrics using Exploratory Data Analysis (EDA).
- Identify patterns and relationships within the dataset.
- Build and compare multiple machine learning classification models.
- Predict defective and non-defective software modules.
- Develop a simple web interface using Streamlit.

## Dataset
- Dataset: NASA PROMISE JM1
- Original records: 13,204
- Features: 21 software metrics
- Target variable: `defects`
- Exact duplicate rows were removed during preprocessing.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Pickle

## Machine Learning Algorithms
Four classification algorithms were implemented:

1. Decision Tree
2. Random Forest
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes

## Project Workflow
1. Dataset collection
2. Data preprocessing
3. Exploratory Data Analysis (EDA)
4. Data visualization
5. Feature scaling
6. Model training
7. Model evaluation and comparison
8. Model saving using Pickle
9. Streamlit application development

## Model Evaluation
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Decision Tree was selected as the preferred model based on its recall and F1-score for the defective class among the four models tested.

## Streamlit Application
The application allows users to:
- Enter 21 software metric values.
- Predict whether a software module is defective or non-defective.
- View the model-estimated prediction probability.
- Reset input values.

## Project Files
- `project1.ipynb` – Data preprocessing, EDA, visualizations, model training and evaluation.
- `App.py` – Streamlit web application.
- `JM1.csv` – Dataset.
- `software_defect_model.pkl` – Saved machine learning model.
- `software_defect_scaler.pkl` – Saved feature scaler.

## How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit application
```bash
python -m streamlit run App.py
```

## Limitations
The evaluation is preliminary. Some identical feature combinations remain across the training and testing sets, which may affect the evaluation results. Further validation is required to assess how well the model generalizes to unseen software modules.

## Future Enhancements
- Improve model performance, particularly defective-class recall.
- Perform stricter validation on unseen data.
- Explore additional machine learning techniques.
- Improve the web application's interface.

## Author
Dhanush
