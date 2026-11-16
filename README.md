# Early Warning System for Student Academic Performance

## Student Information
**Name:** MD SK MUSTAKIM  
**Project Type:** Individual  
**Domain:** Machine Learning / Smart Campus and Education

## Project Description
This project uses Machine Learning to identify students who may be at risk of poor academic performance.

The system analyzes student-related information and predicts whether a student is academically at risk.

## Dataset
- Source: UCI Machine Learning Repository
- Dataset: Student Performance
- Total Records: 649
- Target: Academic Risk
- At Risk (1): Final grade below 10
- Not At Risk (0): Final grade 10 or above

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- UCI ML Repository
- Joblib

## Machine Learning Algorithms
1. Logistic Regression
2. Decision Tree
3. Random Forest

## Model Performance

| Algorithm | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 84.62% | 50.00% | 15.00% | 23.08% |
| Decision Tree | 81.54% | 35.71% | 25.00% | 29.41% |
| Random Forest | 85.38% | 57.14% | 20.00% | 29.63% |

Random Forest achieved the highest accuracy and F1-score among the three models. However, its recall for at-risk students was only 20%, indicating that further improvement is necessary.

## Project Workflow
1. Load the student dataset.
2. Clean and prepare the data.
3. Create academic risk labels.
4. Visualize the dataset.
5. Split data into training and testing sets.
6. Train three Machine Learning models.
7. Evaluate model performance.
8. Generate a confusion matrix.
9. Predict academic risk for a sample student.
10. Save the trained model.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

An internet connection is required to download the dataset when running the program.

## Project Output
- `outputs/student_risk_model.pkl` — Saved Random Forest model
- `outputs/confusion_matrix.png` — Confusion Matrix

## Conclusion
This project demonstrates how Machine Learning can support early identification of students who may need academic assistance. Future improvements should focus on increasing recall and reducing missed at-risk students.