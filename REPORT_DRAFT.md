# Student Grade Prediction Using Linear Regression
## Mini Project Report

**Student name:** Paridhi Chugh  
**Course:** Predictive Analytics  
**Faculty name:** Dr. Lav Upadhyay  
**Institution:** Manipal University Jaipur  
**Submission date:** 15 October 2026

## Abstract
This project demonstrates a supervised machine-learning workflow to estimate a student's final mathematics grade using non-grade student, family and school-related features. The project uses the UCI Student Performance dataset and Linear Regression. The model is evaluated with Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and the coefficient of determination (R²). The model achieved a Mean Absolute Error (MAE) of 2.1564, a Root Mean Squared Error (RMSE) of 2.8618, and an R² score of 0.1602 on the test set. These results indicate that the model has limited predictive performance when using non-grade student and school-related attributes.

## 1. Introduction
Predictive analytics uses historical data to estimate an outcome. In education, analysis of study habits, attendance and school-related information can help explore patterns associated with student performance. Predictions must be interpreted carefully and should not replace teacher judgement.

## 2. Problem Statement
Build and evaluate a regression model that estimates a student's final mathematics grade (`G3`) from available non-grade attributes in the dataset.

## 3. Dataset
The dataset is the UCI Machine Learning Repository's Student Performance dataset (ID 320), collected from school reports and questionnaires at two Portuguese schools. The final grade `G3` ranges from 0 to 20. The dataset is openly available under the Creative Commons Attribution 4.0 license.

**Source:** https://archive.ics.uci.edu/dataset/320/student%2Bperformance

## 4. Methodology
1. Load the dataset using `ucimlrepo`.
2. Select `G3` as the target variable.
3. Exclude `G1` and `G2` to avoid relying on earlier period grades when estimating final grade.
4. Split the records into training (80%) and testing (20%) sets using random state 42.
5. Preprocess numeric and categorical features using a scikit-learn pipeline.
6. Train a Linear Regression model.
7. Evaluate predictions using MAE, RMSE and R².

## 5. Results

The Linear Regression model was evaluated on 130 test records using an 80/20 train-test split with random state 42. The following metrics were obtained:

| Evaluation Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 2.1564 |
| Root Mean Squared Error (RMSE) | 2.8618 |
| R² Score | 0.1602 |

The MAE of 2.1564 indicates that the predicted grades differed from the actual grades by approximately 2.16 marks on average on the 0–20 scale. The RMSE of 2.8618 gives greater weight to larger prediction errors. The R² score of 0.1602 indicates that the model explains a relatively small proportion of the variation in final grades on the test set.

These results show that predicting final grades using non-grade student and school-related attributes alone is challenging. The model's predictive performance is limited, so its results should be interpreted cautiously.

The actual-versus-predicted graph and residual-distribution graph are saved in the `outputs` folder as `actual_vs_predicted.png` and `residual_distribution.png`, respectively. Both plots should be included in the final report.

## 6. Limitations and Ethical Considerations
The dataset is relatively small and was collected from two Portuguese schools, so the results may not generalize to Indian students or other education systems. Some features are sensitive or socially contextual. The model should be used for learning and exploratory analysis, not for labelling students or making high-stakes academic decisions.

## 7. Conclusion
This project illustrates data loading, preprocessing, train/test splitting, regression modelling and model evaluation. Its usefulness depends on the actual measured error and the limits of the dataset. Future work could validate the model on a larger and more locally relevant dataset.

## References
[1] P. Cortez, “Student Performance,” UCI Machine Learning Repository, 2008. https://doi.org/10.24432/C5TG7T  
[2] scikit-learn developers, “LinearRegression.” https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html  
[3] scikit-learn developers, “Model evaluation.” https://scikit-learn.org/stable/modules/model_evaluation.html  

## Appendix: Repository
Add your GitHub repository URL here after creating the repository. Attach the genuine Turnitin similarity report at the end of the final submission PDF if required by the instructor.
