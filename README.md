# Customer-Transaction-Prediction (PRCP-1003)


Project overview
This project develops and compares binary classification models to identify customers who may make a transaction in the future. The target is `target` (`1` = transaction, `0` = no transaction). The dataset contains anonymized numeric features, so individual features cannot be interpreted in business terms.

Dataset
File: `train(1).csv`
Rows: 200,000
Columns: 202 (`ID\_code`, `target`, and `var\_0`–`var\_199`)
Input features: 200 numeric features
Missing values / duplicate rows: none reported by the notebook
Target distribution: 179,902 class-0 rows (89.95%) and 20,098 class-1 rows (10.05%)
The dataset is included in this project package. If you received the notebook separately, put `train(1).csv` in the same folder as the notebook.

Workflow
Load the data and check schema, missing values, duplicates, descriptive statistics, and target balance.
Remove `ID\_code` from model inputs and use `target` as the label.
Split the data into 80% training and 20% test sets using stratification and `random\_state=42`.
Fit `StandardScaler` on training data only and transform both partitions.
Apply SMOTE to the scaled training data only; keep the test data unchanged.
Train and compare Logistic Regression, Decision Tree, Random Forest, Histogram-Based Gradient Boosting, and a tuned Decision Tree.
Tune the Decision Tree using `GridSearchCV`, 3-fold `StratifiedKFold`, and ROC-AUC scoring on the original training partition with `class\_weight="balanced"`.
Evaluate the models with accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and ROC curves.


Results
The following values are the saved notebook results on the 40,000-row hold-out test set using the default 0.5 classification threshold.
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	0.7894	0.2908	0.7609	0.4208	0.8584
Histogram-Based Gradient Boosting	0.8278	0.2223	0.2856	0.2500	0.6765
Random Forest	0.7780	0.1880	0.3642	0.2480	0.6724
Tuned Decision Tree	0.6575	0.1552	0.5418	0.2413	0.6369
Decision Tree	0.5669	0.1225	0.5371	0.1995	0.5696
Interpretation: Logistic Regression had the highest ROC-AUC, recall, precision, and F1-score among the evaluated models. Histogram-Based Gradient Boosting had the highest accuracy, but its recall was lower. Because the target is imbalanced, accuracy should not be used alone.


Important limitations
The model comparison and candidate selection use the same hold-out test set, so the reported best-model result may be optimistic. Use validation data or nested cross-validation for a stronger final estimate.
Only the Decision Tree received cross-validated hyperparameter tuning; the other models used fixed settings.
Only the default 0.5 threshold was reported. Threshold choice should reflect business costs and capacity.
Because Logistic Regression was trained on SMOTE-balanced data, its raw probability estimates may not reflect the real-world transaction rate and should be calibrated before being interpreted as probabilities.
The feature names are anonymized, limiting business interpretation.


Project files
```text
Customer\_Transaction\_Prediction\_Final\_Verified.ipynb  # Analysis and model training
Customer\_Transaction\_Prediction\_Final\_Report.pdf      # Final written report
README.md                                              # Project guide
requirements.txt                                       # Python dependencies
train(1).csv                                           # Dataset
```

Setup
Recommended: Python 3.10 or later.
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebook
Keep the notebook and `train(1).csv` in the same directory.
Install the dependencies above.
Open `Customer\_Transaction\_Prediction\_Final\_Verified.ipynb` in Jupyter Notebook or JupyterLab.
Choose Restart Kernel and Run All Cells.
Model training can take time because the dataset is large and SMOTE increases the training sample count. Results in this README reflect the saved notebook outputs; rerun the notebook in your environment to verify reproducibility.


Recommendation
Logistic Regression is the strongest candidate among the models tested based on ROC-AUC and the reported precision/recall/F1 metrics. It is not ready for production use without further validation, threshold selection, and probability calibration.
