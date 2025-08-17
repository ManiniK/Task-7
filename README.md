# Task-7 : Support Vector Machines (SVM)  

## Objective  
Implement and understand **Support Vector Machines (SVMs)** for both **linear and non-linear classification** using the **Breast Cancer dataset**.  

## Steps Performed  

1. **Data Upload & Inspection**  
   - Uploaded dataset to Colab.  
   - Checked structure, features, and class distribution.  

2. **Data Preprocessing**  
   - Dropped unnecessary columns   
   - Converted categorical target to binary
   - Normalized features using `StandardScaler`.  
   - Split dataset into training (80%) and testing (20%).  

3. **Model Training**  
   - Trained **Linear SVM** (`kernel="linear"`)  
   - Trained **Non-linear SVM** with **RBF kernel** (`kernel="rbf"`)  

4. **Model Evaluation**  
   - Measured accuracy on test set.  
   - Evaluated using **Confusion Matrix** and **Classification Report** (Precision, Recall, F1-score).  

5. **Hyperparameter Tuning**  
   - Used **GridSearchCV** to optimize `C` (regularization) and `gamma` (kernel coefficient).  
   - Chose the best parameters based on cross-validation accuracy.  

6. **Visualization**  
   - Plotted confusion matrix as heatmap.  
   - Visualized decision boundary using **2 selected features**.  

## Results  
- **Linear SVM Accuracy**: ~0.95  
- **RBF SVM Accuracy**: ~0.96 
- Confusion matrix shows very few misclassifications.  
- Decision boundary (2D) clearly separates Malignant and Benign classes.  

## Observations  
- Linear SVM already performs well, but **RBF kernel captures non-linear boundaries** and improves accuracy.  
- **Scaling features** is essential for SVMs since they are distance-based.  
- Hyperparameters (`C`, `gamma`) strongly influence performance:  
  - High `C` → stricter classification (may overfit).  
  - Low `C` → softer margin (may underfit).  
  - `gamma` controls influence of points → small values = smoother boundary, large values = complex boundary.  
- SVMs are effective for medium-sized datasets but may become computationally expensive on very large datasets.  

## Conclusion  

SVMs are powerful classifiers for both linear and non-linear problems.  
On the Breast Cancer dataset, **RBF kernel SVM with tuned parameters achieves near-perfect accuracy**. This makes SVM a robust choice for medical diagnosis tasks where precision and recall are critical.  

---
