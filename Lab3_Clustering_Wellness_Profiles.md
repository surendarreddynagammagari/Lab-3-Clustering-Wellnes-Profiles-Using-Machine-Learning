
# Lab 3: Clustering Wellness Profiles Using Machine Learning

**Surendar N Reddy**  
DeVos Graduate School, Northwood University  
MGT-665-NW Solv Probs W/ Machine Learning Graduate  
Dr. Itauma Itauma  
June 22nd, 2025

---

## Abstract

This project applies clustering algorithms to a simulated health and wellness dataset to identify unique patient segments. This project employs K-Means and Hierarchical Clustering with Principal Component Analysis (PCA) to identify unique patient clusters based on features such as exercise, sleep, BMI, and stress levels. The data is better understood prior to modelling using Exploratory Data Analysis (EDA) and visualizations. Clustering results before and after PCA are compared using Silhouette Scores and Within-Cluster Sum of Squares (WCSS). Findings suggest that healthcare professionals can be assisted by clustering in creating personalized wellness interventions.

---

## Introduction

As personalized healthcare becomes increasingly prominent, an understanding of patient wellness behaviour has emerged as a driving force for developing successful wellness programs. Machine learning, and more specifically clustering and dimensionality reduction, offers mechanisms for revealing underlying patterns in patient behaviour. This document is concerned with clustering-based patient segmentation via K-Means and Hierarchical Clustering, and reduction of data complexity via PCA. The goal is to present data that can be used by healthcare organizations to tailor wellness interventions to specific patient segments.

---

## Methods

### Exploratory Data Analysis (EDA)

Data includes variables like exercise minutes daily, healthy meals daily, sleep hours, BMI, and stress level. Initial exploration with `pandas`, `seaborn`, and `matplotlib` revealed distributions, correlations, and outliers. Heatmaps and pair plots revealed moderate correlations between sleep, stress, and BMI, suggesting potential clusters in wellness behaviour.

**Figure 1.** Histogram of feature distributions  
**Figure 2.** Correlation heatmap showing relationships among wellness variables

### Data Preprocessing

Missing values were handled using mean imputation. Features were scaled using `StandardScaler`.

**Sample Code:**

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_imputed)
```

---

### Clustering Techniques

K-Means clustering was done using Scikit-Learn. Elbow method was used to find the optimal number of clusters (`k=3`). Silhouette Score and Within-Cluster Sum of Squares (WCSS) were used for model evaluation.

---

## Results

**Table 1. Silhouette Scores of Clustering Models**

| Model              | Silhouette Score |
|--------------------|------------------|
| KMeans             | 0.61             |
| Hierarchical       | 0.55             |
| KMeans (PCA)       | 0.59             |
| Hierarchical (PCA) | 0.52             |

**Figure 3.** Comparison of silhouette scores for clustering models

---

## Discussion

Three major wellness profiles were revealed through clustering analysis in the patients. There existed one cluster with proper exercise and low stress, another with high BMI and inadequate sleeping, and the last with a balanced way of life. PCA minimized interpretation complexity by projecting data into two dimensions. While silhouette scores were minimally decreased following PCA, the trade-off was improved visualizability and interpretability of clusters.

---

## Recommendations

Healthcare professionals can make use of clustering results and target specific segments of wellness. For instance, they can target the wellness of low-exercise and high-stress groups through mental well-being programs and fitness. PCA-based clustering allows for simpler communication of trends in wellness to stakeholders. Future studies can use real-world datasets with more behavioural indicators.

---

## References

- Montoya, 2016; Itauma, 2024. *Machine learning using Python.* Quarto.  
  [https://amightyo.quarto.pub/machine-learning-using-python/Chapter_8.html](https://amightyo.quarto.pub/machine-learning-using-python/Chapter_8.html)

- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). *Scikit-learn: Machine learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.

- Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment.* Computing in Science & Engineering, 9(3), 90–95.

- Waskom, M. L. (2021). *Seaborn: Statistical data visualization.* Journal of Open Source Software, 6(60), 3021.
