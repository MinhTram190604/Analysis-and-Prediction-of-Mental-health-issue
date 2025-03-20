# Analysis and Prediction of Mental Health Issue
## Main Objectives:
- Provide insights into how everyday factors correlate with mental health risks.
- Identify key contributors to mental health challenges in a non-clinical setting.
- Develop predictive models to support mental health research.  
  
## Techniques Used:
- Python programming language (pandas, numpy, seaborn, matplotlib, scipy,sklearn).
- Preprocessing data techniques (Handling missing values, Feature transformation, Oversampling - SMOTE)
- Explore data analysis (Univariate analysis, Multivariate analysis, Hypothesis testing, Data visualization).
- Machine learning techniques (XGBoost, Light GBM, Random Forest, hyperparameter tuning).

## Step 1: Preprocessing
Transform 'Sleep Duration'.
Drop 'Name', 'id', 'City', 'Degree', 'Profession', 'Profession'.
Encoding 'Working Professional or Student', 'Gender', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness'
Handling missing values in 'Financial Stress', 'Dietary Habits', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration'
Drop outliers.
Oversampling target variable by Synthetic Minority Over-Sampling technique.

## Step 2: EDA
### Univariate Analysis
#### **1️. Age**
- **Mean:** 40.39 years  
- **Standard Deviation:** 12.38 (indicating significant variation in age)  
- **Min/Max:** 18 - 60 years  
- **Percentiles:**  
  - **25%:** 29 years  
  - **50% (Median):** 42 years  
  - **75%:** 51 years  
- **Insights:**  
  - The age range is from 18 to 60, with the majority between **29 and 51 years**.  
  - No apparent outliers in the age data.  

---

#### **2️. Academic Pressure**
- **Count:** 27,897 (many missing values)  
- **Mean:** 3.14 (on a scale from 1 to 5)  
- **Standard Deviation:** 1.38  
- **Percentiles:**  
  - **25%:** 2  
  - **50%:** 3  
  - **75%:** 4  
- **Insights:**  
  - The average academic pressure is around **3.14**, indicating a moderate level of pressure.  
  - Most responses range from **2 to 4**.  
  - Missing values should be addressed.  

---

#### **3️. Work Pressure**
- **Count:** 112,782 (some missing values)  
- **Mean:** 3.00  
- **Percentiles:**  
  - **25%:** 2  
  - **50%:** 3  
  - **75%:** 4  
- **Insights:**  
  - The distribution is balanced around **3**, suggesting a moderate level of work pressure.  
  - Missing data needs to be handled appropriately.  

---

#### **4️. CGPA (Cumulative Grade Point Average)**
- **Mean:** 7.66  
- **Percentiles:**  
  - **25%:** 6.29  
  - **50%:** 7.77  
  - **75%:** 8.92  
- **Insights:**  
  - The CGPA ranges from **5.03 to 10**, with an average of **7.66**, indicating a generally satisfactory academic performance.  
  - Most students have CGPA between **6.29 and 8.92**.  

---

#### **5️. Study Satisfaction**
- **Mean:** 2.94  
- **Percentiles:**  
  - **25%:** 2  
  - **50%:** 3  
  - **75%:** 4  
- **Insights:**  
  - Average satisfaction is around **2.94**, suggesting a moderate to low level of satisfaction.  
  - Potentially linked to academic pressure or workload.  

---

#### **6️. Job Satisfaction**
- **Mean:** 2.97  
- **Insights:**  
  - The average satisfaction level is similar to **Study Satisfaction**, possibly reflecting overall dissatisfaction with work-life balance.  

---

#### **7️. Work/Study Hours**
- **Mean:** 6.25 hours/day  
- **Standard Deviation:** 3.85  
- **Percentiles:**  
  - **25%:** 3 hours  
  - **50%:** 6 hours  
  - **75%:** 10 hours  
- **Insights:**  
  - High variation in work/study hours, ranging from **0 to 12 hours/day**.  
  - Individuals working/studying **10-12 hours/day** may be at a higher risk of stress.  

---

#### **8️. Financial Stress**
- **Mean:** 2.98 (on a scale of 1 to 5)  
- **Insights:**  
  - The average financial stress is moderate to high, which may influence overall mental health.  

---

#### **9️. Depression**
- **Mean:** 0.18 (binary variable, 0 = No depression, 1 = Depression)  
- **Insights:**  
  - Only **18%** of the sample is marked as experiencing depression, indicating an imbalanced dataset.  
  - This imbalance may require oversampling or specific modeling techniques to prevent bias.  

---

### Multivariate analysis
#### 1. Gender & Depression
Use chi-square statistics to examine the impact of gender on the risk of depression.
**Insight**: Women have a slightly higher risk of depression than men.

---

#### 2. Age & Depression
Use Spearman correlation coefficient to examine the impact of age on the risk of depression.  
**Insight**: Young people have a higher risk of depression.

💭 **Author's Hypothesis:**  
1️. Younger individuals have greater awareness of mental health issues, while older age groups may not recognize their psychological struggles.  
2️. The underdeveloped brain may struggle to handle stress effectively.  
3️. Potential sampling bias in data collection.  

---

#### 3. Working Professional or Student & Depression

![image](https://github.com/user-attachments/assets/76afbdbc-ad2f-49ab-9a23-3f1e48664e9a)



---

#### 4. Other numerical variables (Academic Pressure, Work Pressure, CGPA, Study Satisfaction, Job Satisfaction, Work/Study Hours, Financial Stress, Sleep Duration) & Depression
Use Spearman correlation coefficient to examine the impact of these factors on the risk of depression.  
**Insight**: The higher academic pressure, the greater risk of depression.

---

#### 4. Other variables or Student & Depression
Use Spearman correlation coefficient to examine the impact of work pressure on the risk of depression.  
**Insight**: The higher work pressure, the greater risk of depression.

