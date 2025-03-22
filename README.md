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
- Transform 'Sleep Duration', 'Sleep Duration' column.  
- Drop 'Name', 'id', 'City', 'Degree', 'Profession' column.  
- Encoding 'Working Professional or Student', 'Gender', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness' column.  
- Handling missing values in 'Financial Stress', 'Dietary Habits', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration' column.
- Drop outliers.  
- Oversampling target variable by Synthetic Minority Over-Sampling technique.  

## Step 2: EDA
### Univariate Analysis
![image](https://github.com/user-attachments/assets/e2da0812-fed4-4239-a8c6-3c7402ce8f9f)

#### **1️. Age**
- **Mean:** 40.39 years  
- **Standard Deviation:** 12.38 (indicating significant variation in age)  
- **Min/Max:** 18 - 60 years  
- **Percentiles:**  
  - **25%:** 29 years  
  - **50% (Median):** 42 years  
  - **75%:** 51 years  
- 💡**Insights:**  
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
- 💡**Insights:**  
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
- 💡**Insights:**  
  - The distribution is balanced around **3**, suggesting a moderate level of work pressure.  
  - Missing data needs to be handled appropriately.  

---

#### **4️. CGPA (Cumulative Grade Point Average)**
- **Mean:** 7.66  
- **Percentiles:**  
  - **25%:** 6.29  
  - **50%:** 7.77  
  - **75%:** 8.92  
- 💡**Insights:**  
  - The CGPA ranges from **5.03 to 10**, with an average of **7.66**, indicating a generally satisfactory academic performance.  
  - Most students have CGPA between **6.29 and 8.92**.  

---

#### **5️. Study Satisfaction**
- **Mean:** 2.94  
- **Percentiles:**  
  - **25%:** 2  
  - **50%:** 3  
  - **75%:** 4  
- 💡**Insights:**  
  - Average satisfaction is around **2.94**, suggesting a moderate to low level of satisfaction.  
  - Potentially linked to academic pressure or workload.  

---

#### **6️. Job Satisfaction**
- **Mean:** 2.97  
- 💡**Insights:**  
  - The average satisfaction level is similar to **Study Satisfaction**, possibly reflecting overall dissatisfaction with work-life balance.  

---

#### **7️. Work/Study Hours**
- **Mean:** 6.25 hours/day  
- **Standard Deviation:** 3.85  
- **Percentiles:**  
  - **25%:** 3 hours  
  - **50%:** 6 hours  
  - **75%:** 10 hours  
- 💡**Insights:**  
  - High variation in work/study hours, ranging from **0 to 12 hours/day**.  
  - Individuals working/studying **10-12 hours/day** may be at a higher risk of stress.  

---

#### **8️. Financial Stress**
- **Mean:** 2.98 (on a scale of 1 to 5)  
- 💡**Insights:**  
  - The average financial stress is moderate to high, which may influence overall mental health.  

---

#### **9️. Depression**
- **Mean:** 0.18 (binary variable, 0 = No depression, 1 = Depression)  
- 💡**Insights:**  
  - Only **18%** of the sample is marked as experiencing depression, indicating an imbalanced dataset.  
  - This imbalance may require oversampling or specific modeling techniques to prevent bias.  

---

### Multivariate analysis
Since some variables will be adjusted during preprocessing (e.g., filling null values with -1, mode, or median), the author chooses to conduct univariate analysis before preprocessing. 

#### 1. Gender & Depression
Use chi-square statistics to examine the impact of gender on the risk of depression.  
💡**Insight**: Women have a slightly higher risk of depression than men.

---

#### 2. Age & Depression
Use Spearman correlation coefficient to examine the impact of age on the risk of depression.  
💡**Insight**: Young people have a higher risk of depression.

💭 **Author's Hypothesis:**  
1️. Younger individuals have greater awareness of mental health issues, while older age groups may not recognize their psychological struggles.  
2️. The underdeveloped brain may struggle to handle stress effectively.  
3️. Potential sampling bias in data collection.  

---

#### 3. Working Professional or Student & Depression
There is an imbalance between two groups.
![image](https://github.com/user-attachments/assets/76afbdbc-ad2f-49ab-9a23-3f1e48664e9a)

---

#### 4. Other numerical variables (Academic Pressure, Work Pressure, CGPA, Study Satisfaction, Job Satisfaction, Work/Study Hours, Financial Stress, Sleep Duration) & Depression
Use Spearman correlation coefficient to examine the impact of these factors on the risk of depression.  
💡**Insights**: 
- The higher job satisfaction, the lower risk of depression.
- There is no significant relationship between sleep duration and depression.
- Other variables: the higher level, the greater risk of depression.
💭 **Author's Hypothesis:**     
- Individuals with well performance may be excessively concerned about their academic success, which in turn drives them to achieve high grades and be satisfied with their results. Constant pressure to maintain top grades can result in mental and emotional exhaustion.  
- Long hours of work or study can lead to chronic stress, exhaustion, and burnout, increasing vulnerability to depression.
- Individuals who dedicate excessive time to work or study often sacrifice personal time, hobbies, and social interactions, leading to loneliness and decreased well-being.

---

#### 5. Dietary Habits & Depression  
![image](https://github.com/user-attachments/assets/cab7dee3-f760-4e12-ae3e-f094be4e2344)

💡**Insights**: 
- Unhealthy Diet: Shows a relatively higher proportion of individuals with depression.
- Healthy & Moderate Diet: Appears to have a lower proportion of depression cases.

---

#### 6. Suicidal Thoughts & Depression
![image](https://github.com/user-attachments/assets/630e41fa-2a3d-4352-9188-7b7df3b100e2)

💡**Insight:**   
Not all people who have ever had suicidal thoughts are depressed but the data shows a clear association between suicidal thoughts and depression status. Individuals who have experienced suicidal thoughts are much more likely to have depression compared to those who have not. This suggests a strong relationship between suicidal ideation and depression, reinforcing the importance of mental health support for those expressing such thoughts.

---

#### 7. Family History of Mental Illness & Depression
Use chi-square statistics to examine the impact of family history on the risk of depression.  
💡**Insight:** Inviduals that have family history of mental health problem have higher risk of depression.

---
        
#### 8. Other variables (City, Degree, Profession) & Depression
Since the City variable contains a large number of unique values, meaningful insights cannot be derived from visualizations unless it is categorized into relevant groups based on specific characteristics or criteria (e.g., Tier 1/2/3 cities, capital cities, economic hubs, tourist destinations, etc.).

Similarly, for the Profession and Degree variables, grouping them into distinct categories based on key industry sectors would enhance interpretability and analysis.

---

#### 9. Correlation
![image](https://github.com/user-attachments/assets/951ced48-8e9b-48f5-ad88-dbb9f86f3bf6)  
There is an inconsistency in Correlation Result and Statistical Test. The correlation result may be misleading due to noise, as the dataset includes individuals without Work Pressure (Work Pressure values = -1).  
This leads to an incorrect conclusion: Higher Work Pressure leads to Lower depression risk, while statistical tests indicate the opposite relationship.

💡**Insights:** There is positive correlation between CGPA, Academic Pressure, and Study Satisfaction.
- Students who prioritize academic performance tend to achieve higher GPAs.
- They also report greater satisfaction with their studies.  
This aligns with the idea that high-achieving students often set clear academic goals, leading to both higher performance and a sense of fulfillment in their education.

📌 Decision to Drop Certain Variables as inputs for predictive model:  
1. Working Professional or Student:  
This variable strongly correlates with Age, CGPA, Academic Pressure, Work Pressure, Job Satisfation, Study Satisfation. 
Besides, it can be inferred from those variables, the author thinks it should be removed.  
2. Sleep Duration:  
Statistical analysis shows no significant impact on the target variable (Depression).
Removing it simplifies the model without losing predictive power.

---

## Step 3: Build model and tune hyperparameters
3 models were chosen to predict the target variable ('Depression'): XGBoost, Light GBM, Random Forest due to their strong performance in handling complex datasets and imbalanced classes.

Two evaluation metrics — Recall and Accuracy — were selected to assess the model's performance. These metrics provide complementary perspectives on the model's effectiveness.
- Accuracy is the Kaggle's evaluation criteria.
- Recall is the author's choice.
In an imbalanced dataset, where the majority of individuals do not have depression, a model that predicts "no depression" most of the time can still achieve high accuracy but fail to detect those actually at risk. This means a high-accuracy model may not be effective in identifying at-risk individuals, as it overlooks true positive cases (false negatives).    
By prioritizing both accuracy and recall, the author aims to build a model that is not only **competitive in Kaggle's ranking system** but also **practically useful for real-world mental health screening.**  

**Result:** The best model is tuned Light GBM with 96% recall on validation set and 93% accuracy on test set.
![image](https://github.com/user-attachments/assets/cef51774-5399-4f6c-8e81-2bbd3fcf2681)
