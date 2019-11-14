## SHAP Implementation
Here is the implementation about an application of SHAP values.

**Dataset Info.**
Source: Kaggle-Datasets  
Dataset: Faulty Steel Plates  
Features(x): 27 (25 continuous; 2 category)  
Target(y): 7 classes of defects (six types + "other")  
Sample size: 1941  


```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# read dataset
df = pd.read_csv('faults.csv')

# merge one hot encoded classes into a multiclass
field = (df.iloc[:,-7:]==1).idxmax(1)
df = df.drop(df.columns.values[-7:], axis = 1)
df['Defected_type'] = field

# drop redundant column
df = df.drop('TypeOfSteel_A300', axis = 1)
df = df.rename(columns = {'TypeOfSteel_A400':'TypeOfSteel'})

# split data
y = df['Defected_type']
x = df.copy().drop(columns=['Defected_type'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)
```

Below shows the result of the prediction from a Random Forest Classifier:

```
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,random_state = 0)
rfc.fit(x_train, y_train)
y_predict = rfc.predict(x_test)
y_predict_train= rfc.predict(x_train)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test,y_predict)
plt.figure()
plot_confusion_matrix(cm)

from sklearn.metrics import precision_recall_fscore_support
test_score = precision_recall_fscore_support(y_test,y_predict,average='macro')
train_score = precision_recall_fscore_support(y_train,y_predict_train,average='macro')
print("Total testing score: {}".format(test_score))
print("Total training score: {}".format(train_score))
```

<p>
<img src="./image/Original_CM.png" alt="Original_CM" title="Original_CM" width="700">
</p>

precision | recall | f1-score 
:--------:|:------:|:--------:
0.843     | 0.796  | 0.815

Since there is confuse between the first class "Bumps" and the forth class "Other_Faults", I extract samples in these two classes to rebuild an new model for them. Below shows the result of the prediction from the new Random Forest Classifier:

<p>
<img src="./image/TwoClass_CM.png" alt="TwoClass_CM" title="TwoClass_CM" width="700">
</p>

precision | recall | f1-score 
:--------:|:------:|:--------:
0.701     | 0.703  | 0.706


To know the reason of the misclassification between these two classes, we can use SHAP values to find out the features that mislead the model in each misclassified sample.

```
import shap

# explain the model's predictions using SHAP values
explainer = shap.TreeExplainer(rfc_new)
shap_values = explainer.shap_values(x_test_new)
```

```
# extract sample in "Bumps" and "Other_Faults"
df.loc[~df.loc[:,'Defected_type'].isin(['Bumps','Other_Faults']),'Defected_group']='Ex_B.O'
df.loc[df.loc[:,'Defected_type'].isin(['Bumps','Other_Faults']),'Defected_group']='In_B.O'
y_new = df.loc[df['Defected_group']=='In_B.O']['Defected_type']
x_new = df.loc[df['Defected_group']=='In_B.O'].copy().drop(columns=['Defected_type','Defected_group'])

# split data
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_new, y_new, test_size = 0.2, random_state = 10)
y_test_new_reindex = y_test_new.reset_index(drop=True)
x_test_new_reindex = x_test_new.reset_index(drop=True)

# extract misclassified samples
false_info = []
for i in range(len(y_test_new)):
    if y_test_new_reindex[i] != y_predict_new[i]:
        false_info.append([i,y_predict_new[i]])
false_df = pd.DataFrame(false_info, columns=['id','false_type'])
```

Because Random Forest Classifier is choosen in this example, here I use [TreeExplainer](https://arxiv.org/abs/1905.04610) for calculating SHAP values. Below shows the visualizaiton of one error prediction's explanation.

```
# load JS visualization code to notebook
shap.initjs()

# SHAP values for predicting as "Bumps"
shap.force_plot(explainer.expected_value[0], shap_values[0][13,:], x_test_new_reindex.iloc[13,:])

# SHAP values for predicting as "Other_Faults"
shap.force_plot(explainer.expected_value[1], shap_values[1][13,:], x_test_new_reindex.iloc[13,:])
```

<p align="center">
<img src="./image/visualSHAP_Bumps.png" alt="visualSHAP_Bumps" title="visualSHAP_Bumps" width="1000">
</p>

<p align="center">
<img src="./image/visualSHAP_OFaults.png" alt="visualSHAP_OFaults" title="visualSHAP_OFaults" width="1000">
</p>

To find out the reason of misclassification, I extract top five features which mislead the result from all the error predicted samples, and then calculate the frequency of these features.

```
#  extract top five features that mislead the probability of classification
topfive_index = []
j=0
for i in range(len(false_info)):
    if false_info[i][1] == 'Bumps':
        topfive_index.append(np.argsort(shap_values[0][false_info[i][0],:])[-5:][::-1])
    else:
        topfive_index.append(np.argsort(shap_values[1][false_info[i][0],:])[-5:][::-1])
topfive_df = pd.DataFrame(topfive_index, columns=['1st','2nd','3rd','4th','5th'])
topfive_df.head()

false_t = pd.concat([topfive_df, false_df], axis=1)

# count the frequecy of features
counter_t = Counter()
for i in ['1st','2nd','3rd','4th','5th']:
    counter_t += Counter(false_t[i].value_counts().to_dict())
```

Below is the Top 10 features which mislead the prediction result: 

Rank | Feature 
:---:|:------------:
1    | Square_Index
2    | TypeOfSteel 
3    | Y_Maximum 
4    | Edges_Y_Index 
5    | Empty_Index 
6    | Edges_X_Index 
7    | Y_Minimum 
8    | Orientation_Index
9    | Length_of_Conveyer
10   | Minimum_of_Luminosity
