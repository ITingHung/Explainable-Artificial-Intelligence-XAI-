# Introduction to Interpretable Models

**Table of Contents**
- [Introduction](#Introduction)
    - [Trade off between interpretability and accuracy](#Trade-off-between-interpretability-and-accuracy)
    - [Feature importance](#Feature-importance)

- [Interpretable model](#Interpretable-model)
    - [Additive Feature Attribution Definition](#Additive-Feature-Attribution-Definition)
    - [Existing Methods](#Existing-Methods)
         1. [LIME](#1-LIME)
         2. [DeepLIFT](#2-DeepLIFT)
         3. [Shapley values](#3-Shapley-values)
         4. [SHAP values](#4-SHAP-values)

- [SHAP Implementation](#SHAP-Implementation)

- [Citations](#Citations)

## Introduction
**Model interpretability** can be quite important in many applications. Take health care industry for example, if a machine learning model shows the reason why it classifies a patient to the high risk group of heart disease, then doctors would be able to check out the reasonableness of the result from the model, which increase the user trust. Besides, interpretability also provides the insight for users to improve the model.

### Trade off between interpretability and accuracy
As mentioned above, interpretability is crutial to a machine learning model. It is always easy for a simple model(e.g., Linear regression) to interpret the relationship between inputs and outputs through global level feature importance; however, for complex models like deep network, they usually have low interpretability even though their accuracy is much better than simple models. Hence, here comes out an issue: trade off between interpretability and accuracy. For the most part, there is non-linear relationship between features, which made complex models more suitable for prediction. Therefore, some explanation models(g) are created to do the interpretation from **local level feature importance** for complex models(f). Instead of trying to interpret the whole complex model, the explanation models(g) interpret how the complex model behaved for one data point.

<p align="center">
<img src="./image/Trade_off.png" alt="Trade off between interpretability & accuracy" title="Trade off between interpretability & accuracy" width="500">
</p>

### Feature importance
Feature importance shows the contribution of each feature and interprets the result of machine learning models. There are two ways to calculate feature importances: 
1. Global level (Overall importance): calculate the influence of X(feature) in a **model**’s prediction (E.g., Gini, GR……)
2. Local level (By-instacne importance): calculate the influence of X(feature) in a **specific sample**’s prediction, which means that different sample may have different feature importance.

## Interpretable model
### Additive Feature Attribution Definition
Explanation models use simpliﬁed inputs x' that map to the original inputs through a mapping function x = h<sub>𝑥</sub>(x').

<p align="center">
  <img src="./image/Additive_definition.png" alt="Additive Feature Attribution Definition" title="Additive Feature Attribution Definition" width="500">
</p>

Summing all the effects 𝝓 in explanation model(g) approximates the output of the original model(f).

[Notation]                                 
g: Explanation model (e.g., linear regression)   
𝝓<sub>i</sub>: Contribution of feature i   
x: Sample in original representation  
x': Sample in interpretable representation     
z: Perturbed sample in original representation  
z': Perturbed sample in interpretable representation 

### Existing Methods
#### 1. [LIME](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)
LIME(Local interpretable model-agnostic explanations) is a **Model-Agnostic Approximations** which locally approximate a simple model(explanation model) to a sample by perturbing the input and see how the predictions change. According to the figure showing below, the explanation model which is built for the sample, is not suitable for the whole complex model but can perform well in local level.

<p align="center">
<img src="./image/LIME.png" alt="LIME" title="LIME" width="500">
</p>

**LIME Example: find an explanation model for an image**

Consider a frog image was classified by a complex model, the result shows that "tree frog" is the most likely class, followed by "pool table" and "balloon" with lower probabilities.

<p align="center">
<img src="./image/LIME_complex result.png" alt="LIME_complex result" title="LIME_complex result" width="500">
</p>

In order to interpret the result from the complex model, we use LIME method to build up an explanation model. The first step is to seperate the original image into several interpretable components, which can be viewed as features. Here we use "pixels" for the original image, and "super pixels" for interpretable components. The super pixel is represent in binary, if an interpretabe component exists in the sample, the value would be 1, if it doesn't the value would be 0. The next step is to pertub the super pixels by randomly turn off interpretable components(in this example, make them into gray), then we could get a number of pertubed instances.

<p align="center">
<img src="./image/LIME_preturb.png" alt="LIME_preturb" title="LIME_preturb" width="500">
</p>

In the third step, we use the complex model to classify those pertubed instances and get the prediction result. 

<p align="center">
<img src="./image/LIME_preturb in complex.png" alt="LIME_preturb in complex" title="LIME_preturb in complex" width="500">
</p>

The last step is to build an explanation model through the prediction result of the interpretable components(super pixels) in complex model.

<p align="center">
<img src="./image/LIME_super pixel.png" alt="LIME_super pixel" title="LIME_super pixel" width="500">
</p>

Example & figure reference: [Local Interpretable Model-Agnostic Explanations (LIME): An Introduction](https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime)

**LIME Objective function**

<p align="center">
<img src="./image/LIME_objective function.png" alt="LIME_objective function" title="LIME_objective function" width="500">
</p>

[Notation]  
f: Complex model (e.g., CNN)                                    
g: Explanation model (e.g., linear regression)  
Ω(g): Complexity of explanation model g  
x: Sample in original representation  
x': Sample in interpretable representation   
z: Perturbed sample in original representation
z': Perturbed sample in interpretable representation 
𝜋<sub>𝑥</sub>(z): Proximity measure between an instance z to x  
D: Distance function (e.g., cosine distance for text, L2 distance for images)

#### 2. [DeepLIFT](https://arxiv.org/abs/1704.02685)
DeepLIFT is a **Model-Speciﬁc Approximations** which is used for deep learning model. DeepLIFT can be viewd as an improved version of the gradient method. 

**Gradient Example**  
In the gradient method, the feature contribution is calculated by multiplying the input(x) with the weight: 

<p align="center">
<img src="./image/DeepLIFT_gradient feature importance.png" alt="DeepLIFT_gradient feature importance" title="DeepLIFT_gradient feature importance" width="500">
</p>

In linear regression, it is reasonable to calculate feature contributions from the gradient method; however the method is not suitable for nonlinear models. Below is an example that shows the problem the gradient encounters in a nonlinear model:  

<p align="center">
<img src="./image/DeepLIFT_gradient.png" alt="DeepLIFT_gradient" title="DeepLIFT_gradient" width="500">
</p>

Because there is bias in h<sub>2</sub> function, the contribution calculated from the gradient method is unreasonalble to the actual output. DeepLIFT is introduced to solve the problem mentioned above. 

**DeepLIFT Example**  
Instead of considering the gradient, DeepLIFT considers the slope, hence the feature importance becomes:

<p align="center">
<img src="./image/DeepLIFT_feature importance.png" alt="DeepLIFT_feature importance" title="DeepLIFT_feature importance" width="300">
</p>

Below is the same example that calculates feature contributionuses by DeepLIFT:  

<p align="center">
<img src="./image/DeepLIFT.png" alt="DeepLIFT" title="DeepLIFT" width="500">
</p>

[Notation]  
m: Multiplier (slope)  
C: Feature Importance

Deciding the baseline inputs is crutial and might require domain expertise. Take MNIST digits dataset for example, since all the images are white digit with black background, it is reasonable to choose a black image as the baseline.

<p align="center">
<img src="./image/DeepLIFT_baseline.png" alt="DeepLIFT_baseline" title="DeepLIFT_baseline" width="300">
</p>

Reference: [Interpretable Neural Networks](https://towardsdatascience.com/interpretable-neural-networks-45ac8aa91411)  
Example & figure reference: [DeepLIFT Part 3: Nuts & Bolts (1)](https://www.youtube.com/watch?v=f_iAM0NPwnM&list=PLJLjQOkqSRTP3cLB2cOOi_bQFw6KPGKML&index=3)

#### 3. Shapley values
Shapley value is a solution concept in cooperative game theory, which is used to divide the reward for each player according to their contributions. In machine learning model, shapley value can be viewed as average marginal contribution to calculate the importance of a feature by comparing what a model predicts with and without the feature. The order in which a model sees features can affect its predictions, hence every possible order should be considered when calculate shapley values.

<p align="center">
<img src="./image/Shapley_value.png" alt="Shapley_value" title="Shapley_value" width="500">
</p>

[Notation]  
F: the set of all features; S⊆F  
f<sub>S∪{i}</sub> : model trained with feature i  
f<sub>S</sub>: model trained without feature i  

**Properties of Shapley value**
1. Local accuracy: the explanation model g(x') should have the same result as the original complex model f(x)

2. Missingness: when a feature is missing, the importance of this feature 𝝓 should be zero (meaning no impact to the model)

3. Consistency: if a feature i has higher impact in model A than model B, then the importance 𝝓<sub>i</sub> in model A should always be larger than the one in model B

Young (1985) had proved that Shapley values are the only set of values that satisfy three axioms similar to properties mentioned above and a ﬁnal property that is redundant in this setting.

Reference: [Interpreting complex models with SHAP values](https://medium.com/@gabrieltseng/interpreting-complex-models-with-shap-values-1c187db6ec83)

#### 4. [SHAP values](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
Since every possible orders should be considered in shapley values, when there is lots of features, it will be computationally unfeasible for calculating shapley values. Hence, SHAP(SHapley Additive exPlanations) values are introduced.

1. Kernel SHAP (Linear LIME + Shapley values): Kernal SHAP is a **Model-Agnostic Approximations** which makes Linear LIME recover Shapley values by adjusting loss function L, weighting kernel π<sub>x'</sub> and regularization term Ω.

<p align="center">
<img src="./image/Kernal SHAP_objective function.png" alt="Kernal SHAP_objective function" title="Kernal SHAP_objective function" width="500">
</p>

2. Deep SHAP (DeepLIFT + Shapley values): Deep SHAP is a **Model-Specific Approximations** which is adapted from DeepLIFT to approximate Shapley values for deep learning models. Deep SHAP recursively passing multipliers backwards through the network to combine SHAP values of smaller components into SHAP values for the whole network.

<p align="center">
<img src="./image/DeepSHAP.png" alt="DeepSHAP" title="DeepSHAP" width="500">
</p>

[Notation]  
m: Multiplier (slope)  
𝝓<sub>i</sub>: Contribution of feature i

## SHAP Implementation
Here is the implementation about an application of SHAP values.

[Dataset Info.]
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


## Citations
LIME: [“Why Should I Trust You?” Explaining the Predictions of Any Classifier
](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)  
DeepLIFT: [Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685)  
SHAP Values: [A Unified Approach to Interpreting Model Predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)  
TreeExplainer: [Explainable AI for Trees: From Local Explanations to Global Understanding](https://arxiv.org/abs/1905.04610)

