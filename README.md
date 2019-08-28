# Interpretable model introduction
The following are personal study notes based on

## Introduction
**Model interpretability** can be quite important in many applications. Take health care industry for example, if a machine leanring model shows the reason why it classifies a patient to the high risk group of heart disease, then doctors would be able to check out the reasonableness of the result from the model, which increase the user trust. Besides, interpretability also provides the insight for users to improve the model.

### Feature importance
Feature importance shows the contribution of each feature and interprets the result of machine learning models. There are two ways to calculate feature importance: 
1. Global level (Overall importance): calculate the influence of X(feature) in a **model**’s prediction (E.g., Gini, GR……)
2. Local level (By-instacne importance): calculate the influence of X(feature) in a **specific sample**’s prediction, which means that different sample may have different feature importance.

### Trade off between interpretability & accuracy
As mentioned above, interpretability is crutial to a machine learning model. It is always easy for a simple model(e.g., Linear regression) to interpret the relationship between inputs and outputs through global level feature importance; however, for complex models like deep network, they usually have low interpretability even though their accuracy is much better than simple models. Hence, here comes out an issue: trade off between interpretability and accuracy. For the most part, there is non-linear relationship between features, which made complex models more suitable for prediction. Therefore, some explanation models(g) are created to do the interpretation from **local level feature importance** for complex models(f). Instead of trying to interpret the whole complex model, the explanation models(g) interpret how the complex model behaved for one data point.
<p align="center">
  <img src="./Trade_off.png" alt="Trade off between interpretability & accuracy" title="Trade off between interpretability & accuracy" width="500">
</p>

## Interpretable model
### Additive Feature Attribution Definition:
<p align="center">
  <img src="./Additive_definition.png" alt="Additive Feature Attribution Definition" title="Additive Feature Attribution Definition" width="700">
</p>
$\Phi_i$ is the effect which the feature i attributes. Summing all the effects in explanation model(g) approximates the output of the original model(f).

### Existing Methods:
1. [LIME(Local interpretable model-agnostic explanations)](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf): perturbing the input of a sample and see how the predictions change, then locally approximate a simple model(explanation model) to the sample. According to the figure showing below, the explanation model which is built for the sample, may not be suitable for the whole complex model but do perform well in local level.
![LIME](https://github.com/ITingHung/Interpretable-model-introduction/blob/master/LIME.png "LIME")
LIME Example:

Reference: [Local Interpretable Model-Agnostic Explanations (LIME): An Introduction](https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime)

2. [DeepLIFT](https://arxiv.org/abs/1704.02685):

3. Shapley value:

4. [SHAP(SHapley Additive exPlanations)](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
