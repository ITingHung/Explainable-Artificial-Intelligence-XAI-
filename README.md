# SHAP(SHapley Additive exPlanations) Introduction
The following are personal study notes based on "[A Unified Approach to Interpreting Model Predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)"
## Introduction
**Model interpretability** can be quite important in many applications. Take health care industry for example, if a machine leanring model shows the reason why it classifies a patient to the high risk group of heart disease, then doctors would be able to check out the reasonableness of the result from the model, which increase the user trust. Besides, interpretability also provides the insight for users to improve the model.
### Trade off between interpretability & accuracy
As mentioned above, interpretability is crutial to a machine learning model. It is always easy for a simple model(e.g., Linear regression) to interpret the relationship between inputs and outputs; however, for complex models like neural network, they usually have low interpretability even though their accuracy is much better than simple models. Hence, here comes out an issue: trade off between interpretability and accuracy. For the most part, non-linear relationship exists in datasets, which made complex models favoured by users. Therefore, some explanation models(g) are created to do the explanation for complex models(f) 

<p align="center">
  <img src="./Trade_off.png" alt="Trade of between interpretability & accuracy" title="Trade of between interpretability & accuracy" width="500">
</p>

### Explanation model
