The following are personal study notes based on "[A Unified Approach to Interpreting Model Predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)"
## Introduction
**Model interpretability** can be quite important in many applications. Take health care industry for example, if a machine leanring model shows the reason why it classifies a patient to the high risk group of heart disease, then doctors would be able to check out the reasonableness of the result from the model, which increase the user trust. Besides, interpretability also provides the insight for users to improve the model.
### Trade off between interpretability & accuracy
As mentioned above, interpretability is crutial to a machine learning model. It is always easy for a simple model(e.g., Linear regression) to interpret the relationship between inputs and outputs; however, for complex models like neural network, they usually have low interpretability even though their accuracy is much better than simple models.

![alt Trade off image](https://github.com/ITingHung/SHAP-Introduction/blob/master/Trade%20off%20between%20interpretability%20%26%20accuracy.png)

