# DSAI-HW1-Electricity Forecasting

# Overview
The purpose is to predict the daily **operating reserve** of electrical power in Taiwan. A period of electricity data was given and used to predict the daily operating reserve during **2021.03.23 - 2021.03.29**.

Data Collection
---
The history data of electrical power was collected from **2019.01.01 - 2021.03.21**. The typical implementation of time series data usually take the time series decompose technique, then the data would decompose into three parts: trend, seasonal and residual. The following figure shows the result of time series decomposition.

![image](https://github.com/KevinLu43/DSAI-HW1-Electricity-Forecasting/blob/main/figure/Time%20decompose.png)


[1] gives the relative importance of the features, that gave us some inspiration. The figure below illustrated the importance of the features.

![image](https://github.com/KevinLu43/DSAI-HW1-Electricity-Forecasting/blob/main/figure/importance%20of%20features.JPG)

Therefore, we take the **"Week day"** and **"Is working day"** as features and plug in the data set which was collected.

Model Construction
---
The **LSTM** is widly used recently and had good performance, so the LSTM is considered to construct the predict model. On the other hand, considering the neural network usually has to train plently of parameters, the history data set only contain 811 observations, the **Gated Recurrent Unit (GRU)** is taken to bulid the model because of the less parameters and good performance which were compared to LSTM[2].

The total observations of collecting data is 811, the data were splited as training set and testing set by following 80/20 rule. As mentioned in previous paragraph, the feature: "Is working day" was encoded as dummy variable. The data were clustered by "Week day" at the same time.

After training the model, the RMSE of testing set is:

The final prediction:

Reference
---
[1]Dehalwar, Vasudev, et al. "Electricity load forecasting for Urban area using weather forecast information." 2016 IEEE International Conference on Power and Renewable Energy  (ICPRE). IEEE, 2016.

[2]Chung, Junyoung, et al. "Empirical evaluation of gated recurrent neural networks on sequence modeling." arXiv preprint arXiv:1412.3555, 2014.
