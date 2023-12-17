
In this folder:

1. main.py:                      	   Run main function to start
2. model_building.py                   script contains model related function and model architechture/hyperparameters
3. functions.py                        script contains data processing and tool functions
4. Dataset                             a folder contains all the related dataset
5. environment.yml/requirements.txt    contains the complete package information to build a virtual environment using conda
6. model_weight						   contains pre-trained model weight


You can follow these steps to create virtual environment, install necessary package and run the script.

1. Create conda environment(named Exoduspoint_fp) using environment.yml file:
	conda env create -f environment.yml
2. Activate Conda environment:
    conda activate Exoduspoint_fp
3. Run python code:
	python main.py

If above method doesn't work, you can also use pip install -r requirements.txt


Questions to think about
1) How do you think about stationarity (or lack thereof) in time series data that may result in the training data having a different regime 
to the cross validation or out of sample dataset? If what you train on is very different from the test data(hold out), you will likely to have 
a poorly performing model. What can you do to ameliorate this issue?

Stationarity is a crucial concept in time series analysis. A stationary time series is one whose statistical properties (like mean, variance, 
and autocorrelation) do not change over time.

When such properties evolve over time, it means the underlying data-generating process has changed and the patterns the model learned from may 
no longer apply. When working with times series day, the very first step EDA (exploratory data analysis) is to understand the stationarity of 
the data with visuals and statistical analysis. Training a machine learning model, - Stationarizing the Data is a common  in time series analysis. 
Making the data stationary (e.g., through differencing or using rolling statistics) can help many models perform better, especially traditional 
statistical models.
- RNNs (Recurrent Neural Networks) and LSTMs (Long Short-Term Memory networks) are designed to work with sequential data and can capture long-term 
dependencies and complex relationships in the data. Their ability to remember past information makes them suitable for non-stationary data where 
patterns and relationships might evolve over time. However, it's worth noting that while they can handle non-stationarity to some extent, it's often 
still beneficial to stabilize the data to improve model training and performance.

* Monitoring and Retraining is a crucial aspect of deploying machine learning models, especially for time series data. As the data-generating 
process might change over time, it's essential to monitor the model's performance and retrain it with fresh data when necessary.


2) What are the choices you are making for parameter tuning/feature selection/subset selection/model selection? What can we do to avoid overfitting?

I select to using deep learning approach to complete this project. I used both LSTM and CNN in my model as they both can handle the sequential data 
and capture long-term dependencies and complex relationships. Since stock market data is highly noise, I used many technics to prevent the the 
overfitting, including drop-out, R1 R2 regularization, mini-batch and batch normalization. These method are especially effective in preventing 
overfitting in the neural network, due to the large parameter space nature of deep learning. Besides, I also seperate the data set with training
validation and test dataset carefully. For feature selection, the CNN is especially good at automatica feature selection. I use the about three layer
of converlutional block with different size of filter, such that the model can focus on the influence features and extract useful information. One more
technics I used to prevent over fitting is early stop. Using the callback funtion in TensorFlow, we can stop the training when the pre-defined parameter
stop improving. In this case, I defined the sharpe ratio as the metric to moniter, the training will stop when the sharpe ratio of the validation
set is not increasing.


3) Does your model handle missing values in input features, if they exist?
Yes.
I first examine the patterns of missing data to understand the nature of the missing values: whether they are missing randomly or due 
to some systematic reason. A crucial aspect to consider is data leakage when filling in missing data. To address this, I use past data to 
forward fill my missing values, ensuring that future information doesn't "leak" into past records. For those don't have previous data, like the 10
day moving average, I simply assign 0 to them.