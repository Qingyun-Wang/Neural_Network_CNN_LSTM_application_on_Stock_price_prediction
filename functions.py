import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report


import glob
import os
import io
import sys
from matplotlib.backends.backend_pdf import PdfPages



############################################## parameters  ##########################################################

horizon=5                     # predict horizon
seq_len=500                   # lenth of the time sequence, or size of the 'image' in CNN 
testset_percentage=0.238      # 2016--2017 data as test
val_percentage=0.2            # 20% of the non-test data as validation
buy_threshold=0.008347        #Threshold to sell, caculated by np.quantile(y_train_num,.6),the 60% quantile of training data
sell_threshold= -0.003818     #Threshold to sell, caculated by np.quantile(y_train_num,.3) the 30% quantile of training data

upper_threshold=100           # obtained by taking the value at 80th quantile of training data
lower_threshold=60            # obtained by taking the value at 30th quantile of training data

out_path='New_result.pdf'
cwd=os.getcwd()



############################################## functions and tools ###################################################

def generate_to_pdf(filename):
    """
    Generate PDF object and save plot to it
    """
    # Create a PDF object
    pdf_pages = PdfPages(filename)
    # Example usage:
    save_prints("Sharpe ratio, maximum drawdown is: ", sharpe, drawdown)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    save_plot(fig)
    save_prints("Second print output after a plot.")
    # Close the PDF object to finalize the file
    pdf_pages.close()


def clean_df(df):
    '''
    Clean up the stock dataset
    '''
    df_cleaned=del_columns(df)
    df_cleaned=fill_na(df_cleaned)
    # since we are using next horizon day return for training, the target variable for the last horizon day will be unavailable, thus return df_cleaned[:-(horizon+1)]
    return df_cleaned


def del_columns(df):
    ''''
    Drop the unwanted columns
    '''
    # we drop 'EMA_200' and 'EMA_50' since it has 200 value, but we only have ~2000 value, if we fill with 0, that will introduce too much noise and if we drop the row, we
    # will lose too much data, so we choose to drop 'EMA_200' column directly.
    try:
        df=df.drop('EMA_200',axis=1)
        df=df.drop('EMA_50',axis=1)
    except:
        pass
    return df


def fill_na(df,rolling_window=3):
    """
    When filling these missing value, we need to be careful such that we don't use the data from future, which can 
    cause data leakage. As a result, I will use rolling average of previous data to fill the missing value if possible. 
    However, after doing this we still have many missing value. This can be understand easily from the heat_map that there 
    are chunk of missing values with lenth larger than rolling window or there is no data available before the missing value. 
    In the case that there is no data available in the beginning, we fill with 0
    """

    # Define the window size
    rolling_window = rolling_window
    origin_na=(df.isnull().sum()).sum()

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Compute rolling mean without including future data points
        rolling_means = df[column].rolling(window=rolling_window, min_periods=1).mean()
        # Use this rolling mean to fill NaN values in the current column
        df[column].fillna(rolling_means, inplace=True)
    
    new_na=(df.isnull().sum()).sum()

    if origin_na>new_na and new_na!=0:
        df=fill_na(df,rolling_window=rolling_window+1)
    elif origin_na==new_na and new_na!=0:
        df=df.fillna(0)
    return df


def add_new_feature_stock(df_stock):
    """
    Adding new features like daily return, moving 3/7 day daily return to the dataset
    """
    # Calculate daily_return
    df_stock['daily_return'] = (df_stock['Close'] - df_stock['Close'].shift(1)) / df_stock['Close'].shift(1)
    # Calculate avg_3day_daily_return (3-day moving average of daily_return)
    df_stock['avg_3day_daily_return'] = df_stock['daily_return'].rolling(window=3).mean()
    # Calculate avg_7day_daily_return (7-day moving average of daily_return)
    df_stock['avg_7day_daily_return'] = df_stock['daily_return'].rolling(window=7).mean()
    df_stock=df_stock.dropna()
    df_stock=df_stock.reset_index(drop=True)

    return df_stock



def create_sequence_feature_stock(df):
    """
    Make the 2D data into 3D, every seq_len days are grouped and treated as an image in CNN.
    Out put the 3D dataset. Notice the change of the length of the dataset due to sequence and horizon
    """
    # Initialize a list to store each window
    windows_list_3d = []
    # Loop through the dataframe and extract length-row windows
    for i in range(len(df) - seq_len + 1):
        window = df.iloc[i:i+seq_len].values
        windows_list_3d.append(window)

    # Convert the list of windows to a numpy array
    windows_array_3d = np.array(windows_list_3d)

    return windows_array_3d[:-1-horizon]



def create_sequence_target_stock(df):
    """
    Prepare the target lable for the 3D training data, which is the correct classification for each image created above.
    """
    daily_return=df['daily_return']
    windows_list = []
    # Loop through the dataframe and extract length-row windows
    for i in range(seq_len+1,len(df)-horizon+1):
        future_3day_return=0
        for j in range(horizon):
            future_3day_return += daily_return[i+j]
        windows_list.append(future_3day_return)
    # Convert the list of windows to a numpy array
    windows_array = np.array(windows_list)

    return windows_array




# define sharpe ratio and maximum drawdown to check the performance of model
def trading_performance(y_pred,returns):
    """
    Input:
        y_pred: array, the predicted trading decision
        returns: array, the corresponding daily return
    output:
        sharpe ratio, max_drawdown using the predicted trading decision
    """
    profit=pd.Series(y_pred * returns)
    sharpe=np.sqrt(252)*profit.mean()/profit.std()
    cumulative_profit=profit.cumsum()
    # Calculate running maximum
    running_max = cumulative_profit.cummax()
    # Calculate drawdown
    drawdown = running_max - cumulative_profit
    # Calculate maximum drawdown
    max_drawdown = drawdown.max()
    return sharpe,max_drawdown


def regression_result_to_trade_decision(prediction):
    """
    This function transfer the regression result to trade decision, such that we can know when to trade if we choose to using regression
    instead of classification
    """
    class_lable=np.array([])
    for i in prediction:
        if i >= buy_threshold:
            class_lable=np.append(class_lable,[1])
        elif i <= sell_threshold:
            class_lable=np.append(class_lable,[-1])
        else:
            class_lable=np.append(class_lable,[0])
    return class_lable


def classification_result_to_trade_decision(model,x_data):
    """
    input the trained model and the out of sample data to make the prediction and make the prediction in the form or -1,0,1
    """
    y_prediction=model.predict(x_data)
    indices = np.argmax(y_prediction, axis=1)
    # Map the indices [0, 1, 2] back to classes [-1, 0, 1]
    #class_map = {-1: [1,0,0], 0: [0,1,0], 1: [0,0,1]}
    class_labels = np.array([-1, 0, 1])[indices]
    return class_labels

def dim_3_to_dim_1_class(class3):
    """
    convert the model.predicted result to one number, representing trading decision.
    """
    indices = np.argmax(class3, axis=1)
    # Map the indices [0, 1, 2] back to classes [-1, 0, 1]
    #class_map = {-1: [1,0,0], 0: [0,1,0], 1: [0,0,1]}
    class_labels = np.array([-1, 0, 1])[indices]
    return class_labels


def regression_to_trading_performance(prediction,returns):
    """
    Use the regression predicted number to trade and calculated the sharpe and max_drawdown
    """
    trade_decision=regression_result_to_trade_decision(prediction)
    sharpe,max_drawdown=trading_performance(trade_decision,returns)
    print("Sharpe ratio is: ",sharpe, "Maximum drawdown is: ", max_drawdown)
    return sharpe, max_drawdown


def check_other_index_performance_class(model,val,test,task_type):
    """
    1) go through the Dataset folder, find all 'Processed_*.csv' files
    2) Caculate the sharpe ratio of different index using our pre-trained model and return it
    Note:
    Only the data belong to the same date, that is defined as out of sample when we are training
    using S&P dataset, are used for caculation
    """
    files = glob.glob(cwd+'/Dataset/'+'Processed_*.csv')
    file_suffixes = []
    result=pd.DataFrame(columns=['sharpe', 'max_drawdown'])
    for file in files:
        # Split the filename at underscores and take the second part.
        # Then, split again at the period to remove the '.csv'.
        suffix = file.split('_')[-1].split('.')[0]
        file_suffixes.append(suffix)

        x_test, y_test, daily_return=data_to_test(file,np.concatenate((val,test),axis=0),task_type)

        if task_type==1:
            prediction=model.predict(x_test)
          #  apply_confusion_matrix(prediction,y_test)
            decision=classification_result_to_trade_decision(model,x_test)
            sharpe,max_drawdown=trading_performance(decision,daily_return)
            result.loc[suffix]=[sharpe,max_drawdown]
        elif task_type==0:
            prediction=model.predict(x_test)
        #    fig, percentage=check_confusion_matrix_stock(prediction,y_test)
            sharpe,max_drawdown=regression_to_trading_performance(prediction,daily_return)
            result.loc[suffix]=[sharpe,max_drawdown]
    return result


def scatter_plot_true_vs_predicted(true_values, predicted_values):
    """
    scatter plot true value vs predicted value
    """
    plt.figure(figsize=(10, 7))
    # Scatter plot
    plt.scatter(true_values, predicted_values, alpha=0.6, edgecolors="w", linewidth=0.5)
    # Diagonal line for perfect predictions
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], '--', color="red")
    plt.title('True vs. Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def num_to_class_stock(arr):
    """
    convert numerical value to different class
    """
    class_lable=np.array([])
    for i in arr:
        if i > buy_threshold:
            class_lable=np.append(class_lable,[1])
        elif i < sell_threshold:
            class_lable=np.append(class_lable,[-1])
        else:
            class_lable=np.append(class_lable,[0])
    return class_lable

    
def convert_regression_to_classification_stock(model,data,true_class):
    """
    check the accuracy of classification using regression prediction
    """
    prediction=model.predict(data)
    predicted_class=num_to_class_stock(prediction.squeeze())
    true_class=num_to_class_stock(true_class)
    percentage=(predicted_class==true_class).sum()/len(true_class)
    print('The prediction accuracy for the three class classification task is: ', percentage)


def check_confusion_matrix_stock(y_prediction_num,y_true_num):

    y_pred_class=num_to_class_stock(y_prediction_num)
    true_lable=num_to_class_stock(y_true_num)
    cm = confusion_matrix(true_lable, y_pred_class)
      # Custom labels for plotting
    x_labels = ['Sell', 'Neutral', 'Buy']
    y_labels = ['Sell', 'Neutral', 'Buy']
    fig=plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=x_labels, yticklabels=y_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    percentage=(y_pred_class==true_lable).sum()/len(true_lable)
    return fig, percentage


def apply_confusion_matrix(predictions,target):
    """
    Same funtion as the one above, but used for classification model
    """
    indices_true=np.argmax(predictions, axis=1)
    class_labels=np.array([-1, 0, 1])[indices_true]
    indices_true=np.argmax(target, axis=1)
    true_lable=np.array([-1, 0, 1])[indices_true]
    # Map the indices [0, 1, 2] back to classes [-1, 0, 1]
    cm = confusion_matrix(true_lable, class_labels)
    x_labels = ['Sell', 'Neutral', 'Buy']
    y_labels = ['Sell', 'Neutral', 'Buy']
    fig=plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=x_labels, yticklabels=y_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
 #   plt.show()
  #  print(classification_report(true_lable, class_labels))
    percentage=(class_labels==true_lable).sum()/len(true_lable)
    return fig, percentage
    

def check_true_class_distribution(train,val,test):
    """
    Check class distribution of the data when doing regression model
    """
    train_class=num_to_class_stock(train)
    train_class=[(train_class==-1).sum()/len(train_class),(train_class==0).sum()/len(train_class),(train_class==1).sum()/len(train_class)]
    val_class=num_to_class_stock(val)
    val_class=[(val_class==-1).sum()/len(val_class),(val_class==0).sum()/len(val_class),(val_class==1).sum()/len(val_class)]
    test_class=num_to_class_stock(test)
    test_class=[(test_class==-1).sum()/len(test_class),(test_class==0).sum()/len(test_class),(test_class==1).sum()/len(test_class)]
    print("train_class: ", train_class, "val_class: ",val_class,"test_class: ",test_class)

"""
def check_true_class_distribution_class(train,val,test):
    indices = np.argmax(train, axis=1)
    # Map the indices [0, 1, 2] back to classes [-1, 0, 1]
    #class_map = {-1: [1,0,0], 0: [0,1,0], 1: [0,0,1]}
    class_labels = np.array([-1, 0, 1])[indices]
    print((class_labels==-1).sum()/len(class_labels), (class_labels==0).sum()/len(class_labels),(class_labels==1).sum()/len(class_labels))
    indices = np.argmax(val, axis=1)
    # Map the indices [0, 1, 2] back to classes [-1, 0, 1]
    #class_map = {-1: [1,0,0], 0: [0,1,0], 1: [0,0,1]}
    class_labels = np.array([-1, 0, 1])[indices]
    print((class_labels==-1).sum()/len(class_labels), (class_labels==0).sum()/len(class_labels),(class_labels==1).sum()/len(class_labels))

    indices = np.argmax(test, axis=1)
    # Map the indices [0, 1, 2] back to classes [-1, 0, 1]
    #class_map = {-1: [1,0,0], 0: [0,1,0], 1: [0,0,1]}
    class_labels = np.array([-1, 0, 1])[indices]
    print((class_labels==-1).sum()/len(class_labels), (class_labels==0).sum()/len(class_labels),(class_labels==1).sum()/len(class_labels))
"""

def check_true_class_distribution_class(train,val,test):
    """
    Check class distribution of the data when doing classification model
    """
    indices = np.argmax(train, axis=1)
    # Map the indices [0, 1, 2] back to classes [-1, 0, 1]
    #class_map = {-1: [1,0,0], 0: [0,1,0], 1: [0,0,1]}
    class_labels = np.array([-1, 0, 1])[indices]
    class_labels_train=((class_labels==-1).sum()/len(class_labels), (class_labels==0).sum()/len(class_labels),(class_labels==1).sum()/len(class_labels))
    indices = np.argmax(val, axis=1)
    # Map the indices [0, 1, 2] back to classes [-1, 0, 1]
    #class_map = {-1: [1,0,0], 0: [0,1,0], 1: [0,0,1]}
    class_labels = np.array([-1, 0, 1])[indices]
   #print((class_labels==-1).sum()/len(class_labels), (class_labels==0).sum()/len(class_labels),(class_labels==1).sum()/len(class_labels))
    class_labels_val=((class_labels==-1).sum()/len(class_labels), (class_labels==0).sum()/len(class_labels),(class_labels==1).sum()/len(class_labels))
    indices = np.argmax(test, axis=1)
    # Map the indices [0, 1, 2] back to classes [-1, 0, 1]
    #class_map = {-1: [1,0,0], 0: [0,1,0], 1: [0,0,1]}
    class_labels = np.array([-1, 0, 1])[indices]
  #  print((class_labels==-1).sum()/len(class_labels), (class_labels==0).sum()/len(class_labels),(class_labels==1).sum()/len(class_labels))
    class_labels_test=((class_labels==-1).sum()/len(class_labels), (class_labels==0).sum()/len(class_labels),(class_labels==1).sum()/len(class_labels))
    return [class_labels_train,class_labels_val,class_labels_test]

def predicted_class_distribution(model, x_train,x_val,x_test):
    y_train_pred=model.predict(x_train)
    y_val_pred=model.predict(x_val)
    y_test_pred=model.predict(x_test)
    pred_distribution=check_true_class_distribution_class(y_train_pred,y_val_pred,y_test_pred)
    class_distribute_df=pd.DataFrame(
        pred_distribution,
        columns=['Sell','Neutral', 'Buy'],
        index=['training','validation','testing']
        )
    return class_distribute_df

def true_class_distribution(y_train,y_val,y_test):
    pred_distribution=check_true_class_distribution_class(y_train,y_val,y_test)
    class_distribute_df=pd.DataFrame(
        pred_distribution,
        columns=['Sell','Neutral', 'Buy'],
        index=['training','validation','testing']
        )
    return class_distribute_df

def plot_distribution_table(df,titles="Predicted class distribution"):
    fig3, ax = plt.subplots(figsize=(12, 2))  # Set the size of your figure, customize as needed
    ax.axis('off')  # Turn off the axis
    ax.table(cellText=df.values, 
            colLabels=df.columns, 
            rowLabels=df.index,
            cellLoc = 'center', 
            loc='center')
    ax.set_title(titles)  
    return fig3

def generate_report_csv(model,x_oos_stock,y_oos_stock,daily_return_val,daily_return_test):
            trade=pd.DataFrame(
                columns=['Predicted_class','True_class', 'daily_return'],
                index=['training','validation','testing']
            )
            trade=pd.DataFrame(
                    columns=['Predicted class','True class','out_of_sample_daily'])
            indices_true=np.argmax(y_oos_stock, axis=1)
            class_labels=np.array([-1, 0, 1])[indices_true]
            decision=classification_result_to_trade_decision(model,x_oos_stock)
            out_of_sample_daily=np.concatenate((daily_return_val,daily_return_test),axis=0)
            trade['Predicted class']=decision
            trade['True class']=class_labels
            trade['out_of_sample_daily']=out_of_sample_daily
            return trade


def produce_class_target(returns):
    """
    Input future return pd_series, output the target series, for classfication test
    """
    # Apply conditions to the series values
    return_class = []
    for i in returns:
        if i>buy_threshold:
            return_class.append(1)
        elif i< sell_threshold:
            return_class.append(-1)
        else:
            return_class.append(0)
    return np.array(return_class)


def target_onehot_encode(df):
    # Reshape to (-1, 1) because it expects 2D array
    df = df.reshape(-1, 1)
    # Initialize the encoder
    encoder = OneHotEncoder(sparse_output=False)  # Setting sparse to False to get a dense matrix
    # Fit and transform the data
    df_encoded = encoder.fit_transform(df)
    return df_encoded



def save_prints(*args, **kwargs):
    """Capture print outputs and save to PDF as text."""
    # Redirect standard output to capture print statements
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    # Print the given arguments
    print(*args, **kwargs)
    # Reset standard output to its original value
    sys.stdout = old_stdout
    # Capture the print outputs and plot them as text in a new figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    ax.text(0.05, 0.95, new_stdout.getvalue(), ha='left', va='top', size=12)
    # Save this figure to the PDF
    return fig
   # pdf_pages.savefig(fig)
   # plt.close(fig)


def save_plot(pdf_pages,fig):
    """Save the given plot to the PDF."""
    pdf_pages.savefig(fig)
    plt.close(fig)


def Data_split(df,target):
    """
    split the data into train, val, test
    """
    # Split data into temp (which will be further split into train and validation) and test
    x_temp, x_test, y_temp, y_test = train_test_split(df, target, test_size=testset_percentage, shuffle=False)  
    # Split temp data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=val_percentage, shuffle=False)  # 70% trainning, 30% validation
    return x_train, x_val, x_test, y_train, y_val, y_test


def normalization_feature(*dataset):
    """
    Normalize each image, which is (window_size,num_feature) using z-score normalization
    """
    dataset=list(dataset)
    sc = StandardScaler()
    # update the cols with their normalized values
    for data in dataset:
        for i in range(len(data)):
            data[i]= sc.fit_transform(data[i])
    return dataset



def plot_metric_epoch(his, metric, start_epoch=0):
    """
    Make a metric vs epoch graph and return the figure.
    """
    epochs = range(start_epoch, len(his.history['loss']))
    fig = plt.figure()
    plt.plot(epochs, his.history[metric][start_epoch:], label=metric)
    plt.plot(epochs, his.history['val_' + metric][start_epoch:], label='val_' + metric)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    return fig


def plot_predict_VS_true(prediction,true):
    """
    plot the regression result with real value
    """
    df=pd.DataFrame({'prediction':prediction,'true':true})
    fig=plt.figure(figsize=(10,6))
    plt.plot(df['true'], label='True',  linestyle='--', linewidth=2, alpha=0.7)
    plt.plot(df['prediction'], label='Prediction',linestyle='-', linewidth=.6)
    plt.ylim(-0.05,0.05)
    plt.legend()
    plt.title("True vs. Prediction")
  #  plt.show()
    return fig

def plot_predict_VS_true_energy(prediction,true):
    """
    plot the regression result with real value
    """
    df=pd.DataFrame({'prediction':prediction,'true':true})
    fig=plt.figure(figsize=(10,6))
    plt.plot(df['true'], label='True',  linestyle='--', linewidth=2, alpha=0.7)
    plt.plot(df['prediction'], label='Prediction',linestyle='-', linewidth=.6)
 #   plt.ylim(-0.05,0.05)
    plt.legend()
    plt.title("True vs. Prediction")
  #  plt.show()
    return fig

def prepare_data_stock(path,task_type):
    """
    input file path, out put data ready for training
    """
    snp=pd.read_csv(path)
    snp=snp.select_dtypes(include=['number'])
    snp_cleaned=clean_df(snp)
    snp_new=add_new_feature_stock(snp_cleaned)

    target_sequence=create_sequence_target_stock(snp_new)
    snp_sequence=create_sequence_feature_stock(snp_new)

    x_train, x_val, x_test, y_train_num, y_val_num, y_test_num= Data_split(snp_sequence,target_sequence)

    daily_return=snp_new['daily_return'][seq_len+1:]
    daily_return_train=daily_return[:y_train_num.shape[0]]
    daily_return_val=daily_return[y_train_num.shape[0]:y_train_num.shape[0]+y_val_num.shape[0]]
    daily_return_test=daily_return[y_train_num.shape[0]+y_val_num.shape[0]:y_train_num.shape[0]+y_val_num.shape[0]+y_test_num.shape[0]]

    if task_type ==1:

        target_train_class=produce_class_target(y_train_num)
        target_val_class=produce_class_target(y_val_num)
        target_test_class=produce_class_target(y_test_num)

        y_train=target_onehot_encode(target_train_class)
        y_val=target_onehot_encode(target_val_class)
        y_test=target_onehot_encode(target_test_class)

    elif task_type==0:

        y_train,y_val,y_test=y_train_num, y_val_num, y_test_num

    elif task_type !=1 and task_type!=0:
        raise ValueError("Value error, please choose 1 for classification and 0 for regression model.")

    x_train, x_val, x_test = normalization_feature(x_train,x_val,x_test)

    return  x_train, x_val, x_test, y_train,y_val,y_test,daily_return_train,daily_return_val,daily_return_test


def data_to_test(path,oos_y,task_type):
    """
    input file path, out put data ready for testing. Usually is used to load out of sample from other index, besides s&p
    """
    snp=pd.read_csv(path)
    snp=snp.select_dtypes(include=['number'])
    snp_cleaned=clean_df(snp)
    snp_new=add_new_feature_stock(snp_cleaned)

    target_sequence=create_sequence_target_stock(snp_new)
    snp_sequence=create_sequence_feature_stock(snp_new)
    daily_return=snp_new['daily_return'][seq_len+1:len(snp_new)-horizon+1]

    target_sequence=target_sequence[-len(oos_y):]
    snp_sequence=snp_sequence[-len(oos_y):]
    daily_return=daily_return[-len(oos_y):]

    if task_type ==1:
        target_class=produce_class_target(target_sequence)

        y_test=target_onehot_encode(target_class)

    elif task_type==0:
        y_test=target_sequence

    elif task_type !=1 and task_type!=0:
        raise ValueError("Value error, please choose 1 for classification and 0 for regression model.")

    [x_test] = normalization_feature(snp_sequence)
 
    return  x_test, y_test, daily_return

def perfect_performance_daily():
    """
    return the perfect predicted sharpe and max_drawdown value, max_drawdown should be 0 if every thing is correct.
    """
    profit=np.abs(daily_return_oos)
    profit=pd.Series(profit)
    sharpe=np.sqrt(252)*profit.mean()/profit.std()
    cumulative_profit=profit.cumsum()
    # Calculate running maximum
    running_max = cumulative_profit.cummax()
    # Calculate drawdown
    drawdown = running_max - cumulative_profit
    # Calculate maximum drawdown
    max_drawdown = drawdown.max()
    return sharpe,max_drawdown


###################################### Energy prediction   ################################

def create_sequence_feature_energy(df):
    """
    input df should be shape (m,n), we create a seq_len walking window such that every 60 constinuesly data is one data point,
    the so called 'image' in teh view of CNN. Then the output arr have shape (m-seq_len, seq_len, n), the first dimension is m-seq_len 
    instead of m-seq_len-1, that's because we also need to take one element off since we are making prediction for next moment.
    """
    # Initialize a list to store each window
    windows_list_3d = []
    # Loop through the dataframe and extract length-row windows
    for i in range(len(df) - seq_len + 1):
        window = df.iloc[i:i+seq_len].values
        windows_list_3d.append(window)
    # Convert the list of windows to a numpy array
    windows_array_3d = np.array(windows_list_3d)
    return windows_array_3d[:-1]


def create_sequence_target_energy(df):
    """
    Create time step sequence data for CNN model to use
    """
    data=df.values
    windows_list = []
    # Loop through the dataframe and extract length-row windows
    for i in range(seq_len,len(df)):
        windows_list.append(data[i])
    # Convert the list of windows to a numpy array
    windows_array = np.array(windows_list)

    return windows_array


def add_new_feature_energy(df_stock):
    """
    Feature engineering
    """
    # Calculate daily_return
    df_stock['percent_change'] = (df_stock['Appliances'] - df_stock['Appliances'].shift(1)) / df_stock['Appliances'].shift(1)

    # Calculate avg_3day_daily_return (3-day moving average of daily_return)
    df_stock['avg_percent_change_3'] = df_stock['percent_change'].rolling(window=3).mean()

    # Calculate avg_7day_daily_return (7-day moving average of daily_return)
    df_stock['avg_daily_return_7'] = df_stock['percent_change'].rolling(window=7).mean()

    df_stock['avg_daily_return_10'] = df_stock['percent_change'].rolling(window=10).mean()

    df_stock['avg_daily_return_50'] = df_stock['percent_change'].rolling(window=50).mean()

    df_stock=df_stock.dropna()
    
    df_stock=df_stock.reset_index(drop=True)

    return df_stock


def check_classification_acc_energy(prediction,true):
    """
    Check the accuracy of our energy classification model
    """
    indices = np.argmax(prediction, axis=1)
    prediction_labels = np.array([-1, 0, 1])[indices]
    indices = np.argmax(true, axis=1)
    true_labels = np.array([-1, 0, 1])[indices]
    return (prediction_labels==true_labels).sum()/len(true_labels)

def num_to_class_energy(arr):
    """
    transfer numerical value to the class they belong
    """
    class_lable=np.array([])
    for i in arr:
        if i > upper_threshold:
            class_lable=np.append(class_lable,[1])
        elif i < lower_threshold:
            class_lable=np.append(class_lable,[-1])
        else:
            class_lable=np.append(class_lable,[0])
    return class_lable



def produce_class_target_energy(returns):
    """
    Input future return pd_series, output the target series, for classfication tast
    """
    # Apply conditions to the series values
    return_class = []
    for i in returns:
        if i>upper_threshold:
            return_class.append(1)
        elif i< lower_threshold:
            return_class.append(-1)
        else:
            return_class.append(0)
    return np.array(return_class)


def check_confusion_matrix_energy(y_prediction_num,y_true_num):
    """
    After training of our regression model, use this function to check confusion matrix
    """
    y_pred_class=num_to_class_energy(y_prediction_num)
    true_lable=num_to_class_energy(y_true_num)
    cm = confusion_matrix(true_lable, y_pred_class)
      # Custom labels for plotting
    x_labels = ['Low', 'Medium', 'High']
    y_labels = ['Low', 'Medium', 'High']
    fig=plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=x_labels, yticklabels=y_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    percentage=(y_pred_class==true_lable).sum()/len(true_lable)
    return fig, percentage


def apply_confusion_matrix_energy(predictions,target):
    """
    Same funtion as the one above, but used for classification model
    """
    indices_true=np.argmax(predictions, axis=1)
    class_labels=np.array([-1, 0, 1])[indices_true]
    indices_true=np.argmax(target, axis=1)
    true_lable=np.array([-1, 0, 1])[indices_true]
    # Map the indices [0, 1, 2] back to classes [-1, 0, 1]
    cm = confusion_matrix(true_lable, class_labels)
    x_labels = ['Low', 'Medium', 'High']
    y_labels = ['Low', 'Medium', 'High']
    fig=plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=x_labels, yticklabels=y_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
  #  plt.show()
  #  print(classification_report(true_lable, class_labels))
    percentage=(class_labels==true_lable).sum()/len(true_lable)
    return fig,percentage


def prepare_data_energy(path,task_type):
    """
    input path to the source file, output the x_train, x_val, x_test, y_train,y_val,y_test
    """
    data=pd.read_csv(path)
    data=data.select_dtypes(include=['number'])
    data=add_new_feature_energy(data)
    Appliances=data['Appliances']
   # data.drop(['Appliances'], axis=1, inplace=True)
 
    target_sequence=create_sequence_target_energy(Appliances)
    snp_sequence=create_sequence_feature_energy(data)

    x_train, x_val, x_test, y_train_num, y_val_num, y_test_num= Data_split(snp_sequence,target_sequence)

    if task_type ==1:

        target_train_class=produce_class_target_energy(y_train_num)
        target_val_class=produce_class_target_energy(y_val_num)
        target_test_class=produce_class_target_energy(y_test_num)

        y_train=target_onehot_encode(target_train_class)
        y_val=target_onehot_encode(target_val_class)
        y_test=target_onehot_encode(target_test_class)

    elif task_type==0:

        y_train,y_val,y_test=y_train_num, y_val_num, y_test_num

    elif task_type !=1 and task_type!=0:
        raise ValueError("Value error, please choose 1 for classification and 0 for regression model.")

    x_train, x_val, x_test = normalization_feature(x_train,x_val,x_test)

    return x_train, x_val, x_test, y_train,y_val,y_test
    