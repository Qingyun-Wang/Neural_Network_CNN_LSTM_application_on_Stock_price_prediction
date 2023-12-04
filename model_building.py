
import tensorflow as tf
from tensorflow.keras import backend as K, callbacks
#from sklearn.metrics import accuracy_score as accuracy, f1_score, mean_absolute_error as mae
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten,Reshape
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.layers import LSTM
from tensorflow.keras.metrics import MeanAbsolutePercentageError
import os

from functions import *


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

########################################### Expanding rolling window #######################################################
def weighted_categorical_crossentropy(weights):
    weights = tf.convert_to_tensor(weights)

    def loss(y_true, y_pred):
        # Standard categorical cross entropy
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Calculate the penalty
        # We will consider the first weight as the penalty weight for misclassification of the first class
        penalty = weights[0] * y_true[:, 0] * (-tf.math.log(y_pred[:, 0] + tf.keras.backend.epsilon()))

        return ce + penalty

    return loss
weights = [.05, 1., 1.]  # higher weight for first class
loss_func = weighted_categorical_crossentropy(weights)

def get_oos_decision(task_type,model,model_build,x_train,x_val,x_test,y_train,y_val,y_test,refresh_period=21,epochs=50,batch_size=8): # model_build,
    """
    Implement expanding rolling window, get the out of sample predictions.
    """
    ## define a nested learing_rate_schedule function
    def lr_schedule_stock_iterate(epoch):
        # You can implement any custom learning rate schedule here
        initial_lr = 0.001
        # For example, reducing the learning rate by half every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            lr = new_model.optimizer.lr.numpy()
            lr /= 2.0
            new_model.optimizer.lr.assign(lr)
        return new_model.optimizer.lr.numpy()
    
    new_model=model
    x_train_new=x_train
    y_train_new=y_train
    x_oos=np.concatenate((x_val,x_test),axis=0)
    y_oos=np.concatenate((y_val,y_test),axis=0)
    refresh_num=len(y_oos)//refresh_period
    left_over=len(y_oos)%refresh_period
    oos_decision=[]
    for i in range(0,refresh_num):#refresh_num
        decision=classification_result_to_trade_decision(new_model,x_oos[i*refresh_period:(i+1)*refresh_period])
        oos_decision=np.concatenate((oos_decision,decision))
        x_train_new=np.concatenate((x_train_new,x_oos[i*refresh_period:(i+1)*refresh_period]),axis=0)
        y_train_new=np.concatenate((y_train_new,y_oos[i*refresh_period:(i+1)*refresh_period]),axis=0)
        # Get shuffled indices
        shuffled_indices = np.arange(x_train_new.shape[0])
        np.random.shuffle(shuffled_indices)
        # Use shuffled indices to reorder both arrays
        x_train_new = x_train_new[shuffled_indices]
        y_train_new = y_train_new[shuffled_indices]
        new_model = model_build(task_type)
        lr_scheduler_new = callbacks.LearningRateScheduler(lr_schedule_stock_iterate)
        new_model.fit(x_train_new,y_train_new, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler_new])#lr_scheduler_new,

    decision=classification_result_to_trade_decision(new_model,x_oos[-left_over:])
    oos_decision=np.concatenate((oos_decision,decision))
    return np.array(oos_decision)


def get_oos_decision_regression(task_type,model,model_build,x_train,x_val,x_test,y_train,y_val,y_test,refresh_period=21,epochs=50,batch_size=8):
    """
    Implement expanding rolling window, get the out of sample predictions.
    """
    ## define a nested learing_rate_schedule function
    def lr_schedule_stock_iterate(epoch):
        # You can implement any custom learning rate schedule here
        initial_lr = 0.001
        # For example, reducing the learning rate by half every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            lr = new_model.optimizer.lr.numpy()
            lr /= 2.0
            new_model.optimizer.lr.assign(lr)
        return new_model.optimizer.lr.numpy()
    
    new_model=model
    x_train_new=x_train
    y_train_new=y_train
    x_oos=np.concatenate((x_val,x_test),axis=0)
    y_oos=np.concatenate((y_val,y_test),axis=0)
    refresh_num=len(y_oos)//refresh_period
    left_over=len(y_oos)%refresh_period
    oos_decision=[]
    for i in range(0,refresh_num):#refresh_num
        prediction=new_model.predict(x_oos[i*refresh_period:(i+1)*refresh_period])
        decision=regression_result_to_trade_decision(prediction)
        oos_decision=np.concatenate((oos_decision,decision))
        x_train_new=np.concatenate((x_train_new,x_oos[i*refresh_period:(i+1)*refresh_period]),axis=0)
        y_train_new=np.concatenate((y_train_new,y_oos[i*refresh_period:(i+1)*refresh_period]),axis=0)
        # Get shuffled indices
        shuffled_indices = np.arange(x_train_new.shape[0])
        np.random.shuffle(shuffled_indices)
        # Use shuffled indices to reorder both arrays
        x_train_new = x_train_new[shuffled_indices]
        y_train_new = y_train_new[shuffled_indices]
        new_model = model_build(task_type)
        lr_scheduler_new = callbacks.LearningRateScheduler(lr_schedule_stock_iterate)
        new_model.fit(x_train_new,y_train_new, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler_new])

    prediction=new_model.predict(x_oos[-left_over:])
    decision=regression_result_to_trade_decision(prediction)
    oos_decision=np.concatenate((oos_decision,decision))
    return np.array(oos_decision)



def CNN1D_model(task_type):
    # Define the learning rate scheduler callback

    model = Sequential([

        # Layer-1 convolutional block
        Conv1D(32, 32, activation='LeakyReLU', padding='same',kernel_regularizer=l1_l2(l1=0.005, l2=0.005),input_shape=(seq_len, 83)),  # kernel_regularizer=l1_l2(l1=0.002, l2=0.002),
        BatchNormalization(),
        Dropout(0.2),
        MaxPool1D(2),

        Conv1D(64, 64, activation='LeakyReLU', padding='same',kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),  # kernel_regularizer=l1_l2(l1=0.002, l2=0.002),
        BatchNormalization(),
        Dropout(0.2),
        MaxPool1D(2),

        Conv1D(16, 32, activation='LeakyReLU', padding='same',kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),  # kernel_regularizer=l1_l2(l1=0.002, l2=0.002),
        BatchNormalization(),
        Dropout(0.2),
        MaxPool1D(2),

        
        # Layer-1 LSTM 
    #    LSTM(16, return_sequences=False, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),#kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
    #    Dropout(0.1),
    #    LSTM(24, return_sequences=True),
    #   Dropout(0.1),
    #    BatchNormalization(),
    #    LSTM(16),
    #    BatchNormalization(),
    #    Dropout(0.1),
        Flatten(),


        Dense(32, activation='LeakyReLU',kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),  #,kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='LeakyReLU',kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),
        BatchNormalization(),
        Dropout(0.2),

    ])
    
    if task_type ==1:
        model.add(Dense(3, activation='softmax'))  # 
        # Compile the model
        model.compile(optimizer='Adam',
                    loss=loss_func,
                    metrics=['accuracy'])
    
     #   history= model.fit(train_x,train_y, epochs=epochs, batch_size=8, validation_data=(val_x,val_y), callbacks=[lr_scheduler,early_stopping])
    elif task_type ==0:
        model.add(Dense(1, activation=None))
        # Compile the model
        model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=[RootMeanSquaredError()])
     #   history= model.fit(train_x,train_y, epochs=epochs, batch_size=16, validation_data=(val_x,val_y), callbacks=[lr_scheduler,early_stopping])
    else:
        raise ValueError("Value error, please choose 1 for classification and 0 for regression model.")

    return model



def CNN1D_Re_model(task_type):
    # Define the learning rate scheduler callback

    model = Sequential([

        # Layer-1 convolutional block
        Conv1D(32, 3, activation='LeakyReLU', padding='same',kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),input_shape=(seq_len, 83)),  # kernel_regularizer=l1_l2(l1=0.002, l2=0.002),
        BatchNormalization(),
        Dropout(0.1),
        MaxPool1D(2),

        Conv1D(64, 5, activation='LeakyReLU', padding='same',kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)),  # kernel_regularizer=l1_l2(l1=0.002, l2=0.002),
        BatchNormalization(),
        Dropout(0.1),
        MaxPool1D(2),

        Conv1D(32, 3, activation='LeakyReLU', padding='same',kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)),  # kernel_regularizer=l1_l2(l1=0.002, l2=0.002),
        BatchNormalization(),
        Dropout(0.1),
        MaxPool1D(2),

    
        # Layer-1 LSTM 
    #    LSTM(16, return_sequences=False, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),#kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
    #    Dropout(0.1),
    #    LSTM(24, return_sequences=True),
    #   Dropout(0.1),
    #    BatchNormalization(),
        LSTM(16),
        BatchNormalization(),
        Dropout(0.1),
        Flatten(),


        Dense(32, activation='LeakyReLU',kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)),  #,kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
        BatchNormalization(),
        Dropout(0.1),
        Dense(16, activation='LeakyReLU',kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)),
        BatchNormalization(),
        Dropout(0.1)
    ])
    
    if task_type ==1:
        model.add(Dense(3, activation='softmax'))  # 
        # Compile the model
        model.compile(optimizer='Adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
     #   history= model.fit(train_x,train_y, epochs=epochs, batch_size=8, validation_data=(val_x,val_y), callbacks=[lr_scheduler,early_stopping])
    elif task_type ==0:
        model.add(Dense(1, activation=None))
        # Compile the model
        model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=[RootMeanSquaredError()])
     #   history= model.fit(train_x,train_y, epochs=epochs, batch_size=16, validation_data=(val_x,val_y), callbacks=[lr_scheduler,early_stopping])
    else:
        raise ValueError("Value error, please choose 1 for classification and 0 for regression model.")

    return model




############################################### Energy prediction model ##########################################################


def CNN_LSTM_energy_regression(train_x):

    model = Sequential([
        
        # Layer-1 convolutional block
        Conv1D(16, 3, activation='relu', padding='same',kernel_regularizer=l1_l2(l1=0.01, l2=0.001),input_shape=(train_x.shape[1], train_x.shape[2])),  # kernel_regularizer=l1_l2(l1=0.002, l2=0.002),
        BatchNormalization(),
        Dropout(0.1),
        MaxPool1D(2),

        # Layer-2 convolutional block
        Conv1D(32, 5, activation='relu', padding='same',kernel_regularizer=l1_l2(l1=0.000, l2=0.000)),  # kernel_regularizer=l1_l2(l1=0.002, l2=0.002),
        BatchNormalization(),
        Dropout(0.1),
        MaxPool1D(2),

        # Layer-3 convolutional block
        Conv1D(8, 3, activation='relu', padding='same',kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),  # kernel_regularizer=l1_l2(l1=0.002, l2=0.002),
        BatchNormalization(),
        Dropout(0.1),
        MaxPool1D(2),
        
        # Layer-1 LSTM
        LSTM(8,return_sequences=True),
        BatchNormalization(),
        Dropout(0.1),

        # Layer-2 LSTM
        LSTM(4),
        BatchNormalization(),
        Dropout(0.1),

        # Layer-1 dense
        Dense(16, activation='relu',kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),  #,kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
        BatchNormalization(),
        Dropout(0.1),
        # Layer-2 dense
        Dense(8, activation='relu',kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Dropout(0.1),

        Dense(1, activation=None)
    ])

    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=[MeanAbsolutePercentageError()])

    return model #,history




def CNN_energy_classification(train_x):
    # Define the learning rate scheduler callback

    model = Sequential([
        
        # Layer-1 convolutional block
        Conv1D(16, 3, activation='relu', padding='same',kernel_regularizer=l1_l2(l1=0.01, l2=0.001),input_shape=(train_x.shape[1], train_x.shape[2])),  # kernel_regularizer=l1_l2(l1=0.002, l2=0.002),
        BatchNormalization(),
        Dropout(0.1),
        MaxPool1D(2),

        # Layer-2 convolutional block
        Conv1D(32, 5, activation='relu', padding='same',kernel_regularizer=l1_l2(l1=0.000, l2=0.000)),  # kernel_regularizer=l1_l2(l1=0.002, l2=0.002),
        BatchNormalization(),
        Dropout(0.1),
        MaxPool1D(2),

        # Layer-3 convolutional block
        Conv1D(8, 3, activation='relu', padding='same',kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),  # kernel_regularizer=l1_l2(l1=0.002, l2=0.002),
        BatchNormalization(),
        Dropout(0.1),
        MaxPool1D(2),
        
     #   LSTM(8,return_sequences=True),
     #   BatchNormalization(),
     #   Dropout(0.1),

      #  LSTM(4),
      #  BatchNormalization(),
      #  Dropout(0.1),

        Flatten(),
        # Layer-1 dense
        Dense(16, activation='relu',kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),  #,kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
        BatchNormalization(),
        Dropout(0.1),
        # Layer-2 dense
        Dense(8, activation='relu',kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Dropout(0.1),

        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics='accuracy')

    return model 