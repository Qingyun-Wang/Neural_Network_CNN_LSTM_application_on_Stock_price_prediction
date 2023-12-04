from model_building import *
from functions import *


def run_engine_stock(data_path,task_type,horizon,loadmodel):

    ##################################### model training control ####################################
    def lr_schedule_stock(epoch):
            #custom learning rate schedule 
            initial_lr = 0.001
            #reducing the learning rate by half every 10 epochs
            if epoch % 10 == 0 and epoch > 0:
                lr = model.optimizer.lr.numpy()
                lr /= 2.0
                model.optimizer.lr.assign(lr)
            return model.optimizer.lr.numpy()
    ## control the learning rate of our model
    lr_scheduler = callbacks.LearningRateScheduler(lr_schedule_stock)


    # Define the early stopping callback
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
     ####################################################################################################

    x_train, x_val, x_test, y_train,y_val,y_test,daily_return_train,daily_return_val,daily_return_test=prepare_data_stock(data_path,task_type)
    # Get shuffled indices
    shuffled_indices = np.arange(x_train.shape[0])
    np.random.shuffle(shuffled_indices)
    # Use shuffled indices to reorder both arrays
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    if task_type==1:    
        if loadmodel==1:
            model = CNN1D_model(task_type)
            model.load_weights('model_weight/last3232646416323216_120.h5')
        else:
            model = CNN1D_model(task_type)
            history= model.fit(x_train,y_train, epochs=120, batch_size=8, validation_data=(x_val,y_val), callbacks=[lr_scheduler,early_stopping])
        
        daily_return_oos=np.concatenate((daily_return_val,daily_return_test),axis=0)
    
        pdf_pages = PdfPages(out_path)
        if load_model==0:
            fig1=plot_metric_epoch(history,'loss')
            save_plot(pdf_pages,fig1)
            fig2=plot_metric_epoch(history,'root_mean_squared_error')
            save_plot(pdf_pages,fig2)
        y_oos_stock=np.concatenate((y_val,y_test),axis=0)
        x_oos_stock=np.concatenate((x_val,x_test),axis=0)
        prediction=model.predict(x_oos_stock)
        fig,percentage=apply_confusion_matrix(prediction,y_oos_stock)
        # Close the PDF object to finalize the file
        save_plot(pdf_pages,fig)

        report=generate_report_csv(model,x_oos_stock,y_oos_stock,daily_return_val,daily_return_test)
        report.to_csv('report.csv', index=False)

        df=predicted_class_distribution(model, x_train,x_val,x_test)
        fig3=plot_distribution_table(df) 
        save_plot(pdf_pages,fig3)

        df=true_class_distribution(y_train,y_val,y_test)
        fig3=plot_distribution_table(df,"True class distribution") 
        save_plot(pdf_pages,fig3)

        result_on_index=check_other_index_performance_class(model, y_val,y_test,task_type)
        fig4=plot_distribution_table(result_on_index,'Trading performance on different index')
        save_plot(pdf_pages,fig4)

        expanding=input('Input any key if you want to finish; input y if you want to also run Expanding rolling to calculate the out of sample Sharpe ratio, notice that this may take hours.')
        if expanding=='y':
             oos_decision=get_oos_decision(task_type, model,CNN1D_model,x_train,x_val,x_test,y_train,y_val,y_test,refresh_period=7,epochs=120,batch_size=8)
             sharpe, max_drawdown=trading_performance(oos_decision,daily_return_oos)
             print('The expanding rolling windon calculated S&P out of sample sharpe ratio and max_drawdown is: ', sharpe, max_drawdown)

        pdf_pages.close()

    elif task_type==0:
        if loadmodel==1:
            model = CNN1D_Re_model(task_type)
            model.load_weights('model_weight/ReC323645323L16D3216.h5')
        else:
            model = CNN1D_Re_model(task_type)
            history= model.fit(x_train,y_train, epochs=160, batch_size=8, validation_data=(x_val,y_val), callbacks=[lr_scheduler,early_stopping])
        
        daily_return_oos=np.concatenate((daily_return_val,daily_return_test),axis=0)
        
        pdf_pages = PdfPages(out_path)

        if load_model==0:
            fig1=plot_metric_epoch(history,'loss')
            save_plot(pdf_pages,fig1)
            fig2=plot_metric_epoch(history,'root_mean_squared_error')
            save_plot(pdf_pages,fig2)

        y_oos_stock=np.concatenate((y_val,y_test),axis=0)
        x_oos_stock=np.concatenate((x_val,x_test),axis=0)
        prediction=model.predict(x_oos_stock).squeeze()
        fig=plot_predict_VS_true(prediction,y_oos_stock)
        save_plot(pdf_pages,fig)

        result_on_index=check_other_index_performance_class(model, y_val,y_test,task_type)
        fig4=plot_distribution_table(result_on_index,'Trading performance on different index')
        save_plot(pdf_pages,fig4)

        expanding=input('Input any key if you want to finish; input y if you want to also run Expanding rolling to calculate the out of sample Sharpe ratio, notice that this may take hours.')
        if expanding =='y':
            oos_decision=get_oos_decision_regression(task_type, model,CNN1D_Re_model,x_train,x_val,x_test,y_train,y_val,y_test,refresh_period=7,epochs=160,batch_size=8)
            sharpe, max_drawdown=trading_performance(oos_decision,daily_return_oos)
            print('The expanding rolling windon calculated S&P out of sample sharpe ratio and max_drawdown is: ', sharpe, max_drawdown)

        pdf_pages.close()


        

def run_engine_energy(data_path,task_type):

    ################ model selection creteria ############################
    def lr_schedule_stock(epoch):
            # You can implement any custom learning rate schedule here
            initial_lr = 0.001
            # For example, reducing the learning rate by half every 10 epochs
            if epoch % 10 == 0 and epoch > 0:
                lr = model.optimizer.lr.numpy()
                lr /= 2.0
                model.optimizer.lr.assign(lr)
            return model.optimizer.lr.numpy()
    ## control the learning rate of our model
    lr_scheduler = callbacks.LearningRateScheduler(lr_schedule_stock)
    # Define the early stopping callback
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ###########################################################################

    x_train, x_val, x_test, y_train,y_val,y_test= prepare_data_energy(data_path,task_type)
    if task_type==1:    
        model = CNN_energy_classification(x_train)
        history= model.fit(x_train,y_train, epochs=75, batch_size=16, validation_data=(x_val,y_val),callbacks=[lr_scheduler,early_stopping])
    elif task_type==0:
        model = CNN_LSTM_energy_regression(x_train)
        history= model.fit(x_train,y_train, epochs=30, batch_size=16, validation_data=(x_val,y_val),callbacks=[lr_scheduler,early_stopping])

    if task_type==0:
        pdf_pages = PdfPages(out_path)
        fig1=plot_metric_epoch(history,'loss')
        save_plot(pdf_pages,fig1)
        fig2=plot_metric_epoch(history,'mean_absolute_percentage_error')
        save_plot(pdf_pages,fig2)
        y_oos_energy=np.concatenate((y_val,y_test),axis=0)
        x_oos_energy=np.concatenate((x_val,x_test),axis=0)
        prediction=model.predict(x_oos_energy).squeeze()
        fig=plot_predict_VS_true_energy(prediction,y_oos_energy)
        save_plot(pdf_pages,fig)
        pdf_pages.close()

    elif task_type==1:
        pdf_pages = PdfPages(out_path)
        fig1=plot_metric_epoch(history,'loss')
        save_plot(pdf_pages,fig1)
        fig2=plot_metric_epoch(history,'accuracy')
        save_plot(pdf_pages,fig2)
        y_oos_energy=np.concatenate((y_val,y_test),axis=0)
        x_oos_energy=np.concatenate((x_val,x_test),axis=0)
        prediction=model.predict(x_oos_energy)
        fig,percentage=apply_confusion_matrix_energy(prediction,y_oos_energy)
        # Close the PDF object to finalize the file
        save_plot(pdf_pages,fig)
        print('The accuracy on out of sample is: ', percentage)
        fig=save_prints('The accuracy on out of sample is: ', percentage)
        save_plot(pdf_pages,fig)
        pdf_pages.close()
        


if __name__ == "__main__":
    cwd=os.getcwd()
    data_type=int(input("Enter 1 for stock price prediction; 0 for Appliances energy prediction(better performance): "))
    if data_type==1:
        data_path=input("Enter the path to the data source: (should be 'Dataset/*.csv') if you are using the dataset come with this folder: ")
        data_path = os.path.join(cwd, data_path)
        print('The path to the data source is:', data_path)
        horizon=int(input('Enter the predict horizon(the number of days to be summed which are used as prediction target): '))
        task_type=int(input('Enter 1 if you want to do classification, 0 if you want to do regression: '))
        loadmodel=int(input('Enter 1 if you want to do load the pretrained model, 0 if you want to do re-train the model: '))
        run_engine_stock(data_path,task_type,horizon,loadmodel)
    elif data_type==0:
        data_path=input("Enter the path to the data source: (should be 'Dataset/*.csv') if you are using the dataset come with this folder: ")
        data_path = os.path.join(cwd, data_path)
        print('The path to the data source is:', data_path)
        task_type=int(input('Enter 0 if you want to do regression; 1 if you want to do classification: '))
        run_engine_energy(data_path,task_type)
    else:
        print("Please input correct number!")
        