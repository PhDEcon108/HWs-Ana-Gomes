import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from numpy.random import default_rng
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import scipy
import matplotlib.pyplot as plt

##Part I
#load the data
data=np.loadtxt('data/data.txt')
nobs=data.size

train_sample=500

ic_result=np.empty((0,4))

for p in range(0,5):
    for q in range(0,5):
        for t in {'n','c'}:
            arma_model=ARIMA(data[0:500], trend=t, order=(p, 0, q))
            results=arma_model.fit()
            ic_result=np.append(ic_result, [[f'p={p}, q={q}, const={t}', results.aic, results.bic, results.resid]], axis=0)
            

aic_min=ic_result[np.argmin(ic_result[:,1]), 0]
bic_min=ic_result[np.argmin(ic_result[:,2]), 0]

print(f'The specification that minimizes the AIC is {aic_min} and the one that minimizes the BIC is {bic_min}.')


fig, axs = plt.subplots(2, 2, figsize=(25,10))

plot00=sm.graphics.tsa.plot_acf(ic_result[np.argmin(ic_result[:,1]), 3], ax=axs[0, 0], title=f'ACF - specification {aic_min}', color='black')
plot01=sm.graphics.tsa.plot_pacf(ic_result[np.argmin(ic_result[:,1]), 3], ax=axs[0, 1], title=f'PACF - specification {aic_min}', color='black')
plot10=sm.graphics.tsa.plot_acf(ic_result[np.argmin(ic_result[:,2]), 3], ax=axs[1, 0], title=f'ACF - specification {bic_min}', color='black')
plot11=sm.graphics.tsa.plot_pacf(ic_result[np.argmin(ic_result[:,2]), 3], ax=axs[1, 1], title=f'PACF - specification {bic_min}', color='black')

plt.savefig(f"ACF&PACF_residuals", dpi=300)


##Part II
#Changing from data array to dataframe
dataframe= pd.DataFrame(data)


#Selected model - AIC
arma_model_1=ARIMA(dataframe[0:500], trend='c', order=(4, 0, 0))
resid_1=arma_model_1.fit()

#Selected model - BIC
arma_model_1=ARIMA(dataframe[0:500], trend='c', order=(1, 0, 0))
resid_2=arma_model_1.fit()

#forecasting 2 periods ahead
initial_fcst_aic=resid_1.forecast(steps=2)
initial_fcst_bic=resid_2.forecast(steps=2)

#computing the errors
error_base_aic_2period = dataframe.iloc[train_sample:train_sample+2,0] - initial_fcst_aic
error_base_bic_2period = dataframe.iloc[train_sample:train_sample+2,0] - initial_fcst_bic

#Computing the mean squared error
rmse_aic_2period = (error_base_aic_2period**2).mean()**0.5
rmse_bic_2period = (error_base_bic_2period**2).mean()**0.5

#saving RMSE
rmse_2period = {
    "ARMA, order (4,0,0), with const": rmse_aic_2period,
    "ARMA, order (1,0,0), with const": rmse_bic_2period
}


#https://otexts.com/fpp2/simple-methods.html

#regrading the forecast with the last data point used for estimation
aic_fcst_naive_last=pd.DataFrame([dataframe.iloc[train_sample-1], dataframe.iloc[train_sample-1]])
bic_fcst_naive_last=pd.DataFrame([dataframe.iloc[train_sample-1], dataframe.iloc[train_sample-1]]) 

#regrading the forecast with the average of the sample
aic_fcst_naive_mean=pd.DataFrame([dataframe.iloc[0:train_sample].mean(), dataframe.iloc[0:train_sample].mean()])
bic_fcst_naive_mean=pd.DataFrame([dataframe.iloc[0:train_sample].mean(), dataframe.iloc[0:train_sample].mean()]) 


#must add the option .values[0] because it is returning  more than just the number
error_base_naive_last_aic_2period = dataframe.iloc[train_sample:train_sample+2,0].values[0] - aic_fcst_naive_last.values[0]
error_base_naive_last_bic_2period = dataframe.iloc[train_sample:train_sample+2,0].values[0] - bic_fcst_naive_last.values[0]

error_base_naive_mean_aic_2period = dataframe.iloc[train_sample:train_sample+2,0].values[0] - aic_fcst_naive_mean.values[0]
error_base_naive_mean_bic_2period = dataframe.iloc[train_sample:train_sample+2,0].values[0] - aic_fcst_naive_mean.values[0]

#Computing the mean squared error
rmse_aic_last_2period = (error_base_naive_last_aic_2period**2).mean()**0.5
rmse_bic_last_2period = (error_base_naive_last_bic_2period**2).mean()**0.5

rmse_aic_mean_2period = (error_base_naive_mean_aic_2period**2).mean()**0.5
rmse_bic_mean_2period = (error_base_naive_mean_bic_2period**2).mean()**0.5

rmse_2period["Naive, last data point, order (4,0,0), with const"]=rmse_aic_last_2period
rmse_2period["Naive, last data point, order (1,0,0), with const"]=rmse_aic_last_2period

rmse_2period["Naive, training sample average, order (4,0,0), with const"]=rmse_aic_mean_2period
rmse_2period["Naive, training sample average, order (1,0,0), with const"]=rmse_bic_mean_2period

def genforecast(nforecasts,arima_order,arima_trend,data,tsize):
# Setup forecasts
    nobs=len(data)
    forecasts = {}
    init_training_endog = dataframe.iloc[:tsize]
    model=ARIMA(init_training_endog, trend=arima_trend, order=arima_order)
    modelresults=model.fit()

# Save initial forecast
    forecasts[init_training_endog.index[-1]] = modelresults.forecast(steps=nforecasts)
# Step through the rest of the sample
    for t in range(tsize, nobs-nforecasts):
    # Update the results by appending the next observation
        updated_data = dataframe.iloc[t:t+1]
        modelresults = modelresults.append(updated_data, refit=True)

    # Save the new set of forecasts
        forecasts[updated_data.index[0]] = modelresults.forecast(steps=nforecasts)

# Combine all forecasts into a dataframe
    forecasts=pd.concat(forecasts, axis=1)
    
    return forecasts


fcst_arima_aic_2period=genforecast(2, (4,0,0), 'c', dataframe, train_sample)
fcst_arima_bic_2period=genforecast(2, (1,0,0), 'c', dataframe, train_sample)



def comprmse(df):
    fcst_errors = {}
    for c in df.columns:
        fcst_errors[c] = dataframe.iloc[train_sample:, 0] - df.loc[:,c]

    fcst_errors = pd.DataFrame.from_dict(fcst_errors)    

    def flatten(column):
        return column.dropna().reset_index(drop=True)

    flattened = fcst_errors.apply(flatten)
    flattened.index = (flattened.index + 1).rename('horizon')
    flattened.T.head()

    fcst_errors = flattened.T.dropna().values.flatten()

    RMSE = (fcst_errors**2).mean()**0.5

    return RMSE

rmse_2period["Arma, adding one obs at a time, order (4,0,0), with const"]=comprmse(fcst_arima_aic_2period)
rmse_2period["Arma, adding one obs at a time, order (1,0,0), with const"]=comprmse(fcst_arima_bic_2period)


#Adding obs in the Naive for 2 periods
nobs=len(dataframe)
fcsts_last_2p = {}
init_training_endog = dataframe.iloc[:train_sample]

# Save initial forecast
fcsts_last_2p[init_training_endog.index[-1]] = pd.DataFrame([dataframe.iloc[train_sample-1], dataframe.iloc[train_sample-1]], index=[500,501])

# Step through the rest of the sample
for t in range(train_sample, nobs-2):
    # Update the results by appending the next observation
    updated_data = pd.DataFrame([dataframe.iloc[t], dataframe.iloc[t]], index=[t+1, t+2])
    
    # Save the new set of forecasts
    fcsts_last_2p[updated_data.index[0]-1] = updated_data
    
# Combine all forecasts into a dataframe
fcsts_last_2p=pd.concat(fcsts_last_2p, axis=1)

rmse_2period["Naive, last data point, adding one obs at a time, order (4,0,0) or (1,0,0), with const"]=comprmse(fcsts_last_2p)

#for 2 periods
nobs=len(dataframe)
fcsts_mean_2p = {}
init_training_endog = dataframe.iloc[:train_sample]

# Save initial forecast
fcsts_mean_2p[init_training_endog.index[-1]] = pd.DataFrame([init_training_endog.mean(),init_training_endog.mean()], index=[500,501])

# Step through the rest of the sample
for t in range(train_sample, nobs-2):
    # Update the results by appending the next observation
    updated_data = pd.DataFrame([dataframe.iloc[:t+1].mean(), dataframe.iloc[:t+1].mean()], index=[t+1, t+2])

    # Save the new set of forecasts
    fcsts_mean_2p[updated_data.index[0]-1] = updated_data
    
# Combine all forecasts into a dataframe
fcsts_mean_2p=pd.concat(fcsts_mean_2p, axis=1)

rmse_2period["Naive, training sample average, adding one obs at a time, order (4,0,0) or (1,0,0), with const"]=comprmse(fcsts_mean_2p)

min_2period = min(rmse_2period, key=rmse_2period.get)

print(f'Among the all the models the model that minimizes the root mean squared errors is: {min_2period}')
