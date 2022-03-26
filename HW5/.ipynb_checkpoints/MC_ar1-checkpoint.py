import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from numpy.random import default_rng
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "alpha_0",
    type=float
)

parser.add_argument(
    "alpha_1",
    type=float
)

parser.add_argument(
    "sigma_sq",
    type=float
)

parser.add_argument(
    "n_rep",
    type=int
)


parser.add_argument(
    "n_obs",
    type=int
)


args = parser.parse_args()


gen = np.random.default_rng(739)

ar_param = np.array([args.alpha_1])
ma_param = np.array([0])
ar = np.r_[1, -args.alpha_1]
ma = np.r_[1, 0]
ar1_process = ArmaProcess(ar, ma)



data = ar1_process.generate_sample(nsample = (args.n_obs, args.n_rep),  burnin=1000,  scale=args.sigma_sq, distrvs=gen.normal)
ones = np.full((args.n_obs,args.n_rep),args.alpha_0)
data = data + ones
    
#Estimation method - OLS
thetaOLS = np.empty((0, 3))

for x in range(0, 500):
    AR_OLS_model = AutoReg(data[:,x], trend='c', lags=1)
    AR_OLS_results=AR_OLS_model.fit()
    thetaOLS=np.append(thetaOLS, [[AR_OLS_results.params[0], AR_OLS_results.params[1], AR_OLS_results.resid.std()]], axis=0)
    
#Estimation method - MLE  
thetaMLE = np.empty((0, 3))

for x in range(0, 500):
    AR_MLE_model = ARIMA(data[:,x], trend='c', order=(1, 0, 0))
    AR_MLE_results=AR_MLE_model.fit()
    thetaMLE=np.append(thetaMLE, [[AR_MLE_results.params[0], AR_MLE_results.params[1], AR_MLE_results.resid.std()]], axis=0)
    

    
#BIASES

OLS_alpha0_bias=(np.subtract(args.alpha_0,thetaOLS[:,0]).mean()).round(3)
OLS_alpha1_bias=(np.subtract(args.alpha_1,thetaOLS[:,1]).mean()).round(3)
OLS_sigmasq_bias=(np.subtract(args.sigma_sq,thetaOLS[:,2]).mean()).round(3)

MLE_alpha0_bias=(np.subtract(args.alpha_0,thetaMLE[:,0]).mean()).round(3)
MLE_alpha1_bias=(np.subtract(args.alpha_1,thetaMLE[:,1]).mean()).round(3)
MLE_sigmasq_bias=(np.subtract(args.sigma_sq,thetaMLE[:,2]).mean()).round(3)

print(f"Regarding alpha0, the bias is {OLS_alpha0_bias}, in the OLS, and {MLE_alpha0_bias}, in the MLE.")
print(f"Regarding alpha1, the bias is {OLS_alpha1_bias}, in the OLS, and {MLE_alpha1_bias}, in the MLE.")
print(f"Regarding sigma square, the bias is {OLS_sigmasq_bias}, in the OLS, and {MLE_sigmasq_bias}, in the MLE.")

#Mean squared errors
OLS_alpha0_mse=(np.square(np.subtract(args.alpha_0,thetaOLS[:,0]))).mean().round(3)
OLS_alpha1_mse=(np.square(np.subtract(args.alpha_1,thetaOLS[:,1]))).mean().round(3)
OLS_sigmasq_mse=(np.square(np.subtract(args.sigma_sq,thetaOLS[:,2]))).mean().round(3)

MLE_alpha0_mse=(np.square(np.subtract(args.alpha_0,thetaMLE[:,0]))).mean().round(3)
MLE_alpha1_mse=(np.square(np.subtract(args.alpha_1,thetaMLE[:,1]))).mean().round(3)
MLE_sigmasq_mse=(np.square(np.subtract(args.sigma_sq,thetaMLE[:,2]))).mean().round(3)

print(f"Regarding alpha0, the  mean square errors are {OLS_alpha0_mse}, in the OLS, and {MLE_alpha0_mse}, in the MLE.")
print(f"Regarding alpha1, the  mean square errors are {OLS_alpha1_mse}, in the OLS, and {MLE_alpha1_mse}, in the MLE.")
print(f"Regarding sigma square, the  mean square errors are {OLS_sigmasq_mse}, in the OLS, and {MLE_sigmasq_mse}, in the MLE.")