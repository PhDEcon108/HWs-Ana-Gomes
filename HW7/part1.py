import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.arima_process import ArmaProcess
from numpy.random import default_rng
import numpy as np
from statsmodels.tsa.stattools import adfuller

import warnings
warnings.filterwarnings('ignore')


def test(n, t, alpha, p_value=0.05):
    gen = np.random.default_rng(100)
    ar_param = np.array([alpha])
    ma_param = np.array([0])
    ar = np.r_[1, -alpha]
    ma = np.r_[1, 0]
    ar1_process = ArmaProcess(ar, ma)

    data = ar1_process.generate_sample(nsample = (t, n),  burnin=1000, distrvs=gen.normal)
    
    number_rej_H0=0
    for c in range(0,n):
        dftest = adfuller(data[:,c], maxlag=1, autolag=None, regression='n')
        if dftest[1] < p_value:
            number_rej_H0=number_rej_H0+1
    percentage_rej= number_rej_H0/n
    
    return t, n, alpha, percentage_rej

table=pd.DataFrame(columns=['T=100', 'T=500', 'T=1000'], index=['alpha=0.9','alpha=0.95','alpha=0.99'])

for c in {100, 500, 1000}:
    for x in {.9, 0.95, 0.99}:
        T,N,alpha,rej=test(1000,c,x);
        table.loc[f'alpha={x}',f'T={c}']=rej

print(table)