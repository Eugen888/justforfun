### Importing the necessary libraries <!-- H3 -->
- Bullet 1



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

from sklearn.linear_model import LinearRegression
```


```python
path = "/Users/eugen/Desktop/python/Rebuilding/1000_companies.csv"
df = pd.read_csv(path, index_col='State')
np.set_printoptions(threshold=np.inf, suppress=True, floatmode='maxprec')

#---Variables & Definitions---#
x = df['R&D Spend'].to_numpy().astype(float)
y = df['Profit'].to_numpy()
n = x.size
p = 1
```


```python
#---Coefficients---#
Sxx = np.sum((x - x.mean() ) ** 2)
Sxy = np.sum((x - x.mean()) * (y - y.mean()))
beta1 = Sxy / Sxx
beta0 = y.mean() - beta1 * x.mean()
print(f"β0 (intercept): {beta0}")
print(f"β1 (slope):     {beta1}")
print('\n' + '-'*50)
```

    β0 (intercept): 48401.89881039558
    β1 (slope):     0.8711301627727812
    
    --------------------------------------------------



```python
#---Predictions and Residuals---#
y_hat = beta0 + beta1 * x
resid = y - y_hat
print(f"First 3 ŷ: {y_hat[:3]}")
print(f"First 3 e: {resid[:3]}")
print('\n' + '-'*50)
```

    First 3 ŷ: [192442.57432074 190045.65967788 182069.4263928 ]
    First 3 e: [-180.74432074 1746.40032212 8980.9636072 ]
    
    --------------------------------------------------



```python
#---R_squared---#
SSE = np.sum(resid ** 2)
SST = np.sum((y - y.mean()) **2)
SSR = SST - SSE
R2 = 1 - SSE / SST
R2_adj = 1 - (1-R2) * (n - 1) / (n - p - 1)
print(f"SST: {SST} | SSR: {SSR} | SSE: {SSE}")
print(f"R²: {R2} \nR²_adj: {R2_adj}")
print('\n' + '-'*50)
```

    SST: 1837595478475.3276 | SSR: 1641870714657.7441 | SSE: 195724763817.58344
    R²: 0.893488656175853 
    R²_adj: 0.893381931382442
    
    --------------------------------------------------



```python
#---Error Variance & Error Metrics---#
df_resid = n - (p + 1)
sigma2 = SSE / df_resid
MSE = SSE / n
RMSE = np.sqrt(MSE)
MAE = np.mean(np.abs(resid))
print(f"σ² (unbiased): {sigma2} | df_resid: {df_resid}")
print(f"MSE: {MSE} | RMSE: {RMSE} | MAE: {MAE}")
print('\n' + '-'*50)
 
```

    σ² (unbiased): 196116997.81320986 | df_resid: 998
    MSE: 195724763.81758344 | RMSE: 13990.166682980709 | MAE: 1927.5495906222081
    
    --------------------------------------------------



```python
#---Standard Errors (coefficients)---#
SE_beta1 = np.sqrt(sigma2 / Sxx)
SE_beta0 = np.sqrt(sigma2 * (1 / n + (x.mean() ** 2) / Sxx))
print(f"SE(β0): {SE_beta0} | SE(β1): {SE_beta1}")

#---T-statistics---#
t_beta0 = beta0 / SE_beta0
t_beta1 = beta1 / SE_beta1
print(f"t(β0): {t_beta0} | t(β1): {t_beta1}")

#---P-values---#
p_beta0 = 2 * (1 - t.cdf(abs(t_beta0), df_resid))
p_beta1 = 2 * (1 - t.cdf(abs(t_beta1), df_resid))
print(f"p(β0): {p_beta0:.10f}")
print(f"p(β1): {p_beta1:.10f}")
```

    SE(β0): 894.8185349540317 | SE(β1): 0.009520750654751877
    t(β0): 54.09130110708096 | t(β1): 91.49805455077154
    p(β0): 0.0000000000
    p(β1): 0.0000000000



```python
print("\n---- 7) CONFIDENCE INTERVALS (95%) ----")
alpha = 0.05
tcrit = t.ppf(1 - alpha / 2, df_resid)
ci_beta0 = (beta0 - tcrit * SE_beta0, beta0 + tcrit * SE_beta0)
ci_beta1 = (beta1 - tcrit * SE_beta1, beta1 + tcrit * SE_beta1)
print(f"CI95 β0: {ci_beta0}")
print(f"CI95 β1: {ci_beta1}")


print("\n---- 9) ANOVA & F-STAT ----")
MSR = SSR / p
MSE_resid = SSE / df_resid
F_stat = MSR / MSE_resid
print(f"MSR: {MSR} | MSE(resid): {MSE_resid} | F: {F_stat}")

print("\n---- 10) DURBIN–WATSON ----")
DW = np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)
print(f"Durbin–Watson: {DW}")

```

    
    ---- 7) CONFIDENCE INTERVALS (95%) ----
    CI95 β0: (np.float64(46645.957164848216), np.float64(50157.84045594294))
    CI95 β1: (np.float64(0.8524471763140687), np.float64(0.8898131492314937))
    
    ---- 9) ANOVA & F-STAT ----
    MSR: 1641870714657.7441 | MSE(resid): 196116997.81320986 | F: 8371.89398657597
    
    ---- 10) DURBIN–WATSON ----
    Durbin–Watson: 0.5589355008794035

