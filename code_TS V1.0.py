#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:16:27 2019

@author: Diogo
"""

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.tsa.api as smt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf # Auto-correlation function
from statsmodels.graphics.tsaplots import plot_pacf # Partial auto-correlation function


df = pd.read_csv("MerckDaily.csv", index_col = "Date")

# Informação técnica sobre o DataFrame (ex: tipo de variáveis, quantas linhas "nan")
df.info()


### Estatísticas descritivas ###
# Medidas de estatística descritiva
print("\n", "Estatísticas descritivas")
print(df.describe())

# Mudar o nome da coluna para ser mais fácil de aceder à mesma
df.rename(columns={'Adj Close': 'adjClose'}, inplace=True) 

sk = stats.skew(df.adjClose)
print("Skewness of Adjusted Close:", sk) # Skewness > 0 => positively skewed (to the right)

k = stats.kurtosis(df.adjClose) 
print("Kurtosis of Adjusted Close:", k) # Kurtosis < 0 => distribution is platykurtic

sns.distplot(df.adjClose)

mx = stats.jarque_bera(df.adjClose)
print("\n", "P-value do test Jarque-Bera:", mx[1]) # P-value < 0.05 => rejeitamos a hipótese nula de que a variável segue uma distribuição normal


### Gráficos ###
# Gráficos da série, ACF e PACF
def tsplot(y, title, lags=None, figsize=(14, 7)):
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style='bmh'):
        plt.figure(figsize=figsize) # define o tamanho do output (gráficos)
        layout = (2,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1,0))
        pacf_ax = plt.subplot2grid(layout, (1,1))
        
        y.plot(ax=ts_ax)
#        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title(title)
#        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        
tsplot(df.adjClose, "Gráficos da série original (por níveis)", lags=365,) # Gráficos série original (por níveis)

tsReturns = (df.adjClose - df.adjClose.shift(1)) / df.adjClose.shift(1) # Série dos retornos simples
tsReturns = tsReturns.dropna(how = "any")
tsplot(tsReturns, "Gráficos da série dos retornos", lags=365) # Gráficos série dos retornos simples

tsLnReturns = np.log( df.adjClose / df.adjClose.shift(1) ) # Série do logarítmo natural (ln) dos retornos
tsLnReturns = tsReturns.dropna(how = "any")
tsplot(tsLnReturns, "Gráficos da série do logarítmo natural (ln) dos retornos", lags=365) # Gráficos série do logarítmo natural (ln) dos retornos
plt.show()

# Média da da série dos retornos simples (uso  isto para determinar se a série tem média não nula)
print("\n", "Média da série dos retornos simples:", tsReturns.mean())


### Testes de estacionaridade ###
# Augmented Dickey-Fuller
from statsmodels.tsa.stattools import adfuller

adf1 = adfuller(df.adjClose, regression = "ct")
print("\n", "Teste 'Augmented Dickey-Fuller' para série por níveis (p-value):", adf1[1]) # P-value > 0.05 => série é não estacionária

adf2 = adfuller(tsReturns, regression = "nc")
print("\n", "Teste 'Augmented Dickey-Fuller' para série de retornos (p-value):", adf2[1]) # P-value = 0 < 0.05 => série é estacionária

# Phippips Perron (PP), com H_0: série é não estacionária
from arch.unitroot import PhillipsPerron

pp1 = PhillipsPerron(df.adjClose)
print("\n", "Teste 'Phillips Perron' para série por níveis:", pp1) # P-value > 0.05 => série é não estacionária

pp2 = PhillipsPerron(tsReturns)
print("\n", "Teste 'Phillips Perron' para série de retornos:", pp2) # P-value = 0 < 0.05 => série é estacionária

# Kwiatkowski-Phillips-Schmidt-Shin (KPSS), com H_0: série é estacionária
from statsmodels.tsa.stattools import kpss

kpss1 = kpss(df.adjClose, regression='ct')
print("\n", "Teste KPSS para série por níveis:", kpss1[1]) # P-value < 0.05 => série é não estacionária

kpss2 = kpss(tsReturns, regression='ct')
print("\n", "Teste KPSS para série por níveis:", kpss2[1]) # P-value > 0.05 => série é estacionária


### Procurar o melhor modelo ARIMA ### Função auto_arima usa por defeito o valor de AIC para determinar o modelo ótimo.
from statsmodels.tsa.arima_model import ARIMA
#import pmdarima as pm
#
#model = pm.auto_arima(df.adjClose, start_p=1, start_q=1,
#                      test='adf',       # Diogo: usa o teste ADF para testar a estacionaridade da série. H_0 = série é não estacionária
#                      max_p=4, max_q=4, # maximum p and q
#                      m=1,              # frequency of series
#                      d=1,           # Diogo: estava "None" mas decidi colocar "1" pois já tinha determinado anteriormente que a série em níveis era I(1)
#                      seasonal=False,   # No Seasonality
#                      start_P=0, 
#                      D=0, 
#                      trace=True,
#                      error_action='ignore',  
#                      suppress_warnings=True, 
#                      stepwise=True)
#
#print(model.summary()) # O melhor modelo é ARIMA(2,1,1)

model = ARIMA(df.adjClose, order=(2,1,1))
model_output = model.fit(disp=0)
print(model_output.summary())


### Análise de resíduos do modelo ARIMA ### Resíduos devem ser ruído branco (ver slide 29 da aula6) para podermos validar o modelo
# Representação gráfica
plt.rcParams.update({'figure.figsize':(16,8), 'figure.dpi':70})
resid = pd.DataFrame(model_output.resid)
fig, ax = plt.subplots(1,2)
resid.plot(title="Residuals", ax=ax[0])
resid.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Medidas de estatística descritiva dos resíduos
print(resid.describe())

# Gráficos ACF e PACF para resíduos
plot_acf(resid)
plot_pacf(resid)
plt.show()

# Teste de Ljung-Box autocorrelação dos resíduos
# H_0: resíduos independentes até lag definida (10 neste caso)
# Os p-values estão no segundo grupo de valores
from statsmodels.stats.diagnostic import acorr_ljungbox
lb = acorr_ljungbox(resid, lags=10)
print("\n", 'Teste de Ljung-Box de independência dos resíduos:', lb[1]) # A partir do 7o lag, p-value < 0.05 => esses lags apresentam auto-correlação, o que viola
 # o pressuposto de independência dos resíduos
 
## Teste de Durbin-Watson para autocorrelação dos resíduos. H_0: resíduos não têm autocorrelação com o seu 1o lag
#from statsmodels.stats.stattools import durbin_watson
#DW = durbin_watson(resid)
#print("\n", 'Teste de Durbin-Watson de independência dos resíduos:', DW[0])
 
# Teste de Breush-Godfrey para autocorrelação dos resíduos. H_0: resíduos não têm correlação com os seus "n" lags (neste caso definimos n = 10)
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
bg = acorr_breusch_godfrey(model_output, nlags=10)
print("\n", 'Teste de Breush-Godfrey de independência dos resíduos:', bg) # P-value (2o valor dos 4 apresentados) < 0.05 => existe autocorrelação dos resíduos

# Teste de heterocedasticidade ARCH. H_0: variância é constante. O segundo valor é o p-value
from statsmodels.stats.diagnostic import het_arch
archTest = het_arch(resid[0], maxlag=5, autolag=None)
print("\n", 'Teste ARCH de heterocedasticidade:', archTest[1]) # P-value < 0.05 => rejeita-se H_0 => variância não é constante


### GJR-GARCH Model ### AR(2) + GJR-GARCH(1,1)
from arch import arch_model 

gjrGarch = arch_model(tsReturns, mean = "ARX", lags = 2, o = 1) # importa as 3 equações ao mesmo tempo
model_gjrGarch = gjrGarch.fit(update_freq=5)
print("\n", model_gjrGarch.summary())


### EGARCH Model ### AR(2) + EGARCH(1,1)
from arch import arch_model

egarch = arch_model(tsReturns, mean = "ARX", lags = 2, vol = "EGARCH", o=1) # importa as 3 equações ao mesmo tempo
model_egarch = egarch.fit(update_freq=5)
print("\n", model_egarch.summary())


### Análise de resíduos para modelo EGARCH ### Resíduos devem ser ruído branco (ver slide 29 da aula6) para podermos validar o modelo
# Representação gráfica
residEGarch = pd.DataFrame(model_egarch.resid).dropna()
fig, ax = plt.subplots(1,2)
residEGarch.plot(title="Residuals", ax=ax[0])
residEGarch.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Medidas de estatística descritiva dos resíduos
print(residEGarch.describe())

# Gráficos ACF e PACF para resíduos
plot_acf(residEGarch)
plot_pacf(residEGarch)
plt.show()

# Teste de Ljung-Box autocorrelação dos resíduos
# H_0: resíduos independentes dos resíduos de cada lag k (k = 1,2,...,10. Estava 10 por defeito mas podia-se escolher outro valor)
# Os p-values estão no segundo grupo de valores
lb = acorr_ljungbox(residEGarch, lags=10)
print("\n", 'Teste de Ljung-Box de independência dos resíduos:', lb[1]) # Alguns p-value < 0.05 => Existe autocorrelação dos erros
 
# Teste de Breush-Godfrey para autocorrelação dos resíduos. H_0: resíduos não têm correlação com os seus "n" lags (neste caso definimos n = 10)
#from statsmodels.stats.diagnostic import acorr_breusch_godfrey
#
#bgGJR = acorr_breusch_godfrey(model_egarch, nlags=10)
#print("\n", 'Teste de Breush-Godfrey de independência dos resíduos:', bgGJR) # P-value (2o valor dos 4 apresentados) < 0.05 => existe autocorrelação dos resíduos


### Previsão ###
# EGARCH [AR(2) + EGARCH(1,1)]- previsão de um ponto out-of-sample 
forecasts = model_egarch.forecast(horizon = 1, align = "origin")
print("\n", "Previsão de retorno, 1 dia no futuro:", forecasts.mean.iloc[-1])

