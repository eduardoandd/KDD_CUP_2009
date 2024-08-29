# ======================= OBJETIVO =======================
# A ideia geral envolve a criação de um modelo para predição dessas 3 variáveis com base 
# nos dados fornecidos por essa base de dados real de uma empresa de telecomunicação francesa.

#	Possibilidade do cliente fazer cancelamento da conta
#	Tendência de comprar novos produtos ou serviços
#	Pretensão de comprar upgrades ou adicionais

# ======================= IMPORTAÇÕES =======================
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split,cross_val_predict,GridSearchCV

# ======================= CARREGAMENTO =======================
features=pd.read_csv('orange_small_train.data',sep='\t', na_filter=False) # var independente
#1=sim -1=nao
outcome= pd.read_csv('orange_small_train_churn.labels', header=None) # variavel dependente
np.unique(outcome)

# ======================= VAR NUMERICAS E CATEGORICAS =======================
all_var=np.array(features.columns)
num_vars=np.array(all_var[0:190])
cat_vars=np.array(all_var[190:])

# ======================= VERIFICAÇÃO DA CONSITÊNCIA DAS VARIÁVEIS =======================
# objetivo: ver se todas as entradas numericas são numeros (ser consitente)
features.dtypes.value_counts()
features['Var1'].value_counts()

counts_per_column= pd.DataFrame()
for col in num_vars:
    col_count=features[col].value_counts()
    counts_per_column=pd.concat([counts_per_column,col_count],axis=1, ignore_index=True)

counts_per_column.index=counts_per_column.index.astype('str')
counts_per_column.sort_index(inplace=True)

# objetivo: ver se todas as entradas categoricas são categoricas(ser consistente)
features.iloc[:,190].value_counts()
counts_per_column_cat=pd.DataFrame()
for col in cat_vars:
    col_count=features[col].value_counts()
    counts_per_column_cat=pd.concat([counts_per_column_cat,col_count],axis=1, ignore_index=True)

counts_per_column_cat.index=counts_per_column_cat.index.astype('str')
counts_per_column_cat.sort_index(inplace=True)

# ======================= TRATAMENTO DE VALORES FALTANTES =======================
features=features.replace('',np.nan)

# ======================= CONVERSÃO DE VAR PARA TIPOS APROPRIADOS =======================
for col in num_vars:
    features[col]=features[col].astype('float')

for col in cat_vars:
    features[col]=features[col].astype('category')
