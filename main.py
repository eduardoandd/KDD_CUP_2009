# ======================= OBJETIVO =======================
# A ideia geral envolve a criação de um modelo para predição dessas 3 variáveis com base 
# nos dados fornecidos por uma base de dados real de uma empresa de telecomunicação francesa.

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

# ======================= SELEÇÃO DE VARIÁVEIS =======================
# Objetivo: definir as vars que serão utilizados no modelos
features.isna()
empty_entries_per_column=features.isna().sum(axis=0) # contagem de linhas nulas

#graficos para ajudar a definir o fator de corte
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.boxplot(empty_entries_per_column)
ax2.hist(empty_entries_per_column)
median=empty_entries_per_column.median()
print(median)

#definindo limite (threshold)
threshold = len(features) * 0.25
keep_vars= np.array(features.columns[(empty_entries_per_column <= threshold)])

num_vars=[var for var in num_vars if var in keep_vars]
cat_vars=[var for var in cat_vars if var in keep_vars]

#preenchimento de valores nulos restantes
for col in num_vars:
    col_mean= features[col].mean()
    features[col]=features[col].fillna(col_mean)

for col in cat_vars:
    features[col]=features[col].cat.add_categories('missing')
    features[col]=features[col].fillna('missing')

# ======================= LIMPANDO VARIÁVEIS COM MUITAS CATEGORIAS =======================
n_categories_per_features=features[cat_vars].apply(lambda x: len(set(x))) # retorna o numero unico da categoria para cada var
plt.hist(n_categories_per_features)
cat_vars= np.array(n_categories_per_features[n_categories_per_features <= 1400].index) # variaveis com menos quantidade de null

# ======================= CONSOLIDANDO DATAFRAME =======================
features=features[list(num_vars)+ list(cat_vars)]
