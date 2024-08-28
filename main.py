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
features=pd.read_csv('orange_small_test.data',sep='\t', na_filter=False) #atributos previsores
#1=sim -1=nao
outcome= pd.read_csv('orange_small_train_churn.labels', header=None) # variavel dependente
np.unique(outcome)

# ======================= VAR NUMERICAS E CATEGORICAS =======================
all_var=np.array(features.columns)
num_vars=np.array(all_var[0:190])
cat_vars=np.array(all_var[190:])


