#IMPORT E PIP SOPRA A TUTTO

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

df = pd.read_csv("customer_dataset.csv", sep=";")

df = df[df.automotive != "#N/D"]

# qui farei un dizionario del tipo
# dict = {1: 'agriculture suppliers', 2: 'automotive'...}
# così, quando invocherai le singole etichette come fai dopo, non dovrai scrivere tutta l'etichetta, ma solo
# lista = list(itemgetter(1, 2, 6)(dict))
# e quando hai bisogno di invocare le singole etichette 

dict = {1: 'agriculture suppliers', 2: 'automotive',
        3: 'bakeware', 4: 'beauty & personal care',
        5: 'bedroom decor', 6: 'book', 7: 'business office',
        8: 'camera & photo', 9: 'cd vinyl', 10: 'ceiling fans',
        11: 'cell phones', 12: 'cleaning supplies', 13: 'coffee machines',
        14: 'comics', 15: 'computer accessories', 16: 'computers tablets',
        17: 'diet sports nutrition', 18: 'dvd', 19: 'event & party supplies',
        20: 'fabric', 21: 'fashion & shoes', 22: 'film & photography, 
        23: 'fire safety', 24: 'food', 25: 'fragrance',
        26: 'furniture', 27: 'handbags & accessories', 28: 'hardware',
        29: 'headphones', 30: 'health household', 31: 'home accessories',
        32: 'home appliances', 33: 'home audio', 34: 'home emergency kits',
        35: 'home lighting', 36: 'home security systems', 37: 'jewelry',
        38: 'kids', 39: 'kids fashion', 40: 'kitchen & dining',
        41: 'lawn garden', 42: 'light bulbs', 43: 'luggage',
        44: 'mattresses & pillows', 45: 'medical supplies', 46: "men's fashion",
        47: 'model hobby building', 48: 'monitors', 49: 'music intstruments',
        50: 'office products', 51: 'oral care', 52: 'painting',
        53: 'pet food', 54: 'pet supplies', 55: 'safety apparel',
        56: 'seasonal decor', 57: 'sofa', 58: 'sport outdoors',
        59: 'television & video', 60: 'tools home improvement',
        61: 'toys games', 62: 'underwear', 63: 'videogame', 
        64: 'videogame console', 65: 'wall art', 66: 'watches',
        67: 'wellness & relaxation', 68: "woman's fashion")

#OTTENERE LA LISTA DEI SOLI VALORI DEL DIZIONARIO, QUINDI LISTA CONTENENTE ETICHETTE (FORSE NON NECESSARIA)
#labels = list(map(lambda key: dict[key],dict)) 
        
# NON HO CAPITO RIGO 47 :'(
#df[productcategory_names] = df[productcategory_names].apply(pd.to_numeric, errors='coerce')

lista = list(itemgetter(3, 5, 12, 30, 31, 32, 33, 34, 35, 36,
                        40, 44, 56, 60, 65, 19, 43, 45, 53, 54,
                        24, 38, 51)(dict))
        
df['Home']= df[lista].sum(axis=1)

# RIPETERE VARIABILE "LISTA": LA SOVRASCRITTURA NON È UN PROBLEMA, MENO COMANDI "DEL" DOPO E MENO MEMORIA UTILIZZATA

lista = list(itemgetter(10, 13, 7, 26, 42, 50, 23, 2, 57)(dict))
df['Forniture']= df[lista].sum(axis=1)

del lista
del productcategory_names

######################################### ANCORA DA REFACTORARE ############################################

df_cluster = df.iloc[:, [0,1,2,3,4] #INDICI COLONNA NECESSARI
df_cluster = df_cluster.replace('#N/D', np.NaN)

'''
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(df_cluster[['most_used_paym_meth']]).toarray())
# merge with main df bridge_df on key values
df_cluster = df_cluster.join(enc_df)
'''
#CONCATENATO OPERAZIONI
dummy_df = pd.get_dummies(df_cluster["most_used_paym_meth"]).drop(["most_used_paym_meth"], axis=1).join(dummy_df)


for column in df_cluster.columns[1:]:
    df_cluster[column] = pd.to_numeric(df_cluster[column])
    df_cluster[column].fillna((df_cluster[column].median()), inplace=True)

x = df_cluster.iloc[:, df_cluster.columns != "customer_unique_id"].values
x = StandardScaler().fit_transform(x) # normalizing the features

pca_breast = PCA(n_components=8)
principalComponents_breast = pca_breast.fit_transform(x)
#principal_breast_Df = pd.DataFrame(data = principalComponents_breast
#             , columns = ['principal component 1', 'principal component 2'])


print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))

# >> Explained variation per principal component: [0.36198848 0.1920749 ]

print('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca_breast.explained_variance_ratio_)))



