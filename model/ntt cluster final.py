import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("customer_dataset.csv", sep=";")

'''
plt.hist(data=x, x= "3")
plt.show()
'''

df = df[df.automotive != "#N/D"]

productcategory_names = ['agriculture suppliers', 'automotive',
       'bakeware', 'beauty & personal care', 'bedroom decor', 'book',
       'business office', 'camera & photo', 'cd vinyl', 'ceiling fans',
       'cell phones', 'cleaning supplies', 'coffee machines', 'comics',
       'computer accessories', 'computers tablets', 'diet sports nutrition',
       'dvd', 'event & party supplies', 'fabric', 'fashion & shoes',
       'film & photography', 'fire safety', 'food', 'fragrance', 'furniture',
       'handbags & accessories', 'hardware', 'headphones', 'health household',
       'home accessories', 'home appliances', 'home audio',
       'home emergency kits', 'home lighting', 'home security systems',
       'jewelry', 'kids', 'kids fashion', 'kitchen & dining', 'lawn garden',
       'light bulbs', 'luggage', 'mattresses & pillows', 'medical supplies',
       "men's fashion", 'model hobby building', 'monitors',
       'music instruments', 'office products', 'oral care', 'painting',
       'pet food', 'pet supplies', 'safety apparel', 'seasonal decor', 'sofa',
       'sport outdoors', 'television & video', 'tools home improvement',
       'toys games', 'underwear', 'videogame', 'videogame console', 'wall art',
       'watches', 'wellness & relaxation', "woman's fashion"]

df[productcategory_names] = df[productcategory_names].apply(pd.to_numeric, errors='coerce')



column_names_home = ['bakeware', 'bedroom decor', 'cleaning supplies',
                     'home accessories', 'home appliances', 'home audio',
                     'home emergency kits', 'home lighting', 'home security systems',
                     'health household', 'kitchen & dining', 'mattresses & pillows',
                     'seasonal decor', 'tools home improvement', 'wall art',
                     'event & party supplies', 'luggage', 'medical supplies', 'pet food',
                     'pet supplies', 'food', 'kids', 'toys games']
df['Home']= df[column_names_home].sum(axis=1)


column_names_forniture = ['ceiling fans', 'coffee machines', 'business office','furniture',
                          'light bulbs', 'office products', 'fire safety', 'automotive', 'sofa']
df['Forniture']= df[column_names_forniture].sum(axis=1)

column_names_tech = ['camera & photo', 'cd vinyl', 'cell phones', 'computer accessories', 'computers tablets','dvd',
                     'film & photography', 'hardware', 'headphones', 'monitors', 'television & video',
                     'videogame', 'videogame console']
df['Technology']= df[column_names_tech].sum(axis=1)

column_names_fashion =['beauty & personal care', 'fashion & shoes', 'fabric',
                       'handbags & accessories', 'fragrance','jewelry', "men's fashion", 'underwear',
                       'watches', 'wellness & relaxation',"woman's fashion", 'kids fashion',
                       'diet sports nutrition', 'oral care', 'safety apparel']
df['Fashion/Personal care']= df[column_names_fashion].sum(axis=1)


column_names_hobbies = ['agriculture suppliers', 'book', 'comics','lawn garden',
                        'model hobby building', 'painting','sport outdoors', 'music instruments']
df['Hobbies']= df[column_names_hobbies].sum(axis=1)

del column_names_home
del column_names_tech
del column_names_fashion
del column_names_forniture
del column_names_hobbies
del productcategory_names

df_cluster = df[['customer_unique_id', "order_count (only positive orders)", "product count", "review_rate (n. recensioni / n. prodotti acquistati)",
         "score_medio_reviews", "costoprodotti_medio","costoprodotti_totale", "spedizioni_medie", "spedizioni_totale",
         "most_used_paym_meth", "n_medio_pagamenti", "n_medio_rate",
                 'Home', 'Forniture', 'Technology', 'Fashion/Personal care', 'Hobbies', 'most_frequent_weekday' ]]


df_cluster = df_cluster.replace('#N/D', np.NaN)


dummy_df = pd.get_dummies(df_cluster["most_used_paym_meth"])
df_cluster = df_cluster.drop(["most_used_paym_meth"], axis=1)
df_cluster = df_cluster.join(dummy_df)

for column in df_cluster.columns[1:]:
    df_cluster[column] = pd.to_numeric(df_cluster[column])
    df_cluster[column].fillna((df_cluster[column].mean()), inplace=True)

df_cluster['total_money_spent'] = df_cluster['costoprodotti_totale'] + df_cluster['spedizioni_totale']


###### CLUSTER


x = df_cluster.iloc[:, [1,2,9,10,11,12,13,14,15,21]].values

from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x) # normalizing the features

### K-means

#choosing the right number of k through the elbow method
from sklearn.cluster import KMeans
objective_function=[]
for i in range(1,20):
    clustering=KMeans(n_clusters=i, init='k-means++')
    clustering.fit(x)
    objective_function.append(clustering.inertia_)

plt.plot(range(1,20),objective_function)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters K')
plt.ylabel('objective_function')
plt.show()

#algorithm
kmeans = KMeans(n_clusters=12, init='k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(x)
df_cluster['cluster_kmeans'] = y_kmeans

df_cluster_features = df_cluster[['order_count (only positive orders)','product count', 'n_medio_pagamenti',
       'n_medio_rate', 'Home', 'Forniture', 'Technology',
       'Fashion/Personal care', 'Hobbies','total_money_spent','cluster_kmeans']]
'''
df_cluster_prova = df_cluster[['product count', 'n_medio_pagamenti',
       'n_medio_rate', 'most_frequent_weekday','total_money_spent','cluster_kmeans']]
'''
#mean analysis of clusters per variable
mean_clusters_prova_kmeans = df_cluster_features.groupby("cluster_kmeans").mean()
mean_clusters_prova_kmeans['size'] = df_cluster_features.groupby(['cluster_kmeans'])['cluster_kmeans'].transform(len)

mean_clusters_prova_kmeans['size'] = [15769, 23773, 9876, 16761, 8845, 2778, 13505, 39, 693, 1776, 572, 1143 ]




###########  RADAR CHART

#NORMAL PEOPLE
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.renderers.default = 'browser'
categories = ['Order count','Product count', 'Average n. of payments',
       'Average n. of installments', 'Home', 'Forniture', 'Technology',
       'Fashion/Personal care', 'Hobbies','Total money spent']

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
      r=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2, 0.5],
      theta=categories,
      #fill='toself',
      name='AVERAGE CUSTOMER - HOBBIES (0)'
))
fig.add_trace(go.Scatterpolar(
      r=[0.5, 0.5, 0.5, 0.5, 2, 0.5, 0.5, 0.5, 0.5, 0.5],
      theta=categories,
      #fill='toself',
      name='AVERAGE CUSTOMER - HOME (1)'
))

fig.add_trace(go.Scatterpolar(
      r=[0.5, 0.5, 0.5, 0.5, 0.5, 2, 0.5, 0.5, 0.5, 0.5],
      theta=categories,
      #fill='toself',
      name='AVERAGE CUSTOMER - FURNITURE (2)'
))
fig.add_trace(go.Scatterpolar(
      r=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2, 0.5, 0.5],
      theta=categories,
      #fill='toself',
      name='AVERAGE CUSTOMER - FASHION/PERSONAL CARE (3)'
))
fig.add_trace(go.Scatterpolar(
      r=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2, 0.5, 0.5, 0.5],
      theta=categories,
      #fill='toself',
      name='AVERAGE CUSTOMER - TECHNOLOGY (6)'
))
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 2]
    )),
  showlegend=True,
)
fig.update_traces(marker_opacity=0.1, fill="toself")
fig.show()



## INTERESTING CLUSTERS

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
      r=[1, 1, 1, 10, 1, 1, 1, 1, 1, 3],
      theta=categories,
      #fill='toself',
      name='CLUSTER 4'
))
fig.add_trace(go.Scatterpolar(
      r=[6, 4, 1, 3, 1, 1, 1, 1, 1, 3],
      theta=categories,
      #fill='toself',
      name='CLUSTER 5'
))

fig.add_trace(go.Scatterpolar(
      r=[1, 8, 1, 3.5, 8, 1, 1, 1, 1, 4.5],
      theta=categories,
      #fill='toself',
      name='CLUSTER 8'
))
fig.add_trace(go.Scatterpolar(
      r=[1, 4, 1, 3.5, 1, 6, 1, 1, 1, 3],
      theta=categories,
      #fill='toself',
      name='CLUSTER 9'
))
fig.add_trace(go.Scatterpolar(
      r=[1, 1, 1, 7, 1, 1, 1, 1, 1, 10],
      theta=categories,
      #fill='toself',
      name='CLUSTER 11'
))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 10]
    )),
  showlegend=True,
)
fig.update_traces(marker_opacity=0.1, fill="toself")
fig.update_layout(polar = dict(radialaxis = dict(showticklabels = False)))
fig.show()
















