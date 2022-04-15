import pandas as pd
import numpy as np

##### CLUSTERING NTT

df = pd.read_csv("customer_dataset.csv", sep=";")

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
         "score_medio_reviews", "costoprodotti_medio", "spedizioni_medie",
         "most_used_paym_meth", "n_medio_pagamenti", "n_medio_rate", "most_frequent_time_bin",
                 "most_frequent_weekday_bin", "most_frequent_month_bin",
                 'Home', 'Forniture', 'Technology', 'Fashion/Personal care', 'Hobbies' ]]

df_cluster = df_cluster.replace('#N/D', np.NaN)

'''
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(df_cluster[['most_used_paym_meth']]).toarray())
# merge with main df bridge_df on key values
df_cluster = df_cluster.join(enc_df)
'''

dummy_df = pd.get_dummies(df_cluster["most_used_paym_meth"])
df_cluster = df_cluster.drop(["most_used_paym_meth"], axis=1)
df_cluster = df_cluster.join(dummy_df)

for column in df_cluster.columns[1:]:
    df_cluster[column] = pd.to_numeric(df_cluster[column])
    df_cluster[column].fillna((df_cluster[column].mean()), inplace=True)


       
       

########## CLUSTERS ON PURCHASE TIME #############

x = df_cluster.iloc[:, [11,12,13]].values

from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x) # normalizing the features

### K-means ###

#choosing the right number of k through the elbow method
from sklearn.cluster import KMeans
objective_function=[]
for i in range(1,11):
    clustering=KMeans(n_clusters=i, init='k-means++')
    clustering.fit(x)
    objective_function.append(clustering.inertia_)

plt.plot(range(1,11),objective_function)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters K')
plt.ylabel('objective_function')
plt.show()

#algorithm
kmeans = KMeans(n_clusters=7, init='k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(x)
df_cluster['cluster_time_kmeans'] = y_kmeans
df_cluster_time = df_cluster[['most_frequent_time_bin', 'most_frequent_weekday_bin',
       'most_frequent_month_bin','cluster_time_kmeans']]

from sklearn import metrics
#silhouette score -> it requires time to run
#print(metrics.silhouette_score(x, y_kmeans, metric='euclidean'))
# it is 0.46956759237868867

#mean analysis of clusters per variable
mean_clusters_time = df_cluster_time.groupby("cluster_time_kmeans").mean()

#visualizing the results
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
fig = px.scatter_3d(df_cluster_time, x="most_frequent_time_bin", y="most_frequent_weekday_bin", z="most_frequent_month_bin",
                    color="cluster_time_kmeans", opacity=0.8)
fig.show()



### GAUSSIAN MIXTURE ####
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=6)
labels = gmm.fit_predict(x)

# silhouette score
#print(metrics.silhouette_score(x, labels, metric='euclidean'))
# 0.3481903982465212 for n_components=7
# 0.4573092030018828 for n_components=6
# 0.3834364509372617 for n_components=5


df_cluster_time['cluster_time_GM'] = labels

# store the clusters to compute the Dunn's index
from jqmcvi import base

clus0 = df_cluster_time[df_cluster_time['cluster_time_GM']== 0]
clus1 = df_cluster_time[df_cluster_time['cluster_time_GM']== 1]
clus2 = df_cluster_time[df_cluster_time['cluster_time_GM']== 2]
cluster_list = [clus0.values, clus1.values, clus2.values]

print(base.dunn(cluster_list))


#mean analysis of clusters per variable
mean_clusters_GM = df_cluster_time.groupby("cluster_time_GM").mean()

#visualizing the results
fig = px.scatter_3d(df_cluster_time, x="most_frequent_time_bin", y="most_frequent_weekday_bin", z="most_frequent_month_bin",
                    color="cluster_time_GM", opacity=0.8)
fig.show()

