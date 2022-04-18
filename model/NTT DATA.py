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

#mean analysis of clusters per variable
mean_clusters_GM = df_cluster_time.groupby("cluster_time_GM").mean()

#visualizing the results
fig = px.scatter_3d(df_cluster_time, x="most_frequent_time_bin", y="most_frequent_weekday_bin", z="most_frequent_month_bin",
                    color="cluster_time_GM", opacity=0.8)
fig.show()




#### DBSCAN ####

# not very performing for count data

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Tune DBSCAN "esp" parameter
def get_kdist_plot(X=None, k=None, radius_nbrs=1.0):
    nbrs = NearestNeighbors(n_neighbors=k, radius=radius_nbrs).fit(X)
    # For each point, compute distances to its k-nearest neighbors
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:, k - 1]
    # Plot the sorted K-nearest neighbor distance for each point in the dataset
    fig = plt.figure(figsize=(8, 8))
    plt.plot(distances)
    plt.xlabel('Points/Objects in the dataset', fontsize=12)
    plt.ylabel('Sorted {}-nearest neighbor distance'.format(k), fontsize=12)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    plt.show()
    return fig


k = 2 * x.shape[-1] - 1  # k=2*{dim(dataset)} - 1
get_kdist_plot(X=x, k=k)

DBSCAN_alg = DBSCAN(eps=0.1, min_samples=10)
clusters_DBSCAN = DBSCAN_alg.fit_predict(x)

df_cluster_time['cluster_time_DBSCAN'] = clusters_DBSCAN

#mean analysis of clusters per variable
mean_clusters_DBSCAN = df_cluster_time.groupby("cluster_time_DBSCAN").mean()





#CLUSTER OVER MONEY, RATE, PRODUCT COUNT #NON MOLTO INTERESSANTE
df_db = df_cluster[["costoprodotti_totale","product count", "n_medio_rate"]]
df_scale = df_db.sample(15000, random_state=42)  # random sample
df_label = df_db.sample(15000, random_state=42)

index = df_label.index  # get index
index = index.to_list()
index_sub = pd.DataFrame()
index_sub["index"] = index

scaler = StandardScaler().fit(df_scale)  # scale data for clustering
df_scale = scaler.transform(df_scale)
df_scale = pd.DataFrame(df_scale)
df_scale = df_scale.set_index(index_sub["index"])

from sklearn.cluster import DBSCAN

DBSCAN_cluster = DBSCAN(eps=0.3, min_samples=100).fit(df_scale)  # cluster

index_sub["cluster"] = DBSCAN_cluster.labels_
index_sub = index_sub.set_index("index")
df_dbscan = df_label.copy()
df_dbscan = df_dbscan.join(index_sub)

df_segm_analysis = df_dbscan.groupby(['cluster']).mean()  # means of the cluster based on the variables used
df_dbscan.value_counts(["cluster"])

# VISUALIZE
pio.renderers.default = 'browser'  # set pre-definite browser as palce were to render the image
fig = px.scatter_3d(df_dbscan,
                 x="costoprodotti_totale",
                 y="n_medio_rate",
                 z="product count",
                 color="cluster", opacity=0.8)
fig.show()

#CLUSTER | PRODUCTS CATEGORY/BEHAVIOUR | PCA | KMEANS #_____INTERESSANTE_______
df_segmentation = df_cluster.copy()
df_segmentation = df_segmentation[['customer_unique_id', "product count", "costoprodotti_totale",
                                   "n_medio_pagamenti", "n_medio_rate",
                                   'Home', 'Forniture', 'Technology', 'Fashion/Personal care', 'Hobbies']]

list_cols = ["product count", "costoprodotti_totale",
             "n_medio_pagamenti", "n_medio_rate", 'Home', 'Forniture', 'Technology',
             'Fashion/Personal care', 'Hobbies']

matrix = df_segmentation[list_cols].to_numpy()
df_segmentation = pd.DataFrame(df_segmentation)
#SCALE
scaler = StandardScaler()
scaler.fit(matrix)
scaled_matrix = scaler.transform(matrix)

#PCA
pca = PCA()
pca.fit(scaled_matrix)
pca_samples = pca.transform(scaled_matrix)

#VISUALIZE PCA WITH CUMULATIVE SUM
fig, ax = plt.subplots(figsize=(14, 5))
sns.set(font_scale=1)
plt.step(range(matrix.shape[1]), pca.explained_variance_ratio_.cumsum(), where='mid',
         label='cumulative explained variance')
sns.barplot(np.arange(1, matrix.shape[1] + 1), pca.explained_variance_ratio_, alpha=0.5, color='g',
            label='individual explained variance')
plt.xlim(0, 10)

ax.set_xticklabels([s if int(s.get_text()) % 2 == 0 else '' for s in ax.get_xticklabels()])
plt.ylabel('Explained variance', fontsize=14)
plt.xlabel('Principal components', fontsize=14)
plt.legend(loc='best', fontsize=13)
plt.show()

#FIND NUMBER OF CLUSTERS | ELBOW
objective_function = []
for i in range(1, 20):
    clustering = KMeans(n_clusters=i, init='k-means++')
    clustering.fit(scaled_matrix)
    objective_function.append(clustering.inertia_)

plt.plot(range(1, 20), objective_function)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters K')
plt.ylabel('objective_function')
plt.show()

#KMEANS
n_clusters = 12
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
kmeans.fit(scaled_matrix)
clusters_clients = kmeans.predict(scaled_matrix)

#PCA
pca = PCA(n_components=5)
matrix_3D = pca.fit_transform(scaled_matrix)
mat = pd.DataFrame(matrix_3D)
mat['cluster'] = pd.Series(clusters_clients)

#VISUALIZE PCA COMPONENTS
import matplotlib.patches as mpatches

sns.set_style("white")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

LABEL_COLOR_MAP = {0: 'r', 1: 'tan', 2: 'b', 3: 'k', 4: 'c', 5: 'g', 6: 'deeppink', 7: 'skyblue', 8: 'darkcyan',
                   9: 'orange',
                   10: 'yellow', 11: 'tomato', 12: 'seagreen'}
label_color = [LABEL_COLOR_MAP[l] for l in mat['cluster']]

fig = plt.figure(figsize=(20, 15))
increment = 0
for ix in range(5):
    for iy in range(ix + 1, 5):
        increment += 1
        ax = fig.add_subplot(5, 2, increment)
        ax.scatter(mat[ix], mat[iy], c=label_color, alpha=0.5)
        plt.ylabel('PCA {}'.format(iy + 1), fontsize=12)
        plt.xlabel('PCA {}'.format(ix + 1), fontsize=12)
        ax.yaxis.grid(color='lightgray', linestyle=':')
        ax.xaxis.grid(color='lightgray', linestyle=':')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if increment == 10: break
    if increment == 10: break

    # _______________________________________________
    # I set the legend: abreviation -> airline name
comp_handler = []
for i in range(n_clusters):
    comp_handler.append(mpatches.Patch(color=LABEL_COLOR_MAP[i], label=i))

plt.legend(handles=comp_handler, bbox_to_anchor=(1.1, 0.9),
           title='Cluster', facecolor='lightgrey',
           shadow=True, frameon=True, framealpha=1,
           fontsize=13, bbox_transform=plt.gcf().transFigure)

plt.tight_layout()
plt.show()

df_segmentation.loc[:, "cluster"] = clusters_clients #add column of clusters to the dataframe with whole values
#BUILD DATASET
merged_df = pd.DataFrame()
for i in range(n_clusters):
    test = pd.DataFrame(df_segmentation[df_segmentation['cluster'] == i].mean())
    test = test.T.set_index('cluster', drop=True)
    test['size'] = df_segmentation[df_segmentation['cluster'] == i].shape[0] #ADD SIZE COLUMN
    merged_df = pd.concat([merged_df, test])
# _____________________________________________________
#merged_df.drop('customer_unique_id', axis=1, inplace=True)
#print('number of customers:', merged_df['size'].sum())
merged_df = merged_df.sort_values('costoprodotti_totale') # <--- WHERE TO LOOK FOR THE CLUSTERS
merged_df_scaled = scaler.fit_transform(merged_df) # <-- scaled values to make radars plots

#VISUALIZIATION RADAR

#NORMAL PEOPLE
pio.renderers.default = 'browser'
categories = ['product count', 'costoprodotti_totale', 'n_medio_pagamenti',
       'n_medio_rate', 'Home', 'Forniture', 'Technology',
       'Fashion/Personal care', 'Hobbies']
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
      r=[-0.64, -0.55, -0.30, -0.75, -0.57, -0.42, 0.99, -0.53, -0.50],
      theta=categories,
      #fill='toself',
      name='Cluster 4'
))
fig.add_trace(go.Scatterpolar(
      r=[-0.72, -0.54, -0.29, -0.61, 0.99, -0.42, -0.42, -0.54, -0.51],
      theta=categories,
      #fill='toself',
      name='CLuster 2'
))

fig.add_trace(go.Scatterpolar(
      r=[-.72, -.5, -0.29, -.64, -.56, -0.42, -.42, -.54, 0.98],
      theta=categories,
      #fill='toself',
      name='CLuster 7'
))
fig.add_trace(go.Scatterpolar(
      r=[-.60, -.49, -.29, -0.58, -0.55, 0.98, -.41, -.53, -0.49],
      theta=categories,
      #fill='toself',
      name='CLuster 3'
))
fig.add_trace(go.Scatterpolar(
      r=[-.50, -.46, -.29, -0.46, -0.56, -.42, -.41, 0.99, -0.49],
      theta=categories,
      #fill='toself',
      name='CLuster 1'
))
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[-1, 1]
    )),
  showlegend=True,
)
fig.update_traces(marker_opacity=0.1, fill="toself")
fig.show()

#SECOND RADAR / HIGH N_PAGAMENTI
from plotly.subplots import make_subplots
fig1 = go.Figure()
fig1.add_trace(go.Scatterpolar(
      r=[-.57, -.37, 3.3, -1, 0.09, -.24, -.33, 0.1, -0.14],
      theta=categories,
      fill='toself',
      name='CLuster payment_num | 10'
))
fig1.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[-1, 3]
    )),
  showlegend=True,
)
fig1.update_traces(marker_opacity=0.1, fill="toself")
fig1.show()

#THIRD MULTI RADAR
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'
fig2 = make_subplots(1,2)
fig2.add_trace(go.Scatterpolar(
      r=[0.55162, -0.28559,-0.30448, -0.17567, -0.54336, -0.41206, -0.41206, -0.47670, 3.03807],
      theta=categories,
      fill='toself',
      name='CLuster product count | 6'
))
fig2.add_trace(go.Scatterpolar(
      r=[0.60023, -0.27,-0.29448, 0.08, 3.05, -0.40206, -0.41, -0.45670, -0.48],
      theta=categories,
      fill='toself',
      name='CLuster product count | 0'
))

fig2.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[-1, 3]
    )),
  showlegend=True,
)
fig2.update_traces(marker_opacity=0.1, fill="toself")
fig2.show()

#FOURTH MULTIRADAR / MOST VALUABLE CUSTOMERS
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'
fig3 = make_subplots(1,2)
fig3.add_trace(go.Scatterpolar(
      r=[-0.70, -0.11,-0.31448, 2.54567, -0.038, -0.30, -0.35, -0.33670, -0.26],
      theta=categories,
      fill='toself',
      name='CLuster rate alto | 11'
))
fig3.add_trace(go.Scatterpolar(
      r=[1.97, 0.07,-0.30, 0.37, -0.32, 3.17206, -0.4, -0.25670, -0.44],
      theta=categories,
      fill='toself',
      name='CLuster product count | 5'
))

fig3.add_trace(go.Scatterpolar(
      r=[2.02, 0.32,-0.30, -.12, -0.55, -.42, 3.19, -0.50, -0.48],
      theta=categories,
      fill='toself',
      name='CLuster product count | 9'
))

fig3.add_trace(go.Scatterpolar(
      r=[-0.57, 3.20,-0.29, 1.47, -0.22, -.32206, -0.15, 0.35670, -0.17515],
      theta=categories,
      fill='toself',
      name='CLuster spesa | 8'
))

fig3.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[-1, 3]
    )),
  showlegend=True,
)
fig3.update_traces(marker_opacity=0.1, fill="toself")
fig3.show()

#####CLUSTER CUSTOMER BEHAVIOUR | KMEANS #_____INTERESSANTE______
df_segmentation = df_cluster[['customer_unique_id',"order_count (only positive orders)","product count",
                              "costoprodotti_totale","n_medio_pagamenti", "n_medio_rate"]]

df_segmentation= df_segmentation.drop(['customer_unique_id'], axis=1)

df_seg_scale = df_segmentation.copy()
scaler = StandardScaler()
df_seg_scale= scaler.fit_transform(df_seg_scale)

objective_function = []
for i in range(1, 20):
    clustering = KMeans(n_clusters=i, init='k-means++')
    clustering.fit(df_seg_scale)
    objective_function.append(clustering.inertia_)
plt.plot(range(1, 20), objective_function)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters K')
plt.ylabel('objective_function')
plt.show()

kmeans = KMeans(n_clusters=12, init='k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(df_seg_scale)
df_segmentation['cluster_kmeans'] = y_kmeans
df_RFM = df_segmentation.groupby(['cluster_kmeans']).mean() # <---- DATASET WITH CLUSTER AND MEAN VALUES
df_segmentation.value_counts(["cluster_kmeans"])

#RECENCY FREQUECY MONEY (RFM) #_____INTERESSANTE_____
df_db = df_cluster[["costoprodotti_totale","order_count (only positive orders)", "recency"]] #"n_medio_rate"
df_scale = df_db.copy()
df_label = df_db.copy()

index = df_label.index  # get index
index = index.to_list()
index_sub = pd.DataFrame()
index_sub["index"] = index

scaler = StandardScaler().fit(df_scale)  # scale data for clustering
df_scale = scaler.transform(df_scale)
df_scale = pd.DataFrame(df_scale)
df_scale = df_scale.set_index(index_sub["index"])

#KMEANS
from sklearn.cluster import KMeans
objective_function=[]
for i in range(1,30):
    clustering=KMeans(n_clusters=i, init='k-means++')
    clustering.fit(df_scale)
    objective_function.append(clustering.inertia_)

plt.plot(range(1,30),objective_function)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters K')
plt.ylabel('objective_function')
plt.show()

kmeans = KMeans(n_clusters=10, init='k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(df_scale)
df_label['cluster_kmeans'] = y_kmeans
df_RFM = df_label.groupby(['cluster_kmeans']).mean() #<<-- DATA FRAME CON I CLUSTER
df_label.value_counts(["cluster_kmeans"])

##PCA + VISUALIZATION
df_seg_scale = pd.DataFrame(df_seg_scale)
df_seg_scale.columns = ["order_count (only positive orders)","product count",
                              "costoprodotti_totale","n_medio_pagamenti", "n_medio_rate"]
df_seg_scale["cluster"] = y_kmeans

pca = PCA(n_components=3)
pc_3 = pd.DataFrame(pca.fit_transform(df_seg_scale.drop(["cluster"],axis=1)))
pc_3.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]
pc_3["cluster"] = y_kmeans

pio.renderers.default = 'browser'  # set pre-definite browser as palce were to render the image
fig = px.scatter_3d(pc_3,
                 x="PC1_3d",
                 y="PC2_3d",
                 z="PC3_3d",
                 color="cluster", opacity=0.8)
fig.show()

