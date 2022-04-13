import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_geo = pd.read_csv('01.geo.csv', sep=";", encoding = 'unicode_escape')
df_customers = pd.read_csv('02.customers.csv', sep=";", encoding = 'unicode_escape')
df_sellers = pd.read_csv('03.sellers.csv', sep=";", encoding = 'unicode_escape')
df_orderstatus = pd.read_csv('04.order_status.csv', sep=";", encoding = 'unicode_escape')
df_orderitems = pd.read_csv('05.order_items.csv', sep=";", encoding = 'unicode_escape')
df_orderpayments = pd.read_csv('06.order_payments.csv', sep=";", encoding = 'unicode_escape')
df_productreviews = pd.read_csv('07.product_reviews.csv', sep=";", encoding = 'unicode_escape')
df_products = pd.read_csv('08.products.csv', sep=";", encoding = 'unicode_escape')


#### EXPLORATORY DATA ANALYSIS
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.tight_layout()
df_countplot = df_customers.drop_duplicates(subset=['customer_unique_id'])
g1 = sns.countplot(x = 'customer_autonomous_community',data = df_countplot,palette='crest_r', ax=axes[0],
                  order=df_countplot.groupby('customer_autonomous_community').size().sort_values().index[::-1])
g1.set_xticklabels(g1.get_xticklabels(), rotation=70, horizontalalignment='right')
g1.set_title("Customers' atonomous community",fontweight="bold", size=15)
g1.set_xlabel("Community")
g1.set_ylabel("Count")

df_countplot2 = df_sellers.drop_duplicates(subset=['seller_id'])
g2 = sns.countplot(x = 'seller_autonomous_community',data = df_countplot2,palette='crest_r', ax=axes[1],
                  order=df_countplot2.groupby('seller_autonomous_community').size().sort_values().index[::-1])
g2.set_xticklabels(g2.get_xticklabels(), rotation=70, horizontalalignment='right')
g2.set_title("Sellers' atonomous community",fontweight="bold", size=15)
g2.set_xlabel("Community")
g2.set_ylabel("Count")
plt.subplots_adjust(wspace=0.3, hspace=0.3)
#plt.savefig("customers_sellers_communities.png")
plt.show()
#del df_countplot
#del df_countplot2


### best and least selling product categories
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.tight_layout()
orders_products = pd.merge(df_orderitems, df_products, how='left')
df_barplot1 = orders_products['product_category_name'].value_counts().sort_values(ascending=False).reset_index()
df_barplot1.rename({'index': 'product_category_name', 'product_category_name': 'Count'}, axis=1, inplace=True)
f1 = sns.barplot(x = 'product_category_name',y = "Count",data = df_barplot1.head(10),palette='rocket', ax=axes[0])
f1.set_xticklabels(f1.get_xticklabels(), rotation=70, horizontalalignment='right')
f1.set_title("10 Best Selling Product Categories",fontweight="bold", size=14)
f1.set_xlabel("Product Category")

df_barplot2 = orders_products['product_category_name'].value_counts().sort_values(ascending=True).reset_index()
df_barplot2.rename({'index': 'product_category_name', 'product_category_name': 'Count'}, axis=1, inplace=True)
f2 = sns.barplot(x = 'product_category_name',y = "Count",data = df_barplot2.head(10),palette='rocket_r', ax=axes[1])
f2.set_xticklabels(f2.get_xticklabels(), rotation=70, horizontalalignment='right')
f2.set_title("10 Least Selling Product Categories",fontweight="bold", size=14)
f2.set_xlabel("Product Category")
plt.subplots_adjust(wspace=0.3, hspace=0.3)
#plt.savefig("best_least_products_categories.png")
plt.show()
#del df_barplot1
#del df_barplot2


#dataset.replace('-',np.NaN,inplace=True)
df_orderstatus.isnull().sum().sort_values(ascending=False)
df_orderitems.isnull().sum().sort_values(ascending=False)

'''
vc3 = df_orderstatus['customer_id'].value_counts().sort_values(ascending=False)

#dataset_customers = df_customers
#dataset_customers = dataset_customers.drop_duplicates(subset=['customer_unique_id'])
#dataset_customers = dataset_customers.drop('customer_id', axis=1)

vc = df_sellers['seller_id'].value_counts().sort_values(ascending=False)
vc1 = df_customers['customer_id'].value_counts().sort_values(ascending=False)
vc4 = df_customers['customer_unique_id'].value_counts().sort_values(ascending=False)

vc5 = df_productreviews['order_id'].value_counts().sort_values(ascending=False)
vc6 = df_products['product_id'].value_counts().sort_values(ascending=False)


l = set(df_geo['geo_city'])
l1 = set(df_customers['customer_city'])
'''

#product categories with a review < 4
vc6 = df_productreviews['product_id'].value_counts().sort_values(ascending=False).reset_index()
vc6 = vc6.rename(columns={"index":"product_id", "product_id": "n_product_bought"})
gb = df_productreviews.groupby('product_id')['review_score'].mean().reset_index()
gb = pd.merge(gb, df_products, how='left')
gb = gb.filter(['review_score', 'product_category_name' ], axis=1)
gb = gb.groupby('product_category_name')['review_score'].mean().reset_index()
rw = sns.barplot(x="product_category_name", y="review_score", data=gb[gb['review_score']<4].sort_values('review_score',ascending = False), palette= "YlGn_r")
rw.set_xticklabels(rw.get_xticklabels(), rotation=70, horizontalalignment='right')
rw.set_title("Product categories with a review score < 4",fontweight="bold", size=14)
plt.show()



#sns.histplot(data = df_orderitems, x = "price", kde=True, color = "purple", bins = 10)
#plt.show()



#money spent by customers
orderid_transaction = pd.merge(df_orderpayments, df_orderstatus, how='left')
orderid_transaction = orderid_transaction.filter(['order_id','transaction_value'], axis=1)
orderid_transaction = orderid_transaction.groupby(['order_id']).sum().reset_index()

orderid_customerid = pd.merge(orderid_transaction, df_orderstatus, how='left')
orderid_customerid = orderid_customerid.filter(['order_id','transaction_value', 'customer_id'], axis=1)
orderid_customerid = pd.merge(orderid_customerid, df_customers, how='left')
moneyspent_percustomer = orderid_customerid.filter(['customer_unique_id','transaction_value'], axis=1)
moneyspent_percustomer['transaction_value'] = [x.replace(',', '.') for x in moneyspent_percustomer['transaction_value']]
moneyspent_percustomer['transaction_value'] = moneyspent_percustomer['transaction_value'].astype(float)
moneyspent_percustomer = moneyspent_percustomer.groupby(['customer_unique_id']).sum().reset_index()



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
    df_cluster[column].fillna((df_cluster[column].median()), inplace=True)



from sklearn.preprocessing import StandardScaler
x = df_cluster.iloc[:, df_cluster.columns != "customer_unique_id"].values
x = StandardScaler().fit_transform(x) # normalizing the features


from sklearn.decomposition import PCA
pca_breast = PCA(n_components=8)
principalComponents_breast = pca_breast.fit_transform(x)
#principal_breast_Df = pd.DataFrame(data = principalComponents_breast
#             , columns = ['principal component 1', 'principal component 2'])


print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))

# >> Explained variation per principal component: [0.36198848 0.1920749 ]

print('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca_breast.explained_variance_ratio_)))



