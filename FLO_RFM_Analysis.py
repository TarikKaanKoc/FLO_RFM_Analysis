#########################################
# FLO RFM ANALYSİS
#########################################


# -------------------------------------------------------------------------------------------------------------------


####################
# English
####################

# Bussines Problem

""""
Segmenting the customers of FLO, an online shoe store,
wants to make sense according to these segments.
It will be designed accordingly and will be created according to this particular clustering.
FLO, Wants to determine marketing strategies according to these segments.
"""

# Features:

# Total Features : 12
# Total Row : 19.945
# CSV File Size : 2.7 MB

""""
- master_id : Unique Customer Number
- order_channel : Which channel of the shopping platform is used (Android, IOS, Desktop, Mobile)
- last_order_channel : The channel where the most recent purchase was made
- first_order_date : Date of the customer's first purchase
- last_order_channel : Customer's previous shopping history
- last_order_date_offline : The date of the last purchase made by the customer on the offline platform
- order_num_total_ever_online : Total number of purchases made by the customer on the online platform
- order_num_total_ever_offline : Total number of purchases made by the customer on the offline platform
- customer_value_total_ever_offline : Total fees paid for the customer's offline purchases
- customer_value_total_ever_online :  Total fees paid for the customer's online purchases
- interested_in_categories_12 : List of categories the customer has shopped in the last 12 months
"""

# -------------------------------------------------------------------------------------------------------------------

####################
# Turkish
####################

# İş Problemi

""""
Online ayakkabı mağzası olan FLO Müşterilerini segmentlere ayırıp,
bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardak öbeklenmelere göre gruplar oluşturulacak.
"""

# Features:

# Toplam Değişkenler : 12
# Toplam Gözlem Birimi : 19.945
# CSV Dosya Boyutu : 2.7 MB

""""
- master_id : Eşsiz Müşteri Numarası
- order_channel : Source- Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, İOS, Desktop, Mobil)
- last_order_channel : En son alışverişin yapıldığı kanal
- first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
- last_order_channel : Müşterinin yaptığı son alışveriş tarihi
- last_order_date_offline : Müşterinin offline platformda yaptığı son alışveriş tarihi
- order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
- order_num_total_ever_offline : Müşterinin offline platformda yaptığı toplam alışveriş sayısı
- customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
- customer_value_total_ever_online :  Müşterinin online alışverişlerinde ödediği toplam ücret
- interested_in_categories_12 :  Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
"""

# -------------------------------------------------------------------------------------------------------------------


# Projcet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import squarify



pd.set_option('display.width', 900)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_ = pd.read_csv('flo_data_20k.csv')
df = df_.copy()

def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

def check_df(df, head=5):
    print("--------------------- Shape ---------------------")
    print(df.shape)
    print("--------------------- Types ---------------------")
    print(df.dtypes)
    print("--------------------- Head ---------------------")
    print(df.head(head))
    print("--------------------- Missing Values Analysis ---------------------")
    print(missing_values_analysis(df))
    print("--------------------- Quantiles ---------------------")
    print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df,head=10)

df.describe().T



df.interested_in_categories_12.value_counts()
"""
[AKTIFSPOR]                                     3464
[KADIN]                                         2158
[]                                              2135
[ERKEK]                                         1973
[KADIN, AKTIFSPOR]                              1352
[ERKEK, AKTIFSPOR]                              1178
[ERKEK, KADIN]                                   848
[COCUK]                                          836
"""

df.order_channel.value_counts()
""""
# order_channel 

Android App ---->  9495
Mobile  -------->  4882
Ios App -------->  2833
Desktop --------> 2735

"""

df.last_order_channel.value_counts()

""""
# last_order_channel

Android App ---->  6783
Offline -------->  6608
Mobile  -------->  3172
Ios App -------->  1696
Desktop -------->  1686
"""

df.nunique()

# For omnichannel total over {online + offline}
df['total_of_purchases'] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

# Total cost for omnichannel
df["total_expenditure"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.head()

df.info()

# ---->  first_order_date          -------------->  object
# ---->  last_order_date           -------------->  object
# ---->  last_order_date_online    -------------->  object
# ---->  last_order_date_offline   -------------->  object

# Converting the above mentioned column types from object to datetime format
convert =["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]
df[convert] = df[convert].apply(pd.to_datetime)

df.info()

# # order channel , total of purchase and total expenditure distribution
df.groupby('order_channel').agg({'total_of_purchases':'sum',
                                    'total_expenditure':'count'}).sort_values(by='total_expenditure', ascending=False)

# Top 10 shoppers $$$
df.groupby('master_id').agg({'total_expenditure': 'sum'}).\
    sort_values(by='total_expenditure', ascending=False).head(10)

""""
master_id                                        total_expenditure  

5d1c466a-****-****-****-************             45905.10
d5ef8058-****-****-****-************             36818.29
73fd19aa-****-****-****-************             33918.10
7137a5c0-****-****-****-************             31227.41
47a642fe-****-****-****-************             20706.34
a4d534a2-****-****-****-************             18443.57
d696c654-****-****-****-************             16918.57
fef57ffa-****-****-****-************             12726.10
cba59206-****-****-****-************             12282.24
fc0ce7a4-****-****-****-************             12103.15
"""

# Top 10 customers with the most orders giving *
df.groupby('master_id').agg({'total_of_purchases': 'sum'}).\
    sort_values(by='total_of_purchases', ascending=False).head(10)

""""
master_id                                         total_of_purchases  
 
5d1c466a-****-****-****-************              202.00
cba59206-****-****-****-************              131.00
a57f4302-****-****-****-************              111.00
fdbe8304-****-****-****-************               88.00
329968c6-****-****-****-************               83.00
73fd19aa-****-****-****-************               82.00
44d032ee-****-****-****-************               77.00
b27e241a-****-****-****-************               75.00
d696c654-****-****-****-************               70.00
a4d534a2-****-****-****-************               70.00
"""

def data_preparation_process(df):
    # For omnichannel total over {online + offline}
    df['total_of_purchases'] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

    # Total cost for omnichannel
    df["total_expenditure"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    # Converting the above mentioned column types from object to datetime format
    convert = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    df[convert] = df[convert].apply(pd.to_datetime)

    return df
data_preparation_process(df)

#Total spend by customers, both online and offline
df.groupby("master_id")["total_expenditure"].sum().sort_values(ascending=False).head()

# Total cost of purchase per year (one year!)
one_year = df.groupby("interested_in_categories_12")["total_expenditure"].sum().sort_values(ascending=False).reset_index().head()

plt.figure(figsize=(10,8))
sns.barplot(data=one_year,x='interested_in_categories_12',y='total_expenditure')
plt.show(block=True)

#                                            * RFM METRİCS *
# -------------------------------------------------------------------------------------------------------------------

# Last analysis date
last_tmsp = df["last_order_date"].max() #Timestamp('2021-05-30 00:00:00')

type(last_tmsp)


# Recency Date
af_date = dt.datetime(2021,7,1)
type(af_date)

rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (af_date - last_order_date.max()).days,
                                     'total_of_purchases': lambda total_of_purchases: total_of_purchases.sum(),
                                     'total_expenditure': lambda total_expenditure: total_expenditure.sum()})


rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm.describe().T

#                                            * Creating RFM SCORE *
# -------------------------------------------------------------------------------------------------------------------

# Converting RFM Scores to a Range of 1-5 without breaking the normal distribution.

rfm["recency_score"]=pd.qcut(rfm['Recency'],5,labels=[5,4,3,2,1])

rfm["frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm.head()

# Recency,Frequency,Monetary Metrics ---> RFM SCORE {(Concat)}

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

# example filter
rfm[rfm["RFM_SCORE"]=="545"].head()

#                                            * Customer Segmentation *
# -------------------------------------------------------------------------------------------------------------------

# Regex

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

seg_map

# seg_map include dataframe

rfm['Segment'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)

rfm.head()

rfm[rfm['RFM_SCORE'] == '555'].head(3)

""""
master_id                               Recency  Frequency  Monetary recency_score frequency_score monetary_score RFM_SCORE    Segment
                                                                                                                         
004d5204-****-****-****-************     57       8.00   1170.76             5               5              5       555     champions
00b3ee24-****-****-****-************     54       8.00   2027.78             5               5              5       555     champions
00cf8494-****-****-****-************     35      53.00   6275.33             5               5              5       555     champions
"""



# Examine the distribut. of segments

rfm[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["mean", "count"])


"""""
                   Recency       Frequency       Monetary      
                       mean count      mean count     mean count
Segment                                                         
about_to_sleep       144.03  1643      2.41  1643   361.65  1643
at_Risk              272.33  3152      4.47  3152   648.33  3152
cant_loose           265.16  1194     10.72  1194  1481.65  1194
champions             47.14  1920      8.97  1920  1410.71  1920
hibernating          277.43  3589      2.39  3589   362.58  3589
loyal_customers      112.56  3375      8.36  3375  1216.26  3375
need_attention       143.04   806      3.74   806   553.44   806
new_customers         47.98   673      2.00   673   344.05   673
potential_loyalists   66.87  2925      3.31  2925   533.74  2925
promising             88.69   668      2.00   668   334.15   668

"""

# Segmentation Graph
Segments = rfm['Segment'].value_counts().sort_values(ascending=False)
Segments

""""
hibernating            3589
loyal_customers        3375
at_Risk                3152
potential_loyalists    2925
champions              1920
about_to_sleep         1643
cant_loose             1194
need_attention          806
new_customers           673
promising               668
"""

fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(20, 13)

squarify.plot(sizes=Segments,
              label=['hibernating',
                     'at_Risk',
                     'cant_loose',
                     'about_to_sleep',
                     'need_attention',
                     'loyal_customers',
                     'promising',
                     'new_customers',
                     'potential_loyalists',
                     'champions'],color=["red","yellow","blue", "green","orange"],pad=True, bar_kwargs={'alpha':.75}, text_kwargs={'fontsize':15})
plt.title("FLO Customer Segmentation",fontsize=30)
plt.xlabel('Frequency', fontsize=30)
plt.ylabel('Recency', fontsize=30)
plt.show()

fig.savefig('FLO Customer Segmentation.jpeg')

#
rfm.groupby('Segment').agg({'Recency':'mean',
                            'Frequency':'mean',
                            'Monetary':'mean'}).sort_values(ascending = False
                                                            , by = 'Monetary')

"""""
Segment               Recency  Frequency  Monetary
                                          
cant_loose            265.16      10.72   1481.65
champions              47.14       8.97   1410.71
loyal_customers       112.56       8.36   1216.26
at_Risk               272.33       4.47    648.33
need_attention        143.04       3.74    553.44
potential_loyalists    66.87       3.31    533.74
hibernating           277.43       2.39    362.58
about_to_sleep        144.03       2.41    361.65
new_customers          47.98       2.00    344.05
promising              88.69       2.00    334.15
"""

#                                                       * CASES *
# -------------------------------------------------------------------------------------------------------------------

"""
CASE 1: A new women's shoe brand will be included. The target audience (champions,
loyal_customers) and women are determined as shoppers. We need access to the id numbers of these customers.
"""

SEGMENT_A = rfm[(rfm["Segment"]=="champions") | (rfm["Segment"]=="loyal_customers")]
SEGMENT_A.shape[0] # Out: 5295


SEGMENT_B = df[(df["interested_in_categories_12"]).str.contains("KADIN")] #7603
SEGMENT_B.shape[0] # Out: 7603

one_case = pd.merge(SEGMENT_A,SEGMENT_B[["interested_in_categories_12","master_id"]],on=["master_id"])

one_case.columns

one_case= one_case.drop(one_case.loc[:,'Recency':'interested_in_categories_12'].columns,axis=1)

# Turn the csv format
one_case.to_csv("one_case_customer_information_1.csv")

"""
CASE 2: A 40% discount on men's and children's products is planned. 
The target audience is (cant_loose, about_to_sleep, new_customers). We need to access the id numbers of these customers.
"""

SEGMENT_C = rfm[(rfm["Segment"]=="cant_loose") | (rfm["Segment"]=="about_to_sleep") | (rfm["Segment"]=="new_customers")]

SEGMENT_D = df[(df["interested_in_categories_12"]).str.contains("ERKEK|COCUK")]

second_case = pd.merge(SEGMENT_C,SEGMENT_D[["interested_in_categories_12","master_id"]],on=["master_id"])

second_case= second_case.drop(second_case.loc[:,'Recency':'interested_in_categories_12'].columns,axis=1)

# Turn the csv format

second_case.to_csv("second_case_customer_information_2.csv",index=False)