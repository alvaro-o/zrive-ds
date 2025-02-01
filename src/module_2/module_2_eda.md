```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

*1. Understanding the problem space*

**orders**


```python
orders = pd.read_parquet('/home/alvaro/groceries/orders.parquet')
orders.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 8773 entries, 10 to 64538
    Data columns (total 6 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   id              8773 non-null   int64         
     1   user_id         8773 non-null   object        
     2   created_at      8773 non-null   datetime64[us]
     3   order_date      8773 non-null   datetime64[us]
     4   user_order_seq  8773 non-null   int64         
     5   ordered_items   8773 non-null   object        
    dtypes: datetime64[us](2), int64(2), object(2)
    memory usage: 479.8+ KB



```python
orders.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618849693828, 33618860179588, 3361887404045...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618835243140, 33618835964036, 3361886244058...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2204838822020</td>
      <td>329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700...</td>
      <td>2020-04-30 18:12:30</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618891145348, 33618893570180, 3361889766618...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2208967852164</td>
      <td>f6451fce7b1c58d0effbe37fcb4e67b718193562766470...</td>
      <td>2020-05-01 19:44:11</td>
      <td>2020-05-01</td>
      <td>1</td>
      <td>[33618830196868, 33618846580868, 3361891234624...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>[33667166699652, 33667166699652, 3366717122163...</td>
    </tr>
  </tbody>
</table>
</div>



El índice del dataset se salta números y no se corresponde con el número de filas por alguna razón


```python
orders.user_order_seq.describe()
```




    count    8773.000000
    mean        2.445116
    std         2.707693
    min         1.000000
    25%         1.000000
    50%         1.000000
    75%         3.000000
    max        25.000000
    Name: user_order_seq, dtype: float64




```python
orders.user_order_seq.hist(bins=25)
```




    <Axes: >




    
![png](module_2_eda_files/module_2_eda_7_1.png)
    


La mayoría de usuarios compran una vez, y el numero de usuarios va disminuyendo según el número de compras por usuario aumenta


```python
orders.order_date.hist(bins=23)
```




    <Axes: >




    
![png](module_2_eda_files/module_2_eda_9_1.png)
    


El número de órdenes aumenta a medida que avanza el tiempo


```python
# crear una columna con el número de elementos en cada compra
orders['ordered_items_count'] = orders['ordered_items'].apply(lambda x: len(x))
orders.ordered_items_count.describe()
```




    count    8773.000000
    mean       12.305711
    std         6.839507
    min         1.000000
    25%         8.000000
    50%        11.000000
    75%        15.000000
    max       114.000000
    Name: ordered_items_count, dtype: float64




```python
orders.ordered_items_count.hist(bins=114)
```




    <Axes: >




    
![png](module_2_eda_files/module_2_eda_12_1.png)
    


La mayoría de órdenes contienen alrededor de 15 elementos, con una cola larga en la distribución que llega a 114


```python
monthly_items = orders.groupby(orders['order_date'].dt.to_period('M'))['ordered_items_count'].mean().reset_index()
monthly_items['order_date'] = monthly_items['order_date'].dt.to_timestamp()
plt.plot(monthly_items['order_date'],monthly_items['ordered_items_count'])
```




    [<matplotlib.lines.Line2D at 0x7f42221b7990>]




    
![png](module_2_eda_files/module_2_eda_14_1.png)
    


El número de elementos por cada compra baja con el tiempo

**regulars**


```python
regulars = pd.read_parquet('/home/alvaro/groceries/regulars.parquet')
regulars.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 18105 entries, 3 to 37720
    Data columns (total 3 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   user_id     18105 non-null  object        
     1   variant_id  18105 non-null  int64         
     2   created_at  18105 non-null  datetime64[us]
    dtypes: datetime64[us](1), int64(1), object(1)
    memory usage: 565.8+ KB



```python
regulars.created_at.hist(bins=23)
```




    <Axes: >




    
![png](module_2_eda_files/module_2_eda_18_1.png)
    


La evolución de compras regulares se corresponde con la distribución de compras en general

**abandoned_carts**


```python
abandoned = pd.read_parquet('/home/alvaro/groceries/abandoned_carts.parquet')
abandoned.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 5457 entries, 0 to 70050
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   id          5457 non-null   int64         
     1   user_id     5457 non-null   object        
     2   created_at  5457 non-null   datetime64[us]
     3   variant_id  5457 non-null   object        
    dtypes: datetime64[us](1), int64(1), object(2)
    memory usage: 213.2+ KB



```python
abandoned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>variant_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12858560217220</td>
      <td>5c4e5953f13ddc3bc9659a3453356155e5efe4739d7a2b...</td>
      <td>2020-05-20 13:53:24</td>
      <td>[33826459287684, 33826457616516, 3366719212762...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20352449839236</td>
      <td>9d6187545c005d39e44d0456d87790db18611d7c7379bd...</td>
      <td>2021-06-27 05:24:13</td>
      <td>[34415988179076, 34037940158596, 3450282236326...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>20478401413252</td>
      <td>e83fb0273d70c37a2968fee107113698fd4f389c442c0b...</td>
      <td>2021-07-18 08:23:49</td>
      <td>[34543001337988, 34037939372164, 3411360609088...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>20481783103620</td>
      <td>10c42e10e530284b7c7c50f3a23a98726d5747b8128084...</td>
      <td>2021-07-18 21:29:36</td>
      <td>[33667268116612, 34037940224132, 3443605520397...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>20485321687172</td>
      <td>d9989439524b3f6fc4f41686d043f315fb408b954d6153...</td>
      <td>2021-07-19 12:17:05</td>
      <td>[33667268083844, 34284950454404, 33973246886020]</td>
    </tr>
  </tbody>
</table>
</div>




```python
abandoned['num_abandoned'] = abandoned['variant_id'].apply(lambda x: len(x))
abandoned.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>variant_id</th>
      <th>num_abandoned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12858560217220</td>
      <td>5c4e5953f13ddc3bc9659a3453356155e5efe4739d7a2b...</td>
      <td>2020-05-20 13:53:24</td>
      <td>[33826459287684, 33826457616516, 3366719212762...</td>
      <td>19</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20352449839236</td>
      <td>9d6187545c005d39e44d0456d87790db18611d7c7379bd...</td>
      <td>2021-06-27 05:24:13</td>
      <td>[34415988179076, 34037940158596, 3450282236326...</td>
      <td>9</td>
    </tr>
    <tr>
      <th>45</th>
      <td>20478401413252</td>
      <td>e83fb0273d70c37a2968fee107113698fd4f389c442c0b...</td>
      <td>2021-07-18 08:23:49</td>
      <td>[34543001337988, 34037939372164, 3411360609088...</td>
      <td>20</td>
    </tr>
    <tr>
      <th>50</th>
      <td>20481783103620</td>
      <td>10c42e10e530284b7c7c50f3a23a98726d5747b8128084...</td>
      <td>2021-07-18 21:29:36</td>
      <td>[33667268116612, 34037940224132, 3443605520397...</td>
      <td>13</td>
    </tr>
    <tr>
      <th>52</th>
      <td>20485321687172</td>
      <td>d9989439524b3f6fc4f41686d043f315fb408b954d6153...</td>
      <td>2021-07-19 12:17:05</td>
      <td>[33667268083844, 34284950454404, 33973246886020]</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
monthly_abandoned = abandoned.groupby(abandoned['created_at'].dt.to_period('M'))['num_abandoned'].mean().reset_index()
monthly_abandoned['created_at'] = monthly_abandoned['created_at'].dt.to_timestamp()
plt.plot(monthly_abandoned['created_at'],monthly_abandoned['num_abandoned'])
```




    [<matplotlib.lines.Line2D at 0x7f4222189cd0>]




    
![png](module_2_eda_files/module_2_eda_24_1.png)
    


El número de elementos que se abandonan en cada carrito ha disminuido con el tiempo al igual que el número de elementos comprados

**users**


```python
users = pd.read_parquet('/home/alvaro/groceries/users.parquet')
users.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 4983 entries, 2160 to 3360
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   user_id                4983 non-null   object 
     1   user_segment           4983 non-null   object 
     2   user_nuts1             4932 non-null   object 
     3   first_ordered_at       4983 non-null   object 
     4   customer_cohort_month  4983 non-null   object 
     5   count_people           325 non-null    float64
     6   count_adults           325 non-null    float64
     7   count_children         325 non-null    float64
     8   count_babies           325 non-null    float64
     9   count_pets             325 non-null    float64
    dtypes: float64(5), object(5)
    memory usage: 428.2+ KB



```python
users.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.787692</td>
      <td>2.003077</td>
      <td>0.707692</td>
      <td>0.076923</td>
      <td>0.636923</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.365753</td>
      <td>0.869577</td>
      <td>1.026246</td>
      <td>0.289086</td>
      <td>0.995603</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
users.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2160</th>
      <td>0e823a42e107461379e5b5613b7aa00537a72e1b0eaa7a...</td>
      <td>Top Up</td>
      <td>UKH</td>
      <td>2021-05-08 13:33:49</td>
      <td>2021-05-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1123</th>
      <td>15768ced9bed648f745a7aa566a8895f7a73b9a47c1d4f...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-17 16:30:20</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>33e0cb6eacea0775e34adbaa2c1dec16b9d6484e6b9324...</td>
      <td>Top Up</td>
      <td>UKD</td>
      <td>2022-03-09 23:12:25</td>
      <td>2022-03-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>675</th>
      <td>57ca7591dc79825df0cecc4836a58e6062454555c86c35...</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2021-04-23 16:29:02</td>
      <td>2021-04-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4694</th>
      <td>085d8e598139ce6fc9f75d9de97960fa9e1457b409ec00...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-02 13:50:06</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



user_nuts1 (la región) tiene varios valores nan que podrían imputarse por los valores más comunes

Los counts de clientes tienen muchos valores nan. Podría tener sentido imputarlos por 0 o crear una columna que indique si los clientes han sido contados o no


```python
# Cantidad de usuarios de cada región
nuts_1_notna=users.user_nuts1.value_counts()
nuts_1_notna
```




    user_nuts1
    UKI    1318
    UKJ     745
    UKK     602
    UKH     414
    UKD     358
    UKM     315
    UKE     303
    UKG     295
    UKF     252
    UKL     224
    UKC     102
    UKN       4
    Name: count, dtype: int64




```python
# imputar valores nulos
users['user_nuts1'] = users['user_nuts1'].fillna('UKI')
```


```python
# calcular la proporción de valores nulos para count_people cada mes
proporcion_nan = users.groupby('customer_cohort_month')['count_people'].apply(lambda x: x.isna().mean()).reset_index()
print(proporcion_nan)
```

       customer_cohort_month  count_people
    0    2020-04-01 00:00:00      0.000000
    1    2020-05-01 00:00:00      0.181818
    2    2020-06-01 00:00:00      0.028571
    3    2020-07-01 00:00:00      0.023810
    4    2020-08-01 00:00:00      0.315789
    5    2020-09-01 00:00:00      0.838235
    6    2020-10-01 00:00:00      0.965909
    7    2020-11-01 00:00:00      0.928571
    8    2020-12-01 00:00:00      0.981651
    9    2021-01-01 00:00:00      0.996324
    10   2021-02-01 00:00:00      0.995455
    11   2021-03-01 00:00:00      0.975610
    12   2021-04-01 00:00:00      0.829268
    13   2021-05-01 00:00:00      0.910180
    14   2021-06-01 00:00:00      0.902703
    15   2021-07-01 00:00:00      0.938462
    16   2021-08-01 00:00:00      0.959677
    17   2021-09-01 00:00:00      0.944444
    18   2021-10-01 00:00:00      0.962525
    19   2021-11-01 00:00:00      0.973396
    20   2021-12-01 00:00:00      0.975771
    21   2022-01-01 00:00:00      0.968023
    22   2022-02-01 00:00:00      0.969697
    23   2022-03-01 00:00:00      0.972789



```python
proporcion_nan['customer_cohort_month'] = pd.to_datetime(proporcion_nan['customer_cohort_month'])
plt.plot(proporcion_nan.customer_cohort_month.dt.strftime('%Y-%m'),proporcion_nan.count_people)
```




    [<matplotlib.lines.Line2D at 0x7f42221b6410>]




    
![png](module_2_eda_files/module_2_eda_34_1.png)
    


Conforme avanza el tiempo cuentan menos a los clientes


```python
users = users.assign(people_isna = users.count_people.isna())
orders_counted_users = orders.merge(users[['user_id','people_isna']], on='user_id',how='left')
```


```python
orders_counted_users.groupby('people_isna')['user_order_seq'].mean().plot(kind='bar')
```




    <Axes: xlabel='people_isna'>




    
![png](module_2_eda_files/module_2_eda_37_1.png)
    


Los usuarios que han sido contados han comprado más veces de media


```python
users[['count_people','count_adults','count_children','count_babies','count_pets']].mean().plot(kind='bar')
```




    <Axes: >




    
![png](module_2_eda_files/module_2_eda_39_1.png)
    



```python
users.user_segment.unique()
```




    array(['Top Up', 'Proposition'], dtype=object)




```python
users.user_segment.value_counts()
```




    user_segment
    Top Up         2643
    Proposition    2340
    Name: count, dtype: int64




```python
users.groupby('customer_cohort_month').user_segment.value_counts().unstack().plot(kind='line')
```




    <Axes: xlabel='customer_cohort_month'>




    
![png](module_2_eda_files/module_2_eda_42_1.png)
    



```python
users.groupby('user_nuts1').user_segment.value_counts().unstack().plot(kind='bar')

```




    <Axes: xlabel='user_nuts1'>




    
![png](module_2_eda_files/module_2_eda_43_1.png)
    


Hay leves diferencias por región entre los segmentos 'proposition' y 'top up'


```python
n_regulars = regulars.groupby('user_id').variant_id.nunique().rename('n_regulars')
users_regulars = users.merge(n_regulars, on='user_id',how='left').fillna(0)
```


```python
users_regulars.groupby('user_segment').n_regulars.mean().plot(kind='bar')
```




    <Axes: xlabel='user_segment'>




    
![png](module_2_eda_files/module_2_eda_46_1.png)
    


Los usuarios 'proposition' tienen un número bastante mayor de regulars

**inventory**


```python
inventory = pd.read_parquet('/home/alvaro/groceries/inventory.parquet')
inventory.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1733 entries, 0 to 1732
    Data columns (total 6 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   variant_id        1733 non-null   int64  
     1   price             1733 non-null   float64
     2   compare_at_price  1733 non-null   float64
     3   vendor            1733 non-null   object 
     4   product_type      1733 non-null   object 
     5   tags              1733 non-null   object 
    dtypes: float64(2), int64(1), object(3)
    memory usage: 81.4+ KB



```python
inventory.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34370361229444</td>
      <td>4.99</td>
      <td>5.50</td>
      <td>whogivesacrap</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, eco, toilet-rolls]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34284951863428</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[kitchen-roll]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33667283583108</td>
      <td>1.79</td>
      <td>1.99</td>
      <td>thecheekypanda</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, cruelty-free, eco, tissue, vegan]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33803537973380</td>
      <td>1.99</td>
      <td>2.09</td>
      <td>colgate</td>
      <td>dental</td>
      <td>[dental-accessories]</td>
    </tr>
  </tbody>
</table>
</div>




```python
ordered_items = orders.explode('ordered_items').rename({'ordered_items':'variant_id'}, axis=1)
ordered_items = ordered_items.merge(inventory,on='variant_id',how='left')
```


```python
ordered_items.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 107958 entries, 0 to 107957
    Data columns (total 12 columns):
     #   Column               Non-Null Count   Dtype         
    ---  ------               --------------   -----         
     0   id                   107958 non-null  int64         
     1   user_id              107958 non-null  object        
     2   created_at           107958 non-null  datetime64[us]
     3   order_date           107958 non-null  datetime64[us]
     4   user_order_seq       107958 non-null  int64         
     5   variant_id           107958 non-null  object        
     6   ordered_items_count  107958 non-null  int64         
     7   price                92361 non-null   float64       
     8   compare_at_price     92361 non-null   float64       
     9   vendor               92361 non-null   object        
     10  product_type         92361 non-null   object        
     11  tags                 92361 non-null   object        
    dtypes: datetime64[us](2), float64(2), int64(3), object(5)
    memory usage: 9.9+ MB



```python
inventory_orders = ordered_items.loc[ordered_items.price.notna(),['variant_id','order_date']]
inventory_monthly_items = inventory_orders.groupby(inventory_orders['order_date'].dt.to_period('M')).variant_id.count().reset_index()
inventory_monthly_items['order_date'] = inventory_monthly_items['order_date'].dt.to_timestamp()
```


```python
monthly_items = ordered_items.groupby(ordered_items['order_date'].dt.to_period('M'))['variant_id'].count().reset_index()
monthly_items['order_date'] = monthly_items['order_date'].dt.to_timestamp()

plt.plot(monthly_items['order_date'],monthly_items['variant_id'],label='total products sold')
plt.plot(inventory_monthly_items['order_date'],inventory_monthly_items['variant_id'],label='total products sold listed in inventory')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f41f4c3e190>




    
![png](module_2_eda_files/module_2_eda_54_1.png)
    


Varios productos fueron pedidos pero no aparecen en el inventario. Se debe a que el inventario sólo incluye los productos que actualmente se venden por lo que algunos productos fueron deslistados en algún momento. Esto puede ser un obstáculo para analizar el dataset


```python
ordered_items.groupby('id').price.sum().describe()
```




    count    8773.000000
    mean       55.832173
    std        24.316070
    min         0.000000
    25%        44.250000
    50%        51.860000
    75%        60.590000
    max       319.800000
    Name: price, dtype: float64




```python
orders_size = ordered_items.groupby(['id','order_date'])['price'].sum().reset_index()
orders_size.groupby(pd.Grouper(key='order_date',freq='W'))['price'].mean().plot()
```




    <Axes: xlabel='order_date'>




    
![png](module_2_eda_files/module_2_eda_57_1.png)
    


El precio medio de cada orden ha subido, aunque no se está teniendo en cuenta el precio de los productos que han sido deslistados del inventario, por lo que esto podría no ser así. El precio medio por compra es de unos 50 euros


```python
top_types = ordered_items.groupby('product_type').id.count().sort_values(ascending=False).head(10)
top_types.plot(kind='bar')
```




    <Axes: xlabel='product_type'>




    
![png](module_2_eda_files/module_2_eda_59_1.png)
    



```python
inventory.groupby('product_type').variant_id.count().sort_values(ascending=False).head(10).plot(kind='bar')
```




    <Axes: xlabel='product_type'>




    
![png](module_2_eda_files/module_2_eda_60_1.png)
    



```python
top_products = ordered_items.groupby('variant_id').id.count().sort_values(ascending=False).head()
top_products
```




    variant_id
    34081589887108    4487
    39284117930116    2658
    34137590366340    1459
    34081331970180    1170
    34284951863428    1133
    Name: id, dtype: int64




```python
inventory.loc[inventory.variant_id.isin(top_products.index),:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>34284951863428</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[kitchen-roll]</td>
    </tr>
    <tr>
      <th>63</th>
      <td>34081589887108</td>
      <td>10.79</td>
      <td>11.94</td>
      <td>oatly</td>
      <td>long-life-milk-substitutes</td>
      <td>[oat-milk, vegan]</td>
    </tr>
  </tbody>
</table>
</div>



Diversos productos de limpieza, la comida enlatada y los sustitutos lácteos son los tipos de productos más vendidos. Tres de los cinco productos más vendidos ya no están en el inventario


```python
def compute_cohort_stats(x):
    cohort_size = x.loc[lambda x: x.order_month_diff == 0, 'user_id'].nunique()
    return (x.groupby('order_month_diff')['user_id'].nunique() / cohort_size).rename(
        'retention_rate'
    )

fig, ax = plt.subplots(figsize=[15,5])

retention_curves = (
    orders.assign(
        first_order_month = lambda x: x.groupby('user_id')['order_date']
        .transform('min')
        .dt.to_period('M')
    )
    .assign(order_month=lambda x: x.order_date.dt.to_period('M'))
    .assign(
        order_month_diff=lambda x: (x.order_month - x.first_order_month).apply(
            lambda x: x.n
        )
    )
    .groupby('first_order_month')
    .apply(compute_cohort_stats)
    .reset_index()
)

colors = plt.cm.Blues(
    np.linspace(0.1, 0.9, retention_curves.first_order_month.nunique())
)
count = 0
for label, df in retention_curves.groupby('first_order_month'):
    df.loc[lambda x: x.order_month_diff < 12].plot(
        x='order_month_diff',
        y='retention_rate',
        ax=ax,
        label=label,
        color=colors[count]
    )
    count+=1

plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.title('One year retention curves by monthly cohort')
plt.ylabel('Retention rate')
```

    /tmp/ipykernel_73906/3947819905.py:22: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      .apply(compute_cohort_stats)





    Text(0, 0.5, 'Retention rate')




    
![png](module_2_eda_files/module_2_eda_64_2.png)
    


**OBSERVACIONES**

- Usuarios clasificados en dos categorías que se diferencian principalmente por el número de productos en su lista de regulares.
- Información sobre la unidad familiar de algunos de los clientes.

- El número de compras ha aumentado mientras que el número de productos en cada cesta de la compra ha disminuido: ha habido un cambio en el comportamiento de compra.
- Muchos productos se han descontinuado del inventario por lo que faltan datos sobre compras pasadas.
- Precio medio de compra de unos 50 euros. Aparentemente ha subido respecto al inicio pero no se puede saber a causa de los productos que fueron comprados en el pasado que no aparecen en el inventario.
- Los productos que más se venden son los de limpieza, junto a comida envasada y sustitutivos de la leche. El más vendido con diferencia es la leche de avena oatly. Algunos de los productos más vendidos ya no están en el inventario.

- La retención de clientes es mejor para cohortes pasadas. Cerca del 30% de los usuarios de la primera cohorte han sido preservados hasta las últimas fechas del dataset, mientras que las últimas cohortes prácticamente no han vuelto a comprar tras su primer mes.

*2. Exploratory Data Analysis*


```python
boxbuilder = pd.read_csv('/home/alvaro/groceries/feature_frame.csv')
boxbuilder.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1547128 entries, 0 to 1547127
    Data columns (total 27 columns):
     #   Column                            Non-Null Count    Dtype  
    ---  ------                            --------------    -----  
     0   variant_id                        1547128 non-null  int64  
     1   product_type                      1547128 non-null  object 
     2   order_id                          1547128 non-null  int64  
     3   user_id                           1547128 non-null  int64  
     4   created_at                        1547128 non-null  object 
     5   order_date                        1547128 non-null  object 
     6   user_order_seq                    1547128 non-null  int64  
     7   outcome                           1547128 non-null  float64
     8   ordered_before                    1547128 non-null  float64
     9   abandoned_before                  1547128 non-null  float64
     10  active_snoozed                    1547128 non-null  float64
     11  set_as_regular                    1547128 non-null  float64
     12  normalised_price                  1547128 non-null  float64
     13  discount_pct                      1547127 non-null  float64
     14  vendor                            1547127 non-null  object 
     15  global_popularity                 1547127 non-null  float64
     16  count_adults                      1547127 non-null  float64
     17  count_children                    1547127 non-null  float64
     18  count_babies                      1547127 non-null  float64
     19  count_pets                        1547127 non-null  float64
     20  people_ex_baby                    1547127 non-null  float64
     21  days_since_purchase_variant_id    1547127 non-null  float64
     22  avg_days_to_buy_variant_id        1547127 non-null  float64
     23  std_days_to_buy_variant_id        1547127 non-null  float64
     24  days_since_purchase_product_type  1547127 non-null  float64
     25  avg_days_to_buy_product_type      1547127 non-null  float64
     26  std_days_to_buy_product_type      1547127 non-null  float64
    dtypes: float64(19), int64(4), object(4)
    memory usage: 318.7+ MB


Las fechas no están en formato datetime. Hay un valor nulo en algunas de las columnas.


```python
# convertir fechas a datetime
boxbuilder.created_at = pd.to_datetime(boxbuilder.created_at)
boxbuilder.order_date = pd.to_datetime(boxbuilder.order_date,format='%Y-%m-%d %H:%M:%S')
boxbuilder.dtypes
```




    variant_id                                   int64
    product_type                                object
    order_id                                     int64
    user_id                                      int64
    created_at                          datetime64[ns]
    order_date                          datetime64[ns]
    user_order_seq                               int64
    outcome                                    float64
    ordered_before                             float64
    abandoned_before                           float64
    active_snoozed                             float64
    set_as_regular                             float64
    normalised_price                           float64
    discount_pct                               float64
    vendor                                      object
    global_popularity                          float64
    count_adults                               float64
    count_children                             float64
    count_babies                               float64
    count_pets                                 float64
    people_ex_baby                             float64
    days_since_purchase_variant_id             float64
    avg_days_to_buy_variant_id                 float64
    std_days_to_buy_variant_id                 float64
    days_since_purchase_product_type           float64
    avg_days_to_buy_product_type               float64
    std_days_to_buy_product_type               float64
    dtype: object




```python
boxbuilder.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808429314180</td>
      <td>3537167515780</td>
      <td>2020-10-06 10:37:05</td>
      <td>2020-10-06</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
boxbuilder[boxbuilder.vendor.isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1547127</th>
      <td>33719430283396</td>
      <td>facialskincare</td>
      <td>2917177655428</td>
      <td>3904236028036</td>
      <td>2021-02-08 14:50:24</td>
      <td>2021-02-08</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 27 columns</p>
</div>



Al estar todos los valores nulos en una sola fila puede que no merezca la pena averiguar cómo imputar cada uno de ellos


```python
boxbuilder=boxbuilder.dropna()
```


```python
boxbuilder.outcome.value_counts()
```




    outcome
    0.0    1521308
    1.0      25819
    Name: count, dtype: int64




```python
cols_binarias=pd.DataFrame()
for col in boxbuilder.columns:
    if set (boxbuilder[col].unique()).issubset({0, 1}):
        cols_binarias[col] = boxbuilder[col]
cols_binarias.columns
```




    Index(['outcome', 'ordered_before', 'abandoned_before', 'active_snoozed',
           'set_as_regular', 'count_babies'],
          dtype='object')




```python
cols_binarias = cols_binarias.drop(['outcome','count_babies'],axis=1)
for col in cols_binarias:
    print(f'{col}')
    print(f'Value counts: {boxbuilder[col].value_counts().to_dict()}')
    print(f'Outcome medio: {boxbuilder.groupby(col)["outcome"].mean().to_dict()}')
    print('\n')

```

    ordered_before
    Value counts: {0.0: 1501121, 1.0: 46006}
    Outcome medio: {0.0: 0.01168926422320386, 1.0: 0.17980263443898623}
    
    
    abandoned_before
    Value counts: {0.0: 1545782, 1.0: 1345}
    Outcome medio: {0.0: 0.01606759556004663, 1.0: 0.7301115241635687}
    
    
    active_snoozed
    Value counts: {0.0: 1542280, 1.0: 4847}
    Outcome medio: {0.0: 0.0163465777939155, 1.0: 0.1254384155147514}
    
    
    set_as_regular
    Value counts: {0.0: 1538924, 1.0: 8203}
    Outcome medio: {0.0: 0.015352934907766725, 1.0: 0.26721931000853344}
    
    


Todas las columnas binarias tienen un outcome medio superior cuando su valor es igual a 1. Esto quiere decir que cuando el usuario ha interactuado anteriormente con el producto es más probable que lo compre. Como la frecuencia de valores igual a 1 en estas variables es bastante baja, se podrían combinar todas ellas en una única columna.


```python
cols_numericas = boxbuilder.select_dtypes(include=['float']).drop(cols_binarias,axis=1)
cols_numericas.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>outcome</th>
      <th>normalised_price</th>
      <th>discount_pct</th>
      <th>global_popularity</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
      <td>1.547127e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.668835e-02</td>
      <td>1.070403e-01</td>
      <td>1.817481e-01</td>
      <td>1.550771e-02</td>
      <td>2.017721e+00</td>
      <td>5.498062e-02</td>
      <td>3.580184e-03</td>
      <td>5.146378e-02</td>
      <td>2.072702e+00</td>
      <td>3.318216e+01</td>
      <td>3.497072e+01</td>
      <td>2.687895e+01</td>
      <td>3.173823e+01</td>
      <td>3.093761e+01</td>
      <td>2.605230e+01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.281010e-01</td>
      <td>1.012933e-01</td>
      <td>2.045323e-01</td>
      <td>2.051363e-02</td>
      <td>2.101152e-01</td>
      <td>3.275417e-01</td>
      <td>5.972746e-02</td>
      <td>3.017916e-01</td>
      <td>3.946277e-01</td>
      <td>4.416048e+00</td>
      <td>8.679242e+00</td>
      <td>6.099032e+00</td>
      <td>1.346131e+01</td>
      <td>3.557168e+00</td>
      <td>2.623210e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>2.141502e-02</td>
      <td>1.669449e-03</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.509980e+00</td>
      <td>0.000000e+00</td>
      <td>7.000000e+00</td>
      <td>5.338093e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000e+00</td>
      <td>5.123340e-02</td>
      <td>7.159905e-02</td>
      <td>4.842615e-03</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.300000e+01</td>
      <td>3.000000e+01</td>
      <td>2.439688e+01</td>
      <td>3.000000e+01</td>
      <td>2.900000e+01</td>
      <td>2.427618e+01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000e+00</td>
      <td>7.834101e-02</td>
      <td>1.135857e-01</td>
      <td>1.063830e-02</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.300000e+01</td>
      <td>3.400000e+01</td>
      <td>2.806743e+01</td>
      <td>3.000000e+01</td>
      <td>3.100000e+01</td>
      <td>2.608188e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000e+00</td>
      <td>1.135809e-01</td>
      <td>2.008032e-01</td>
      <td>2.027027e-02</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.300000e+01</td>
      <td>3.900000e+01</td>
      <td>3.072716e+01</td>
      <td>3.000000e+01</td>
      <td>3.300000e+01</td>
      <td>2.793528e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000e+00</td>
      <td>8.807265e-01</td>
      <td>1.325301e+00</td>
      <td>4.254386e-01</td>
      <td>5.000000e+00</td>
      <td>3.000000e+00</td>
      <td>1.000000e+00</td>
      <td>6.000000e+00</td>
      <td>5.000000e+00</td>
      <td>1.480000e+02</td>
      <td>6.400000e+01</td>
      <td>4.234402e+01</td>
      <td>1.480000e+02</td>
      <td>3.950000e+01</td>
      <td>3.183274e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols_numericas['outcome'] = boxbuilder['outcome']
corr = cols_numericas.corr()
sns.heatmap(corr[(corr >= 0.1)| (corr <= -0.1)],cmap='viridis',linewidths=0.1)
```




    <Axes: >




    
![png](module_2_eda_files/module_2_eda_80_1.png)
    


Correlaciones notables entre las variables que tienen que ver con la unidad familiar. 'avg' y 'std days to buy' también tienen cierta correlación entre sí. 


```python
col = 'normalised_price'
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 0, col], label='0')
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 1, col], label='1')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f40cb0ce590>




    
![png](module_2_eda_files/module_2_eda_82_1.png)
    



```python
col = 'global_popularity'
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 0, col], label='0')
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 1, col], label='1')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f40caeea2d0>




    
![png](module_2_eda_files/module_2_eda_83_1.png)
    



```python
col = 'count_adults'
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 0, col], label='0')
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 1, col], label='1')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f40c823cbd0>




    
![png](module_2_eda_files/module_2_eda_84_1.png)
    


Parece ser que en este dataset han imputado los valores nulos de las variables 'count people' por un único valor, alterando la distribución.


```python
col = 'days_since_purchase_variant_id'
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 0, col], label='0')
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 1, col], label='1')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f40c81c4550>




    
![png](module_2_eda_files/module_2_eda_86_1.png)
    


Casi todas las filas de esta columna tienen el mismo valor.


```python
col = 'avg_days_to_buy_variant_id'
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 0, col], label='0')
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 1, col], label='1')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f40c8185390>




    
![png](module_2_eda_files/module_2_eda_88_1.png)
    



```python
cols_categoricas = boxbuilder.select_dtypes(include=['object'])
cols_categoricas.nunique()
```




    product_type     53
    vendor          183
    dtype: int64



Muchas categorías de vendedores y tipos de productos, una opción es hacer frequency encoding


```python
# Frequency encoding
for col in cols_categoricas.columns: 
    frequency_map = boxbuilder[col].value_counts(normalize=True).to_dict()
    boxbuilder[f'{col}_freq_encoded'] = boxbuilder[col].map(frequency_map)
```


```python
col = 'vendor_freq_encoded'
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 0, col], label='0')
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 1, col], label='1')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f40c7e14550>




    
![png](module_2_eda_files/module_2_eda_92_1.png)
    



```python
col = 'product_type_freq_encoded'
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 0, col], label='0')
sns.kdeplot(boxbuilder.loc[lambda x: x.outcome == 1, col], label='1')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f40c64204d0>




    
![png](module_2_eda_files/module_2_eda_93_1.png)
    


**OBSERVACIONES**

- La variable objetivo tiene un valor igual a uno en una muy pequeña proporción de las filas.
- Hay varias columnas que indican si el usuario ha interactuado previamente con el procucto de varias formas. Estas se podrían juntar en una única columna.
- Las variables categóricas tienen una alta cardinalidad, por lo que es razonable codificarlas por frecuencia.
- Hay ciertas correlaciones entre algunas de las variables. En la fase de desarrollo del modelo podría probarse a deshacerse de algunas o combinarlas para reducir la dimensionalidad.
- Algunas distribuciones de variables parecen haber sido alteradas por la imputación de valores nulos. En la fase de desarrollo podría comprobarse si esto perjudica al modelo.
