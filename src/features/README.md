# Break down of EDA

## Overview of columns
### df_train_transaction columns
- TransactionID - unique identifier
- isFraud - Target
- TransactionDT - time delta from given data
- TransactionAmt - Actual amount
- ProductCD - product code
- card1-card6 - payment card information, card type, card cat, issue bank, country, etc
- addr - address
- dist - distance 
- R/P_emaildomain - purchaser and recipient email domain
- C1-14 masked counting info
- D1-D15 time delta, such as days between previous transaction
- M1-M9, match, such as names on card, address etc 
- VXXX Engineered rich features, ranking, counting etc
 
#### Categorical Features
- ProductCD 
- card1 - card6 
- addr1, addr2 
- P_emaildomain R_emaildomain 
- M1 - M9

### df_train_indentity columns
- TransactionID - secondary key
- id_01-id_38: network connection info, digital signature etc 
- DeviceType, DeviceInfo
#### Categorical Features
- DeviceType
- DeviceInfo
- id_12 - id_38

## Fraud distribution
![Fraud Pie](eda_images/fraud_dist.png)

Only have 3.5% of transactions are fraud in train data

# Categorical variables

## Product CD
![ProductCD EDA](eda_images/productCD_eda.png)

### Observations:
ProductCD type C has the highest fraud rate

## Card1-6
card1-6 are all categorical, but card1, card2, card3, card5 have too many unique values, especially card1 13k values. Frequency encode to preserve "rare" and "frequent" notation, but not exploding number of dimensions compared to doing things like OHE
![Card EDA](eda_images/card_eda.png)

### Observations:
For card3, there is originally 114 unique values, fraud transactions seems to be congested in non frequent values. 
For card4, discover card has the highest fraud rate.
For card6, credit card has the highest fraud rate. 

## Addr
Address have many unique values as well, do frequency encoding again. 
![Addr EDA](eda_images/addr_eda.png)
No immediate obvious relation

## Email, P and R
### Pemail_domain
Many unique values, ~60 domains for P_emaildomain. Frequency encoding may not be as good because some email may consistently be bad, and frequency encode loses this information. Infrequent emails (less than 500) are parked under 'Others'

Before | After
:-------------------------:|:-------------------------:
![Before Clean](eda_images/Pemail%20distribution_b4.png) | ![After Clean](eda_images/Pemail%20distribution_after.png)

Plotting fraud rates with count, and total transction, seperated by P_emaildomain

![P Email](eda_images/Pemail_fraud.png)

We see that mail.com is quite suspicious
### Remail_domain
Similary, Remail has alot of unique domains, after cleaning up in a similar manner, except infrequent is 300, we plot the same graph.

Before | After
:-------------------------:|:-------------------------:
![Before Clean](eda_images/Remail%20distribution_b4.png) | ![After Clean](eda_images/Remail%20distribution_after.png)

![R Email](eda_images/Remail_fraud.png)

We see that icloud.com is quite suspicious. Google also has a relatively high fraud rate 

## M1- M9
![M1-M9](eda_images/M1-M9.png)

Nan has a relatively higher fraud rate in most M1-M9 columns, with exceptions of M4, M5. 

## DeviceInfo
Many unique devices, around ~1700 unique values, frequency encode to plot the distribution with respect to fraud. Many Nan too

![DeviceInfo](eda_images/DeviceInfo_eda.png)

## DeviceType

![DeviceType](eda_images/DeviceType_eda.png)

mobile dvices have the highest fraud rates

## id_columns
id columns are tied to identity table, of which only ~144k out of 590k transactions have identity record, many columns are thus NaN, 
![Id_cat](eda_images/id12-id38.png)
### Findings:
Generally id12-38 seems quite useful, where there is clear difference in fraud rate between either category or frequency encoded. 
id_18, id_24, id_32 seems to be able to clean up, and 

# Timedelta 
The time given is in seconds, timedelta from a given date. The min value is 86400, which 60s * 60min * 24h. So it starts one day from a given time. We are able to break it down to hour_of_day, day_of_week. Test data do not intersect. 

![TimeDelta](eda_images/timedelta.png)

### Findings:
- Number of transactions seems to follow a cyclic pattern across the days, fraud rate on the other hand doesnt seem to follow the same pattern
- Fraud rate by hour shows a very distinct rise in fraud rate, with a peak of more than 10% at 7 
- Fraud rate of day of week doesnt seem to have any significant pattern 
- Fraud rate shifts across the day, it's non stationary and drifts with time

# Numerical features
We will exclude V-columns as there are too many of them 339. 

## TransactionAmt

![TransactionAmt](eda_images/isFraud_log.png)
No immediate obvious patterns, fraud seems to have slighlt more extreme values but as a standalone feature, transaction amount doesnt seem that useful

![TransactionAmtDist](eda_images/transaction_amt_dist.png)
As mentioned, the curve seems fatter near the ends for fraud, but given the low fraud rate, difficult to draw any conclusions


![TransactionAmtDist](eda_images/transaction_amt_dist_productCD.png)
However, when we split transaction amount with ProductCD, the transaction seems to show a clearer differentiation.

## Plotting dist1 and dist2
![distanceImages](eda_images/dist_eda.png)
Distance images doesnt seem immediately useful. But take note that nan rates are 0.59 and 0.93 for dist1 and dist2 respectively.  

## PLotting the C1-14
Plotting C1-15, we see some valuable columns 
![C_ColumnsEDA](eda_images/C_eda.png)
Some columns seem quite useful. Only a handful of columns like C6 doesnt seem that useful. 

## Plotting the D1-15
![D_ColumnsEDA](eda_images/D_eda.png)
A similar process is done with D features. Likewise, only some D featues seem to not have significantly different, like D6, D9 etc. 

## Plotting id1-id11
![id_columns](eda_images/id1_11.png)
id columns themselves do not seem that useful

# Correlation and Redundancy 
Plotting the correlation map for all numeric (except V)
![nonV_corr](eda_images/correlationmap_numeric.png)

We see many C columns are highly correlated with one anotherm and many D as well. 