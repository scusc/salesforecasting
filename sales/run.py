from datetime import datetime, timedelta,date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from __future__ import division

import keras

df_sales = pd.read_csv('test.csv')
df_sales.rename(columns = {'Order Date':'date'},inplace=True)

df_sales1 = pd.read_csv('date_sales.csv')
df_sales1.sort_values(by=['date'],inplace =True)

df_sales1 = df_sales1.append(df_sales)

df_sales1.reset_index(inplace=True)

df_sales1.drop(columns=['index'],inplace=True)


new = pd.date_range(df_sales1.date.iloc[-1], periods=8)
s = pd.Series(new[1:])
new = df_sales1['date'].append(s, ignore_index=True)
df1 = pd.DataFrame(new, columns=['date'])
df1['date'] = pd.to_datetime(df1['date']).dt.normalize()
df1['sales'] = df_sales1['sales']
df1.fillna(value=0,inplace=True)

df_diff = df1.copy()
df_diff['prev_sales'] = df_diff['sales'].shift(1)
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['sales'] - df_diff['prev_sales'])

df_supervised = df_diff.drop(['prev_sales'],axis=1)
for inc in range(1,13):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)

df_supervised = df_supervised.dropna().reset_index(drop=True)


from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['sales','date'],axis=1)


train_set, test_set = df_model[0:-7].values, df_model[-7:].values

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)

# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = keras.models.load_model("model.h5")

y_pred = model.predict(X_test,batch_size=1)

y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])

pred_test_set = []
for index in range(0,len(y_pred)):
    #print (np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))

pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])


pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

result_list = []
sales_dates = list(df1[-8:].date)
act_sales = list(df1[-8:].sales)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['date'] = sales_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)


df_result.rename(columns={"pred_value":"sales"},inplace=True)
df_result.to_csv('test.csv',index=False)


sales=pd.DataFrame()
sales['Prediction Date'], sales['Predicted Sales'] = df_result['date'],df_result['sales']

df_sales_pred = pd.merge(df1,df_result,on='date',how='left')


sales.to_csv('output.csv',index=False)































