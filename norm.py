import pickle
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler


scaler = StandardScaler()
df_raw = pd.read_csv('./ETT-small/train_set.csv')
df_test = pd.read_csv('./ETT-small/test_set.csv')
df_val = pd.read_csv('./ETT-small/validation_set.csv')

cols_data = df_raw.columns[1:]

df_data = df_raw[cols_data]
df_test = df_test[cols_data]
df_val = df_val[cols_data]

train_data = df_data
test_data = df_test
val_data = df_val

scaler.fit(train_data.values)
train = scaler.transform(df_data.values)
test = scaler.transform(df_test.values)
val = scaler.transform(df_val.values)

train_df = pd.DataFrame(train, columns=cols_data)
test_df = pd.DataFrame(test, columns=cols_data)
val_df = pd.DataFrame(val, columns=cols_data)

train_df.to_csv('./train_set.csv')
test_df.to_csv('./test_set.csv')
val_df.to_csv('./validation_set.csv')

# 将统计参数保存到文件
with open('norm_params.pickle', 'wb') as f:
    pickle.dump(scaler, f)

with open('norm_params.pickle', 'rb') as f:
    scalers = pickle.load(f)
    print(scalers.mean_)
# x = scalers.inverse_transform(train)
mean_ = scalers.mean_
std_ = scaler.scale_
x = (train * std_) + mean_
x = pd.DataFrame(x, columns=cols_data)
print(x.head())
print(df_raw.head())