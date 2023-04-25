import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report,RocCurveDisplay
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier

df1 = pd.read_csv("C:/Users/yulas/Desktop/HousePrice/data/test.csv",delimiter=',', quotechar='"')
df2 = pd.read_csv("C:/Users/yulas/Desktop/HousePrice/data/train.csv",delimiter=',', quotechar='"')
df = pd.concat([df1, df2])
print(df.head())

#################################
# VERİ ANALİZİ #
#################################

cat_cols = [col for col in df.columns if df[col].dtype == "O"]
print(cat_cols)
num_cols = [col for col in df.columns if df[col].dtype != "O"]
print(num_cols)
num_but_cat = [col for col in df.columns if (df[col].nunique() < 10) & (df[col].dtype != "O")]
print(num_but_cat)
cat_cols += num_but_cat
print(cat_cols)
num_cols = [col for col in num_cols if col not in num_but_cat]
print(num_cols)

print("###### ÖZET #####")
print(f"Observations: {df.shape[0]}")
print(f"Variables: {df.shape[1]}")
print(f'num_cols: {len(num_cols)}')
print(f'num_but_cat: {len(num_but_cat)}')

def target_analys (dataframe,target,col):
    if col in cat_cols:
        print(dataframe.groupby(col).agg({target : ["mean"]}))
    elif col in num_cols:
        print(dataframe.groupby(target).agg({col : ["mean"]}))

target_analys(df,"SalePrice","SalePrice")
target_analys(df,"SalePrice","LotArea")   

def outlier (dataframe,col,q1 = 0.05, q3 = 0.95):
    quartile1 = dataframe[col].quantile(q1)
    quartile3 = dataframe[col].quantile(q3)
    IQR = q3 - q1
    low = q1 - IQR*1.5
    up = q3 + IQR*1.5
    return low,up

age_outlier = outlier(df,"LotArea")
print(age_outlier)

def check_out (dataframe,col):
    low,up = outlier(dataframe,col)
    if dataframe[(dataframe[col] > up) | (dataframe[col] < low)].any(axis=None):
        print("True")
        return True
    else:
        print("False")
        return False
    
check_out(df,"LotArea")

def show_out (dataframe,col):
    low,up = outlier(dataframe,col)
    if dataframe[(dataframe[col] > up) | (dataframe[col] < low)].shape[0] > 10:
         print(dataframe[((dataframe[col] < low) | (dataframe[col] > up))].head()) 
    else:
        print(dataframe[((dataframe[col] < low) | (dataframe[col] > up))])

show_out(df,"LotArea")

Missing = df.isnull().values.any()

def missing(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    print(na_columns,ratio)

if Missing == True:
    for col in df.columns:
        print(df.isnull().sum())
        missing(df,col)
else:
    print("Don't have any missing values")  

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "SalePrice", cat_cols)


##################################
# MODELLEME #
##################################
for col in df.columns:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()

y= df["SalePrice"]

x = df.drop(["SalePrice"],axis=1)

x_train , x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.20, random_state=17)

model = LogisticRegression().fit(x_train,y_train)
y_predict = model.predict(x_test)

print(classification_report(y_test, y_predict))


log_model = LogisticRegression().fit(x, y)

y_pred = log_model.predict(x) 
print(y_pred[0:10],
      y[0:10])

print(classification_report(y_test, y_predict))

knn_model = KNeighborsClassifier()
print(knn_model.get_params()) # ön tanımlı parametreler

knn_params = {"n_neighbors": range(2, 50)} #amacımız bu komşuluk sayısını değiştirerek uygun olan en optimum komşu sayısını bulmak

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)
print(knn_gs_best)
print(knn_gs_best.best_params_)
