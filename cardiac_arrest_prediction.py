import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import seaborn as sns
import optuna



df = pd.read_csv(r"C:\Users\adamp\OneDrive\Desktop\projekty\cardiac_arrest_prediction\data.csv"
                   ,engine='python')


df.info()
df.describe()

for col in df.columns:
    print('{}: {}'.format(col,df[col].nunique()))
    
    
    
df.replace('?', np.NAN, inplace = True)    
    
 
    
    
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()    
    
    
df.drop(columns = ['thal','ca','slope',"restecg"], inplace = True)   
    
df.dropna(inplace = True)   

df = df.astype({
       "trestbps": float,
        "chol": float,
        "fbs": float,
       
        "thalach": float,
        "exang": float
    })

df.info()

plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
 
# removing outliers
outlier_condition = ((df < (Q1 - 1.5 * IQR)) | (df> (Q3 +1.5 * IQR)))
df_o_iqr = df[~outlier_condition.any(axis=1)]



plt.figure(figsize=(10, 5))
sns.boxplot(data=df_o_iqr)
plt.xticks(rotation=90)
plt.show()



X = df_o_iqr.iloc[:,:-1]
y = df_o_iqr.iloc[:,-1]



scaler = StandardScaler()
scaler.fit(X)
std_X = scaler.transform(X)



X_train, X_test, y_train, y_test = train_test_split(std_X, y, test_size=0.2)

# testing difrent models

#LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
score_lr = lr.score(X_test,y_test)


#RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
score_rfc = rfc.score(X_test,y_test)


#ElasticNet
elastic_df = pd.DataFrame({'param_value': np.arange(start = 0.1, stop = 10.2, step = 0.1),
                      'r2_result': 0.,
                      'number_of_features':0})
 
for i in range(elastic_df.shape[0]):
    
    alpha = elastic_df.at[i, 'param_value']
    EN = ElasticNet(alpha=alpha,max_iter=500)
    EN.fit(X_train, y_train)
    elastic_df.at[i, 'r2_result'] = r2_score(y_test, EN.predict(X_test))
    elastic_df.at[i, 'number_of_features'] = len(EN.coef_[ EN.coef_ > 0])



best_EN = ElasticNet(alpha = 0.1,max_iter=500)
best_EN.fit(X_train,y_train)
score_best_EN = best_EN.score(X_test,y_test)










