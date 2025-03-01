import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import optuna
import seaborn as sns
from sklearn.metrics import mean_squared_error


df = pd.read_csv(r"C:\Users\adamp\OneDrive\Desktop\projekty\data_machine\data.csv"
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




#Definitons for models
def model_testing(model, X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)




#LogisticRegression
def objectiveLR(trial):

    C = trial.suggest_loguniform('C', 1e-4, 10.0) 
    solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear'])  


    lr = LogisticRegression(C=C, solver=solver, max_iter=500)

    lr.fit(X_train,y_train)
    

    accuracy = lr.score(X_test, y_test)
    
    return accuracy 






#RandomForestClassifier
def objectiveRFC(trial):


    n_estimators = trial.suggest_int('n_estimators', 50, 300)  
    max_depth = trial.suggest_int('max_depth', 3, 20) 
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)  
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)  
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])  

    rfc = RandomForestClassifier(n_estimators= n_estimators,
                                 max_depth =  max_depth, 
                                 min_samples_split= min_samples_split, 
                                 min_samples_leaf = min_samples_leaf,
                                 max_features = max_features,
                                 random_state=42)

    rfc.fit(X_train,y_train)
    

    accuracy = rfc.score(X_test, y_test)
    
    return accuracy 






#ElasticNet
def objectiveEN(trial):
    alpha = trial.suggest_loguniform('alpha', 1e-4, 10.0) 
    l1_ratio = trial.suggest_uniform('l1_ratio', 0.0, 1.0)  

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)  

    return mse




# testing diffrent models

#LogisticRegression
study = optuna.create_study(direction='maximize') 
study.optimize(objectiveLR, n_trials=50)

best_params_LR = study.best_params

LR = LogisticRegression(C=best_params_LR['C'], solver=best_params_LR['solver'], max_iter=500)
score_LR = model_testing(LR, X_train, X_test, y_train, y_test)

#RandomForestClassifier
study = optuna.create_study(direction='maximize') 
study.optimize(objectiveRFC, n_trials=50)

best_params_rfc = study.best_params

RFC = RandomForestClassifier(n_estimators= best_params_rfc['n_estimators'],
                             max_depth =  best_params_rfc['max_depth'], 
                             min_samples_split= best_params_rfc['min_samples_split'], 
                             min_samples_leaf = best_params_rfc['min_samples_leaf'],
                             max_features = best_params_rfc['max_features'],
                             random_state=42)

score_RFC = model_testing(RFC, X_train, X_test, y_train, y_test)


#ElasticNet

study = optuna.create_study(direction="minimize")  
study.optimize(objectiveEN, n_trials=50) 


best_pam_EN = study.best_params

EN= ElasticNet(alpha=best_pam_EN['alpha'], l1_ratio=best_pam_EN['l1_ratio'])

score_EN = model_testing(EN, X_train, X_test, y_train, y_test)


