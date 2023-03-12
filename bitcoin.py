import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('BABD-13.csv')
print(data.columns)
print(data.describe())
print(data.info())
print(data.isna().sum())
col=data.select_dtypes(include='number').columns.values
print(col)
#s    n.countplot(data['labels'])
#plt.show()

data['z-scores']=(data.PAIa13-data.PAIa13.mean())/(data.PAIa13.std())
df=data[(data['z-scores'] >-3)&(data['z-scores']<3)]
q1=df.PAIa13.quantile(0.25)
q3=df.PAIa13.quantile(0.75)
iqr=q3-q1
u=q3+1.5*iqr
l=q1-1.5*iqr
df=df[(df.PAIa13 <u)&(df.PAIa13 <l)]


q_1=df.PAIa12.quantile(0.25)
q_3=df.PAIa12.quantile(0.75)
iq_r=q_3-q_1
u_=q_3+1.5*iq_r
l_=q_1-1.5*iq_r
df=df[(df.PAIa12 <u_)&(df.PAIa12 <l_)]
col=df.select_dtypes(include='number').columns.values
#for i in col:
#    sn.boxplot(df[i])
#    plt.show()

x=df[['PAIa13','PAIa16-1','PAIa16-R1','PAIa21-1','PAIa21-2','PAIa21-3','PAIa21-4','PAIa21-R1','PAIa21-R2','PAIa21-R3','PAIa21-R4','PAIa22-1','PAIa22-R1',
      'PDIa11-1','PDIa11-2','PDIa11-R1','PDIa11-R2','PTIa1','S1-2','S1-6','S2-2','S2-3','S6']]
a=df['PAIa13','PAIa16-1','PAIa16-R1','PAIa21-1','PAIa21-2','PAIa21-3','PAIa21-4','PAIa21-R1','PAIa21-R2','PAIa21-R3','PAIa21-R4','PAIa22-1','PAIa22-R1',
      'PDIa11-1','PDIa11-2','PDIa11-R1','PDIa11-R2','PTIa1','S1-2','S1-6','S2-2','S2-3','S6']
for i in a:
      for j in a:
            plt.scatter(df[i],marker='o',label=f'{i}',color='red')
            plt.scatter(df[j],marker='x',label=f'{j}',color='blue')
            plt.title(f'{i} vs {j}')
            plt.legend()
            plt.show()
for i in a:
      for j in a:
            plt.plot(df[i],marker='o',label=f'{i}',color='red')
            plt.plot(df[j],marker='x',label=f'{j}',color='blue')
            plt.title(f'{i} vs {j}')
            plt.legend()
            plt.show()
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
df['Label']=lab.fit_transform(df['label'])
y=df['Label']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
lr=LogisticRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
print(xgb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)
print(dtree.score(x_test,y_test))