import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
## Importing the datasets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
# print('The shape of the train data is (row, column):'+str(train.shape))
# print(train.info())
# print("The shape of the test data is (row, column):"+str(test.shape))
# print(test.info())
passengerid = test.PassengerId
# print(train.info())
# print("*"*40)
# print(test.info())
total = train.isnull().sum().sort_values(ascending=False)
#round() 方法返回浮点数x的四舍五入值。False降序排列
percent = round(train.isnull().sum().sort_values(ascending=False)
                /len(train)*100,2)
pd.concat([total,percent],axis=1,keys=['Total','Percent'])
#DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。
percent = pd.DataFrame(round(train.Embarked.value_counts
                             (dropna=False,# dropna : boolean, default True　默认删除na值
                              normalize=True)*100,2))# normalize : boolean, default False　如果设置为true，则以百分比的形式显示
total = pd.DataFrame(train.Embarked.value_counts(dropna=False))
total.columns = ["Total"]
percent.columns = ['Percent']
pd.concat([total,percent],axis=1)
fig, ax = plt.subplots(figsize=(16,12),ncols=2)#nrows,ncols: 用来确定绘制子图的行数和列数
ax1 = sns.boxplot(x='Embarked',y="Fare",hue="Pclass",data=train, ax = ax[0])#hue（str）：dataframe的列名，按照列名中的值分类形成分类的条形图
ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test, ax = ax[1]);
ax1.set_title("Training Set", fontsize = 18)
ax2.set_title('Test Set',  fontsize = 18)
# fig.show()
# plt.show()
train.Embarked.fillna("C",inplace=True)
# print("Train Cabin missing: " + str(train.Cabin.isnull().sum()/len(train.Cabin)))
# print("Test Cabin missing: " + str(test.Cabin.isnull().sum()/len(test.Cabin)))
survivers = train.Survived
#删除表中的某一行或者某一列更明智的方法是使用drop，它不改变原有的df中的数据，而是返回另一个dataframe来存放删除后的数据。本文出处主要来源于必备工具书《利用python进行数据分析》
train.drop(["Survived"],axis=1,inplace=True)
all_data = pd.concat([train,test],ignore_index=False)
#cabin n.隔间,座舱
all_data.Cabin.fillna("N",inplace=True)#inplace参数的取值：True、False True：直接修改原对象 False：创建一个副本，修改副本，原对象不变（缺省默认）
all_data.Cabin = [i[0] for i in all_data.Cabin]# C123 舱位 可能是字母 或者是字母数字组合 此处将首字母取出 首字母更有用
with_N = all_data[all_data.Cabin == "N"]
witout_N = all_data[all_data.Cabin != "N"]
all_data.groupby("Cabin")['Fare'].mean().sort_values()# 计算每个船舱字母的平均值。
#print(all_data.groupby("Cabin")['Fare'].mean().sort_values())
def cabin_estimator(i):
    a = 0
    if i<16:
        a = "G"
    elif i>=16 and i<27:
        a = "F"
    elif i>=27 and i<38:
        a = "T"
    elif i>=38 and i<47:
        a = "A"
    elif i>= 47 and i<53:
        a = "E"
    elif i>= 53 and i<54:
        a = "D"
    elif i>=54 and i<116:
        a = 'C'
    else:
        a = "B"
    return a
with_N["Cabin"] = with_N.Fare.apply(lambda x:cabin_estimator(x))
all_data = pd.concat([with_N,witout_N],axis=0)
all_data.sort_values(by = 'PassengerId',inplace = True)
train = all_data[:891]
test = all_data[891:]
train["Survived"] = survivers
print(test[test.Fare.isnull()])
#      PassengerId  Pclass                Name   Sex  ...  Ticket  Fare  Cabin Embarked
# 152         1044       3  Storey, Mr. Thomas  male  ...    3701   NaN      B        S
## replace the test.fare null values with test.fare mean
missing_value = test[(test.Pclass == 3) & (test.Embarked == "S") & (test.Sex == "male")].Fare.mean()
test.Fare.fillna(missing_value,inplace=True)
# print ("Train age missing value: " + str((train.Age.isnull().sum()/len(train))*100)+str("%"))
# print ("Test age missing value: " + str((test.Age.isnull().sum()/len(test))*100)+str("%"))
pal = {'male':"green", 'female':"Pink"}
plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Sex",
            y = "Survived",
            data=train,
            palette = pal,
            linewidth=2 )
plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25)
plt.ylabel("% of passenger survived", fontsize = 15)
plt.xlabel("Sex",fontsize = 15);
# plt.show()
pal = {1:"seagreen", 0:"gray"}
sns.set(style="darkgrid")
plt.subplots(figsize = (15,8))
ax = sns.countplot(x = "Sex",
                   hue="Survived",
                   data = train,
                   linewidth=2,
                   palette = pal
)

## Fixing title, xlabel and ylabel
plt.title("Passenger Gender Distribution - Survived vs Not-survived", fontsize = 25)
plt.xlabel("Sex", fontsize = 15);
plt.ylabel("# of Passenger Survived", fontsize = 15)

## Fixing xticks
#labels = ['Female', 'Male']
#plt.xticks(sorted(train.Sex.unique()), labels)

## Fixing legends
leg = ax.get_legend()
leg.set_title("Survived")
legs = leg.texts
legs[0].set_text("No")
legs[1].set_text("Yes")
# plt.show()
# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
## I have included to different ways to code a plot below, choose the one that suites you.
ax=sns.kdeplot(train.Pclass[train.Survived == 0] ,
               color='gray',
               shade=True,
               label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'] ,
               color='g',
               shade=True,
               label='survived')
plt.title('Passenger Class Distribution - Survived vs Non-Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Passenger Class", fontsize = 15)
## Converting xticks into words for better understanding
labels = ['Upper', 'Middle', 'Lower']
plt.xticks(sorted(train.Pclass.unique()), labels);
# print('train[train.Fare > 280]')
#print(train.describe())
print(train.describe(include =['O']))




























