import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import math

# 1.PROBLEM STATEMENT AND READ DATA

pd.set_option('display.max_columns', 7)
pd.set_option('display.width', None)
data = pd.read_csv('datasets/car_state.csv')
print('************************************************************************************')
print('PRIKAZ PRVIH I POSLEDNJIH PET REDOVA U TABELI'.center(85, ' '))
print('************************************************************************************')
print(pd.concat([data.head(), data.tail()]))

# 2.DATA ANALYSIS

print('************************************************************************************')
print('PRIKAZ KONCIZNIH INFORMACIJA O SADRZAJU TABELE'.center(85, ' '))
print('************************************************************************************')
print(data.info())
print('************************************************************************************')
print('PRIKAZ STATISTICKIH INFORMACIJA O SVIM ATRIBUTIMA'.center(85, ' '))
print('************************************************************************************')
# print(data.describe())
print(data.describe(include=[object]))
# print(data.describe(include='all'))
print('************************************************************************************')

pp = sb.pairplot(data=data, y_vars=['status'],
                 x_vars=['buying_price', 'maintenance', 'doors', 'seats', 'trunk_size', 'safety'])
plt.show()

# 4.FEATURE ENGINEERING

# not useful features:
# useful features: buying_price, maintenance, doors, seats, trunk_size, safety

data_train = data[['buying_price', 'maintenance', 'doors', 'seats', 'trunk_size', 'safety']]  # DataFrame
labels = data['status']  # Series

categorical_cols = ['buying_price', 'maintenance', 'trunk_size', 'safety']
df = pd.get_dummies(data, columns=categorical_cols)
pd.set_option('display.max_columns', 17)
pd.set_option('display.width', None)

df['doors'] = df['doors'].str.replace(' or more', '')
df['doors'] = pd.to_numeric(df['doors'])
df['seats'] = df['seats'].str.replace(' or more', '')
df['seats'] = pd.to_numeric(df['seats'])


def normalize(df, feature_name):
    max_value = df[feature_name].max()
    min_value = df[feature_name].min()
    df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return df


normalize(df, 'doors')
normalize(df, 'seats')


X = df[df.columns.difference(['status'])]  # DataFrame
y = df['status']


def rastojanje(row1, row2):
    suma = 0
    for i in range(len(row1)):
        suma += (row1[i] - row2[i]) * (row1[i] - row2[i])
    d = math.sqrt(suma)
    return d


def moj_knn(dataframe, train, prvi_test):
    niz = []
    # k = int(np.round(math.sqrt(len(train))))
    k = math.floor(len(train.index) ** (1 / 2))
    if k % 2 != 1:
        k += 1
    num = len(train.index)
    for i in range(num):
        d = rastojanje(prvi_test, train.iloc[i])
        niz.append([train.iloc[i], d])
    niz.sort(key=lambda x: x[1])
    niz = niz[:k]
    noviniz = []
    for j in range(len(niz)):
        noviniz.append(niz[j][0])
    return noviniz


print('************************************************************************************')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)
num = len(X_train.index)
tacni = 0
netacni = 0
y_pred = []
for i in range(len(X_test)):
    array = moj_knn(df, X_train, X_test.iloc[i])
    cnt = [0, 0, 0, 0]
    ind = X_test.iloc[i].name
    for p in range(len(y_train)):
        for j in range(len(array)):
            if y_train.index[p] == array[j].name:
                if y_train.iloc[p] == 'unacceptable':
                    cnt[0] += 1
                if y_train.iloc[p] == 'acceptable':
                    cnt[1] += 1
                if y_train.iloc[p] == 'good':
                    cnt[2] += 1
                if y_train.iloc[p] == 'very good':
                    cnt[3] += 1
    maximum = max(cnt)
    a = np.where(cnt == np.amax(cnt))
    naj = a[0][-1]
    mojstatus = ''
    if naj == 0:
        mojstatus = 'unacceptable'
    if naj == 1:
        mojstatus = 'acceptable'
    if naj == 2:
        mojstatus = 'good'
    if naj == 3:
        mojstatus = 'very good'
    if mojstatus == y_test[ind]:
        tacni += 1
    else:
        netacni += 1
    y_pred.append(naj)
print('tacni: ', tacni)
print('netacni: ', netacni)
print('tacnost: ', tacni / len(y_test) * 100, '%')

# built-in
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

y_pred_skl = neigh.predict(X_test)
s = 0
for i, index in enumerate(y_test.index):
    if y_test[index] == y_pred_skl[i]:
        s += 1

print('Ugradjena tacnost: ', (s / len(y_test.index)) * 100, '%')


def get_value(test):
    value = -1
    if test == 'unacceptable':
        value = 0
    if test == 'acceptable':
        value = 1
    if test == 'good':
        value = 2
    if test == 'very good':
        value = 3
    return value


def loss(y_pred, y_test):
    sum = 0
    for i in range(len(y_pred)):
        sum += abs(y_pred[i] - get_value(y_test.iloc[i]))
    return sum / len(y_pred)


def loss_ugr(y_pred, y_test):
    sum = 0
    for i in range(len(y_pred)):
        sum += abs(get_value(y_pred[i]) - get_value(y_test.iloc[i]))
    return sum / len(y_pred)


print('Loss: ', loss(y_pred, y_test), '%')
print('Ugradjeni loss: ', loss_ugr(y_pred_skl, y_test), '%')
