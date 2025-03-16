from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import f1_score

df = pd.read_csv("E:/Ml-Project/HW2_Shahmir_40221570004/titanic.csv")

df = df.drop(['Cabin'], axis=1)
df = df.drop(['PassengerId'], axis=1)
df = df.drop(['Ticket'], axis=1)
df = df.drop(['Name'], axis=1)

# fill_values = round(df['Age'].mode()[0],0)
# fill_values = round(df['Age'].mean(),0)

fill_values = df.groupby('Survived')['Age'].transform(lambda x: round(x.mean(), 0) if x.notna().any() else None)

df['Age'].fillna(fill_values, inplace=True)

# change datatype from string to float
from sklearn.preprocessing import LabelEncoder

stringColumns = df.select_dtypes(include=['object']).columns

for columnName, columnValue in df.items():
    if ((columnName in stringColumns)):
        df[columnName] = LabelEncoder().fit_transform(df[columnName])

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.20, shuffle=False)

Test_X = test.drop(['Survived'], axis=1)
Test_Y = test[['Survived']]

Train_X = train.drop(['Survived'], axis=1)
Train_Y = train[['Survived']]

new_Train_Y = Train_Y.values.ravel()

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(Train_X, new_Train_Y)

trainPred = clf.predict(Train_X)
testPred = clf.predict(Test_X)

f1_score(Test_Y, testPred, average='macro')

Test_survived_count = (Test_Y['Survived'] == 1).sum()
Pred_survived_count = (testPred == 1).sum()
print(Test_survived_count)
print(Pred_survived_count)