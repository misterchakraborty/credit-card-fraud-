import pandas as pd

from sklearn.preprocessing  import StandardScaler     # used for standarizing the amount

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# loading dataset
df = pd.read_csv("creditcard_2023.csv")  #df is data frame

print(df.head())
print(df.info())
print(df.columns)


print(df.isnull().sum())  # check if any null values(isnull) & sum() gives you the count of it
df = df.dropna()  #removes the rows with missing values


scaler = StandardScaler()   # instance of the standard scaller class
df['Normalized_Amount'] = scaler.fit_transform(df[['Amount']])    

df = df.drop(['Amount'] , axis=1)   # drop Amount column

df = df.drop(['id'] , axis=1) # drom id column

x = df.drop('Class' , axis=1)   # x = features  
y = df['Class']   # target value

x_train , x_test , y_train , y_test = train_test_split(x ,y , test_size=0.2 , random_state=42)

smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

