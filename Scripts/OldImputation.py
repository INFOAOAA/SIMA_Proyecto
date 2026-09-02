import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split
from sklearn.impute import IterativeImputer
import pandas as pd



imp = IterativeImputer(max_iter=10, random_state=42)
df = pd.read_csv('BD_2022_Completo.csv')
df = df.drop(['Fecha y hora', 'Estacion'],axis = 1)
train, test = train_test_split(df, test_size=0.2, random_state=42)
imp.fit(train)
print(np.round(imp.transform(test)))

test_imputed = pd.DataFrame(
    np.round(imp.transform(test)), 
    columns=test.columns, 
    index=test.index
)

test_imputed.to_csv('BD_2022_test_imputed.csv', sep=',')