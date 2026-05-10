import pandas as pd

df = pd.read_csv('data/train.csv')
df = df.rename(
    columns = {
        'id': 'CustomerID',
        'gender': 'Gender',
        'tenure': 'Tenure'
    }
)
reference = df.sample(1000, random_state = 42)
reference = reference.drop(
    columns = ['CustomerID', 'Churn']
)
reference.to_csv('data/reference.csv', index = False)