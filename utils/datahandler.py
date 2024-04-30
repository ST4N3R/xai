import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataHandler():
    def __init__(self) -> None:
        self.df = None


    def load_data(self, path=r"C:\Programowanie\GitHub\xai\healthcare-dataset-stroke-data.csv") -> None:
        self.df = pd.read_csv(path)

    
    def fill_values(self) -> None:
        self.df.fillna(value=self.df['bmi'].mean(), inplace=True)


    def preprocess_data(self) -> None:
        cat_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        ohe = pd.get_dummies(self.df[cat_columns], prefix=cat_columns)

        self.df = pd.concat([self.df, ohe], axis=1)
        self.df.drop(cat_columns, axis=1, inplace=True)
        self.df.drop(["id"], axis=1, inplace=True)


    def standarization(self) -> None:
        self.preprocess_data()
        self.fill_values()

        scaler = StandardScaler()
        self.df = pd.concat([pd.DataFrame(scaler.fit_transform(self.df.iloc[:, :-1]), columns=self.df.columns[:-1]), 
                             self.df.iloc[:, -1]], 
                             axis=1)
        self.df.reset_index(inplace=True, drop=True)
    

    def get_data_split(self, test_size=0.2, seed=2021) -> list:
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        return train_test_split(X, y, test_size=test_size, random_state=seed)


    def get_data(self) -> pd.DataFrame:
        return self.df