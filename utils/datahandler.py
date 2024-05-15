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


    def standarization(self) -> None:
        scaler = StandardScaler()
        self.df = pd.concat([pd.DataFrame(scaler.fit_transform(self.df.iloc[:, :-1]), columns=self.df.columns[:-1]), 
                             self.df.iloc[:, -1]], 
                             axis=1)
        self.df.reset_index(inplace=True, drop=True)


    def preprocess_data(self, ohe = False) -> None:
        self.df.drop(["id"], axis=1, inplace=True)
        if ohe:
            cat_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            ohe = pd.get_dummies(self.df[cat_columns], prefix=cat_columns)

            self.df = pd.concat([self.df, ohe], axis=1)
            self.df.drop(cat_columns, axis=1, inplace=True)

            self.standarization()
        else:
            self.df['gender'] = self.df['gender'].replace({"Male": 0, "Female": 1, "Other": -1})
            self.df['ever_married'] = self.df['ever_married'].replace({"Yes": 1, "No": 0})
            self.df['work_type'] = self.df['work_type'].replace({'Private': 0, 
                                                                 'Self-employed': 1, 
                                                                 'Govt_job': 2, 
                                                                 'children': -1, 
                                                                 'Never_worked': -2})
            self.df['Residence_type'] = self.df['Residence_type'].replace({"Urban": 0, "Rural": 1})
            self.df['smoking_status'] = self.df['smoking_status'].replace({'formerly smoked': 0, 
                                                                           'never smoked': 1, 
                                                                           'smokes': -1, 
                                                                           'Unknown': 2})

            
    def prepare_data(self, ohe = False) -> None:
        self.preprocess_data(ohe)
        self.fill_values()
    

    def get_data_split(self, test_size=0.2, seed=2021) -> list:
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        return train_test_split(X, y, test_size=test_size, random_state=seed)


    def get_data(self) -> pd.DataFrame:
        return self.df