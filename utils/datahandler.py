import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataHandler():
    def __init__(self) -> None:
        self.df = None
        self.scaler = StandardScaler()


    def load_data(self, path=r"C:\Users\Stanisław\Programowanie\GitHub\xai\healthcare-dataset-stroke-data.csv") -> None:
        #Wczytanie zbioru danych
        self.df = pd.read_csv(path)

    
    def fill_values(self) -> None:
        #Uzupełnienie wartości pustych, w tym przypadku jest to średnia
        self.df.fillna(value=self.df['bmi'].mean(), inplace=True)


    def standarization(self) -> None:
        #Standaryzacja danych zgodnie ze StandardScaler
        self.df = pd.concat([pd.DataFrame(self.scaler.fit_transform(self.df.iloc[:, :-1]), columns=self.df.columns[:-1]), 
                             self.df.iloc[:, -1]], 
                             axis=1)
        self.df.reset_index(inplace=True, drop=True)


    def preprocess_data(self, standarization=False) -> None:
        #Przygotowanie danych do uczenia, get_dummies działa zamiast OneHotEncoder
        #Jest możliwość dodania standaryzacji, ale on psuje wizualizacje. Nie można wtedy sprawdzić jaki wiek był w danym rekordzie
        self.df.drop(["id"], axis=1, inplace=True)

        cat_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        ohe = pd.get_dummies(self.df[cat_columns], prefix=cat_columns, dtype=int)

        self.df = pd.concat([ohe, self.df], axis=1)
        self.df.drop(cat_columns, axis=1, inplace=True)

        if standarization:
            self.standarization()

            
    def prepare_data(self, standarization=False) -> None:
        #Główna funkcja do przygotowania danych
        self.preprocess_data(standarization)
        self.fill_values()
    

    def reverse_standarization(self, data: pd.DataFrame) -> pd.DataFrame:
        df_reverse =pd.concat([pd.DataFrame(self.scaler.inverse_transform(data), columns=data.columns)],
                             axis=1)
        return df_reverse


    def get_data_split(self, test_size=0.2, seed=2021) -> list:
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        return train_test_split(X, y, test_size=test_size, random_state=seed)


    def get_data(self) -> pd.DataFrame:
        return self.df