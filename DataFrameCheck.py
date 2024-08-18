import pandas as pd

class DataFrameCheck:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def dim(self):
        print(f"El DataFrame tiene {self.df.shape[0]} filas y {self.df.shape[1]} columnas.")
 
    
    def column_name(self):
        print("Los nombres de las columnas son:", list(self.df.columns))

    def duplicated_shown(self):
        print(f" Filas duplicadas: ")
        listof=self.df.duplicated().index.tolist()
        # Filter the elements of columns  where the index is not in the listof
        filtered_column = self.df.loc[~self.df.index.isin(listof)]
        print(filtered_column)

  
    def sample(self):
        print("Primeros 5 elementos del DataFrame:")
        print(self.df.head(5))
        print("\nÚltimos 5 elementos del DataFrame:")
        print(self.df.tail(5))
        print("\nMuestreo aleatorio de 5 elementos:")
        print(self.df.sample(5))

    
    def var_numeric(self):
        print("Descripción rápida de las variables numéricas del DataFrame:")
        print(self.df.describe())

    
    def nr_nulls(self):
        print("Cantidad de valores nulos por columna:")
        print(self.df.isnull().sum())
        
    
    def nr_unique(self):
        print("Cantidad de valores únicos por columna:")
        print(self.df.nunique())
       
    
    def nr_duplicate(self):
        print(f"El DataFrame tiene {self.df.duplicated().sum()} filas duplicadas.")
       

       


