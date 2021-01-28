import pandas as pd
from sklearn.preprocessing import LabelEncoder



    # Create a function that converts all values of df['score'] into numbers
def Encoder(df):
          columnsToEncode = list(df.select_dtypes(include=['category','object']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
          return df
      
        
    # Create dataframe
city_data = {'city_level': [1, 3, 1, 2, 2, 3, 1, 1, 2, 3],'city_pool' : ['y','y','n','y','n','n','y','n','n','y'],
                'Rating': [1, 5, 3, 4, 1, 2, 3, 5, 3, 4],
                'City_port': [0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                'city_temperature': ['low', 'medium', 'medium', 'high', 'low','low', 'medium', 'medium', 'high', 'low']}

df = pd.DataFrame(city_data, columns = ['city_level', 'city_pool', 'Rating', 'City_port', 'city_temperature'])
print(); print(df)
df = Encoder(df)

print(); print(df)