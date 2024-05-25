import pandas as pd
import numpy as np
import sqlite3

start_date = "2021-01-01"
end_date = "2024-05-23"
date_range = pd.date_range(start=start_date, end=end_date, freq='H')

data = pd.DataFrame(date_range, columns=['datetime'])

def generate_co2_levels(hour, base_level=900):
    
    day_variation = 150  # Вдень коливання в межах +/- 150
    night_variation = 300  # Вночі коливання в межах +/- 300
    
    # Закономірності для CO2
    if 6 <= hour < 9:
        # Ранкове різке зниження
        return base_level - np.random.randint(0, night_variation)
    elif 9 <= hour < 18:
        # Вдень коливання в межах
        return base_level + np.random.randint(-day_variation, day_variation)
    else:
        # Вночі зростання
        return base_level + np.random.randint(0, night_variation)

data['CO2Level'] = data['datetime'].apply(lambda dt: generate_co2_levels(dt.hour))

# Корекція середнього значення до 900
mean_co2 = data['CO2Level'].mean()
adjustment = 900 - mean_co2
data['CO2Level'] = (data['CO2Level'] + adjustment).astype(int)

# Підрахунок кількості записів
num_records = len(data)

# Вивід середнього значення та кількості записів
print("Середнє значення CO2:", data['CO2Level'].mean())
print("Кількість записів:", num_records)

# Підключення до SQLite бази даних (створення нової бази даних)
conn = sqlite3.connect('co2_levels.db')
cursor = conn.cursor()

# Створення таблиці
cursor.execute('''
CREATE TABLE IF NOT EXISTS co2_levels (
    datetime TEXT PRIMARY KEY,
    CO2Level INTEGER
)
''')

# Додавання даних до таблиці
data.to_sql('co2_levels', conn, if_exists='replace', index=False)
conn.close()
