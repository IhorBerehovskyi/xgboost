import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Підключення до SQLite бази даних
conn = sqlite3.connect('co2_levels3.db')

# Зчитування даних з таблиці
data = pd.read_sql_query("SELECT * FROM co2_levels", conn)

# Закриття з'єднання
conn.close()

# Перетворення колонки 'datetime' в формат datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Вибір останніх 50 записів
last_50_data = data.tail(250)

# Переконайтеся, що колонки 'datetime' та 'CO2Level' є одновимірними масивами
datetimes = last_50_data['datetime'].values
co2_levels = last_50_data['CO2Level'].values

# Побудова графіка
plt.figure(figsize=(12, 6))
plt.plot(datetimes, co2_levels, label='Рівень CO2')
plt.xlabel('Дата та час')
plt.ylabel('Рівень CO2')
plt.title('Рівень CO2 (останні 50 записів)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Відображення графіка
plt.show()

