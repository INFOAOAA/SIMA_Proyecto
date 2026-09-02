import pandas as pd
import os

path = os.path.abspath('BasesDeDatosCsv/Limpia/BD_2022_Limpia.csv')
df = pd.read_csv(path)
date = pd.to_datetime(df['Fecha y hora'], yearfirst=True)

df = pd.DataFrame(
    {
        "datetime": pd.date_range(
            start="2020-01-01 00:00:00",
            end="2026-1-1 00:00:00",
            freq="1D",  # Example frequency (change to 'D', 'h', 's', etc.)
        )
    }
)

months = df["datetime"].dt.month

bins = [0, 2, 5, 8, 11, 12]
labels = ["Winter", "Spring", "Summer", "Fall", "Winter"]

df["season"] = pd.cut(
    months, bins=bins, labels=labels, ordered=False, include_lowest=True
)


print(df.tail())