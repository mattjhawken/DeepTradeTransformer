import pandas as pd
import os


def update_data(id: str):
    path = f"data/{id}"

    if os.path.exists(path):



def fetch_data(id: str):
    path = "data/" + id

    if os.path.exists(path):
        data = pd.read_csv(path)
        data["Date"] = pd.to_datetime(data.index, errors="coerce")
