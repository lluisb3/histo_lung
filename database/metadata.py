from pathlib import Path
import json
import pandas as pd
import numpy as np
from utils import csv_writer

thispath = Path(__file__).resolve()

datadir =Path(thispath.parent.parent / "data")


# Opening JSON file
json_files = [i for i in datadir.rglob("*.json") if "lung_data" in str(i)]

he_csv = np.squeeze(pd.read_csv(Path(datadir / "lung_data" / "he_images.csv")).to_numpy())

clean_he_csv = [i[4:-4] for i in he_csv]

he_json = []
for file in json_files:
    f = open(file)
    data =json.load(f)
    keys = np.fromiter(data.keys(), dtype='U14')
    matches = [i for i in keys if i in clean_he_csv]
    he_json.extend(matches)
    f.close()

if len(he_json) > len(clean_he_csv):
    he_json = np.array(he_json)
    m = np.zeros_like(he_json, dtype=bool)
    m[np.unique(he_json, return_index=True)[1]] = True

    header = ["This images are repeated in JSON files"]
    csv_writer(datadir, "repeated_in_json.csv", "w", header)
    for a, num in zip(m, he_json):
        if a == False:
            csv_writer(datadir, "repeated_in_json.csv", "a", [num])
    print(a)

# Eliminate images repeated between JSON files
he_json = np.unique(he_json)
