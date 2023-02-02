from pathlib import Path
import json
import pandas as pd
import numpy as np
import utils

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

he_json = np.array(he_json)
# he_json = [int(i) for i in he_json]
m = np.zeros_like(he_json, dtype=bool)
m[np.unique(he_json, return_index=True)[1]] = True

header = ["this bitches"]
csv_writer(datadir, "repetead_in_json", "w", header)
estos = []
for a, num in zip(m, he_json):
    if a == False:
        estos.append(num)
        csv_writer(datadir, "repeated_in_json", "a", num)


# returns JSON object as a dictionary
data = json.load(f)
  
# Iterating through the json
# list
# for i in data['emp_details']:
#     print(i)
  
# Closing file
f.close()