from pathlib import Path
import json
import pandas as pd
import numpy as np
from utils import csv_writer

thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")


# Opening JSON file
json_files = [i for i in datadir.rglob("*.json") if "lung_data" in str(i)]

he_csv = np.squeeze(pd.read_csv(Path(datadir / "lung_data" / "he_images.csv")).to_numpy())

clean_he_csv = [i[4:-4] for i in he_csv]

he_json = []
for file in json_files:
    f = open(file)
    data =json.load(f)
    matches = [i for i in data.keys() if i.split("_")[0] in clean_he_csv]
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
    outputdir = Path(datadir.parent / "csv_files_extra")
    print(f"Found repeated values in JSON files. Saved in .csv file in {datadir.parent}")

else:
    print("Not repeated values found on the JSON files")

    # Eliminate images repeated between JSON files
    he_json = np.unique(he_json)

    # Obtain levels from JSON files from images contained in the he_images.csv
    labels_he = {}
    for file in json_files:
        f = open(file)
        data =json.load(f)
        for num, labels in zip(data, data.values()):
            if num in he_json:
                labels_he[num] = labels
                    
        f.close()


    labels_df = pd.DataFrame.from_dict(labels_he, orient="index") 
    for col, row in labels_df.iterrows():
        for name in he_csv:
            if col.split("_")[0] in name:
                labels_df.rename(index={col: name}, inplace=True)
    labels_df.to_csv(Path(datadir / "labels.csv"), header=True, index_label="image_num")



    print(f"Labels from JSON files created in {datadir} for HE ink")

