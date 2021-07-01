"""Temporary script to unpickle and json the minmaxscaler data"""
import pathlib
import json
import pickle
scaler_paths = list(pathlib.Path('./data/rad').glob('*.p'))

for scaler_path in scaler_paths:
    with open(str(scaler_path),"rb") as f:
        scaler = pickle.load(f)
    
    scaler_json = {}
    scaler_json["x_scale"] = list(scaler.scale_)
    scaler_json["x_min"] = list(scaler.min_)

    json_path = str(scaler_path.parent) +"/"+ scaler_path.stem + ".json"
    with open(json_path, 'w') as f:
        json.dump(scaler_json, f)
