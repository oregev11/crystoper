import os
import json 

def make_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def dump_json(obj, path):
    
    with open(path, 'w') as f:
        return json.dump(obj, f)