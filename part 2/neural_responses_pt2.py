import pandas as pd
import numpy as np
import os

def file_path(file_path):
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir, file_path)
    return abs_file_path

