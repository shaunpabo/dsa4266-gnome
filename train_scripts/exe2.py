import sys

sys.path.append('studies/ProjectStorage/LeeZiJie/')

from utils2 import *
import os
import pandas as pd
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from multiprocessing import get_context
import random


def task(i):
    file = open("studies/ProjectStorage/LeeZiJie/xgb_full.pkl","rb")
    clf_xgb = pickle.load(file)
    print("parsing file" + i)
    json_file = "samples/" + i + "/data.json"
    info_file = "samples/" + i + "/data.info"
    cancer = i.split('_')[1]

    data = parse_data(info_file, json_file)

    rfecv_features = ['sd_-1',
 'mean_-1',
 'dwelling_t0',
 'sd_0',
 'mean_0',
 'dwelling_t1',
 'sd_1',
 'mean_1',
 'nucleo_-3_A',
 'nucleo_-3_G',
 'nucleo_-3_T',
 'nucleo_-2_A',
 'nucleo_-2_G',
 'nucleo_-1_G',
 'nucleo_2_C',
 'nucleo_2_T',
 'nucleo_2_A',
 'nucleo_3_A',
 'nucleo_3_G',
 'pwm_score',
 'knf_AA',
 'knf_AC',
 'knf_GG',
 'knf_TT',
 'knf_TA',
 'knf_TG',
 'knf_CA',
 'knf_CT',
 'cksnap_AC',
 'cksnap_GA',
 'cksnap_GT',
 'cksnap_GC',
 'js_all',
 'eiip']
    
    
    identifier = ["transcript_id", "transcript_position", "start", "end", "n_reads"]

    data = data.loc[:,identifier + rfecv_features]

    scaled_data = scale_data(data, rfecv_features, identifier)

    y_hat = get_predict_col(scaled_data, clf_xgb, rfecv_features)

    data = data[data['transcript_id'].str.contains("ENST")]
    del scaled_data
    sample_prop = {"A549": 0.2,
                   "Hct116": 2/15,
                   "HepG2": 0.2,
                   "K562": 2/15,
                   "MCF7": 0.2}

    print("Concatenating prediction column...")
    #df = pd.concat([pd.DataFrame({"cell_line": [cancer] * len(data) }), data, y_hat], axis = 1).sample(frac= sample_prop[cancer])
    df = pd.concat([pd.DataFrame({"cell_line": [cancer] * len(data) }), data, y_hat], axis = 1)
    print("Completed job" + i)
    return df


if __name__ == '__main__':
    # report a message
    context = get_context('fork')
    print('Starting task...')
    df = pd.DataFrame()
    all_folders = os.listdir("samples")
    # create the process pool
    with ProcessPoolExecutor(max_workers=6, mp_context=context) as executor:
        # perform calculations

        futures = [executor.submit(task, folder) for folder in all_folders]
        
        for future in as_completed(futures):
            df = pd.concat([df, future.result()], axis = 0)
            
        executor.shutdown()
    transcript_sets = []
    cancer = ["A549", "Hct116", "HepG2", "K562", "MCF7"]
    for i in cancer:
        transcript_sets.append(set(df[df['cell_line'] == i]['transcript_id']))
    transcript_set_intersect = transcript_sets[0]
    for j in range(1, len(transcript_sets)):
        print(len(transcript_set_intersect))
        transcript_set_intersect = transcript_set_intersect.intersection(transcript_sets[j])
    n_transcript = len(transcript_set_intersect)
    sample_transcript = pd.Series(list(transcript_set_intersect))[random.sample(range(0, n_transcript), math.floor(0.4*n_transcript))]
    df = df[df['transcript_id'].isin(sample_transcript)]
    df.to_csv("studies/ProjectStorage/LeeZiJie/compiled_df.csv", index = False)
    # report a message
    print('Done.')  
