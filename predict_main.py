import pandas as pd
import pickle
import argparse
import os
import sys
from multiprocessing import get_context, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from predict_utils import *

def task(i):
    print("parsing file" + i)
    json_file = "./m6anet/" + i + "/data.json"
    info_file = "./m6anet/" + i + "/data.info"
    cancer = i.split('_')[1]

    data = parse_data(info_file, json_file)

    with open("./weights/rfecv_features.pkl","rb") as f:
        rfecv_features = pickle.load(f)
    
    identifier = ["transcript_id", "transcript_position", "start", "end", "n_reads"]

    data = data.loc[:,identifier + rfecv_features]

    with open("./weights/xgbmodel.pkl","rb") as f:
        clf_xgb = pickle.load(f)
    scaled_data = scale_data(data, rfecv_features, identifier)

    y_hat = get_predict_col(scaled_data, clf_xgb, rfecv_features)
    # del scaled_data
    print("Concatenating prediction column...")
    df = pd.concat([pd.DataFrame({"cell_line": [cancer] * len(data) }), data, y_hat], axis = 1)

    print("Completed job" + i)
    return df

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Prediction script for DSA4266 Team Gnome')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test_sample", action="store_true")
    group.add_argument("--sgnex", action="store_true")
    args = parser.parse_args()
    if args.test_sample:
        # PARSE DATA
        data = parse_test_data("./data/dataset2.json.gz")
        # TODO: read sample data rather than parse test data

        # PREPROCESSING
        features = ['dwell_time_-1', 'sd_-1', 'mean_-1', 'dwell_time_0', 
                    'sd_0', 'mean_0', 'dwell_time_1', 'sd_1', 'mean_1', 'nucleo_-3_A', 
                    'nucleo_-3_C', 'nucleo_-3_G', 'nucleo_-3_T', 'nucleo_-2_A', 
                    'nucleo_-2_G', 'nucleo_-2_T', 'nucleo_-1_G', 'nucleo_-1_A', 
                    'nucleo_1_C', 'nucleo_2_C', 'nucleo_2_T', 'nucleo_2_A', 'nucleo_3_A', 
                    'nucleo_3_G', 'nucleo_3_T', 'nucleo_3_C', 'pwm_score', 'knf_AA',
                    'knf_AG', 'knf_AT', 'knf_AC', 'knf_GG', 'knf_GA', 'knf_GT', 'knf_GC',
                    'knf_TT', 'knf_TA', 'knf_TG', 'knf_TC', 'knf_CC', 'knf_CA', 'knf_CT',
                    'knf_CG', 'cksnap_AA', 'cksnap_AG', 'cksnap_AT', 'cksnap_AC',
                    'cksnap_GG', 'cksnap_GA', 'cksnap_GT', 'cksnap_GC', 'cksnap_TT',
                    'cksnap_TA', 'cksnap_TG', 'cksnap_TC', 'cksnap_CC', 'cksnap_CA',
                    'cksnap_CT', 'cksnap_CG', 'dacc_bet', 'js_all', 'eiip']
        
        data = data[features]
        data.rename(columns={'dwell_time_-1': 'dwelling_t-1', 
                            'dwell_time_0': 'dwelling_t0', 
                            'dwell_time_1': 'dwelling_t1'}, inplace=True)

        # load scaler
        scaler = pickle.load(open("./weights/scaler.pkl", "rb"))
        data = pd.DataFrame(scaler.transform(data), columns = data.columns)

        # rfecv features
        rfecv_features = pickle.load(open("./weights/rfecv_features.pkl", "rb"))
        data = data.loc[:, rfecv_features]

        # Load model
        best_model = pickle.load(open("./weights/rfmodelgs.pkl",'rb'))

        y_pred = best_model.predict(data)
        y_pred_proba = best_model.predict_proba(data)

        data['y_pred'] = y_pred
        data['class_0_proba'] = y_pred_proba[:, 0]
        data['class_1_proba'] = y_pred_proba[:, 1]
        data.to_csv("./output/dataset1_predictions.csv", index=False)

    else:
        all_folders = os.listdir("./m6anet")

        num_cores = cpu_count()

        # report a message
        context = get_context('fork')
        print(f'Starting predict task with {num_cores} cpu cores')
        # create the process pool
        with ProcessPoolExecutor(max_workers=num_cores, mp_context=context) as executor:
            # perform calculations
            df_list = []
            futures = [executor.submit(task, folder) for folder in all_folders]
            for future in as_completed(futures):
                df_list.append(future.result())

        df = pd.concat(df_list, ignore_index=True)
        df.to_csv("./output/compiled_df.csv", index=False)
        # report a message
        print('Combined csv saved to ./output/compiled_df.csv')