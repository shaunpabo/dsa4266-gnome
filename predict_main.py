import pandas as pd
import pickle
import argparse
import os
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from multiprocessing import get_context, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from predict_utils import *

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Prediction script for DSA4266 Team Gnome')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test_sample", action="store_true")
    group.add_argument("--sgnex", action="store_true")
    args = parser.parse_args()
    if args.test_sample:
        print("Test predictions")
        # PARSE DATA
        data = pd.read_csv('./data/stdtestset.csv')
        y_data = data.label

        # PREPROCESSING

        # rfecv features
        rfecv_features = pickle.load(open("./weights/xgb_rfecv_features.pkl", "rb"))
        data = data.loc[:, rfecv_features]

        # Load model
        best_model = pickle.load(open("./weights/xgbmodelgs.pkl",'rb'))

        # PREDICT
        y_pred = best_model.predict(data)
        y_pred_proba = best_model.predict_proba(data)

        data['y_pred'] = y_pred
        data['class_0_proba'] = y_pred_proba[:, 0]
        data['class_1_proba'] = y_pred_proba[:, 1]
        data.to_csv("./output/test_data_predictions.csv", index=False)
        print("Predictions saved to ./output/test_data_predictions.csv")

        auc_score = roc_auc_score(y_data, y_pred_proba[:,1])
        ap = average_precision_score(y_data, y_pred_proba[:,1])

        print(f"Test Acc: {accuracy_score(y_data, y_pred):.6f} | Test AUC-ROC {auc_score:.6f} | Test PR-ROC {ap:.6f}")

    else:
        all_folders = os.listdir("./m6anet")


        # report a message
        context = get_context('fork')
        print(f'Starting predict task with 6 cpu cores')
        # create the process pool
        with ProcessPoolExecutor(max_workers=6, mp_context=context) as executor:
            # perform calculations
            df_list = []
            futures = [executor.submit(task, folder) for folder in all_folders]
            for future in as_completed(futures):
                df_list.append(future.result())

        df = pd.concat(df_list, ignore_index=True)
        df.to_csv("./output/compiled_df.csv", index=False)
        # report a message
        print('Combined csv saved to ./output/compiled_df.csv')