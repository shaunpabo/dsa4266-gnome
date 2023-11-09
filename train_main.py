import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
import math
from utils import parse_data

if __name__=='__main__':
    # Read and parse data
    X_train = pd.read_csv("../data/stdtrainset.csv",index_col=0)
    X_test = pd.read_csv("../data/stdtestset.csv",index_col=0)
    X_val = pd.read_csv("../data/stdvalset.csv", index_col=0)

    y_train = X_train.label
    y_test = X_test.label
    y_val = X_val.label

    X_train = X_train.drop(columns=["label"])
    X_test = X_test.drop(columns=["label"])
    X_val = X_val.drop(columns=["label"])

    # Feature scaling
    features = ['dwelling_t-1', 'sd_-1', 'mean_-1', 'dwelling_t0', 'sd_0', 
                'mean_0', 'dwelling_t1', 'sd_1', 'mean_1', 'nucleo_-3_A', 
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
    
    # X_train = train[features]
    # X_val = val[features]
    # X_test = test[features]

    # scaler = StandardScaler()
    # X_train = pd.DataFrame(scaler.fit_transform(X_train), columns =X_train.columns)
    # y_train = train["label"]

    # # pickle scaler
    # pickle.dump(scaler, open("./weights/rf_scaler.pkl", "wb"))

    # X_val = pd.DataFrame(scaler.transform(X_val), columns = X_val.columns)
    # y_val = val["label"]

    # X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
    # y_test = test["label"]

    # FEATURE SELECTION
    # min_features_to_select = 1

    # # To account for weight imbalances
    # scale_pos_weight = math.sqrt(y_train.value_counts().values[0]/y_train.value_counts().values[1])

    # # create a RF model
    # clf_RF = RandomForestClassifier(random_state=4266)

    # # Recursively eliminate features with cross validation
    # rfecv = RFECV(estimator=clf_RF, cv=5, scoring='roc_auc', n_jobs=-1, verbose=10, step=1, min_features_to_select= min_features_to_select)
    # rfecv.fit(X_train, y_train)

    # print("Optimal number of features : %d" % rfecv.n_features_)

    # X_train_new = rfecv.transform(X_train)
    # print("Num Features Before:", X_train.shape[1])
    # print("Num Features After:", X_train_new.shape[1])

    # X_train_new = X_train.iloc[:, rfecv.support_]
    # rfecv_features = X_train_new.columns.tolist()
    # # save rfecv features
    # pickle.dump(rfecv_features, open("./weights/rfecv_features.pkl", "wb"))
    rfecv_features = pickle.load(open("./weights/rfecv_features.pkl", "rb"))
    X_train_new = X_train.loc[:, rfecv_features]
    X_test_new = X_test.loc[:, rfecv_features]
    X_val_new = X_val.loc[:,rfecv_features]

    # To account for weight imbalances
    # scale_pos_weight = math.sqrt(y_train.value_counts().values[0]/y_train.value_counts().values[1])

    ## MODEL

    # create a XGB model
    # clf = xgb.XGBClassifier(random_state=4266, colsample_bytree = 0.8, colsample_bynode = 0.8, colsample_bylevel = 0.8, use_label_encoder = False,
    #                     eval_metric = "auc", objective = "binary:logistic", scale_pos_weight = scale_pos_weight, n_estimators = 200)
    
    clf = RandomForestClassifier(random_state=4266)

    params = {
        'bootstrap': [True, False],
        # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
        }

    print("Gridsearching...")
    gscv_model = GridSearchCV(clf, param_grid = params, verbose = 2, cv=5, scoring = 'roc_auc', n_jobs=-1)
    gscv_model.fit(X_train_new,y_train)
    print("Gridsearch complete")
    pickle.dump(gscv_model.best_estimator_,open("./weights/rfmodelgs.pkl", "wb"))
    print("Best model saved to ./weights/rfmodelgs.pkl")

    y_pred = gscv_model.predict(X_val_new)
    y_pred_proba = gscv_model.predict_proba(X_val_new)
    auc_score = roc_auc_score(y_val, y_pred_proba[:,1])
    ap = average_precision_score(y_val, y_pred_proba[:,1])

    print(f"Val Acc: {accuracy_score(y_val, y_pred)} | Val AUC-ROC {auc_score} | Val PR-ROC {ap}")

    y_pred = gscv_model.predict(X_test_new)
    y_pred_proba = gscv_model.predict_proba(X_test_new)
    auc_score = roc_auc_score(y_test, y_pred_proba[:,1])
    ap = average_precision_score(y_test, y_pred_proba[:,1])

    print(f"Test Acc: {accuracy_score(y_test, y_pred)} | Test AUC-ROC {auc_score} | Test PR-ROC {ap}")