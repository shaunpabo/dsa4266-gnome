import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
import math
from utils import parse_data
from multiprocessing import cpu_count

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

    # FEATURE SELECTION
    min_features_to_select = 1

    # create a RF model
    clf_RF = RandomForestClassifier(random_state=4266)

    # Recursively eliminate features with cross validation
    rfecv = RFECV(estimator=clf_RF, cv=5, scoring='roc_auc', n_jobs=-1, verbose=10, step=1, min_features_to_select= min_features_to_select)
    rfecv.fit(X_train, y_train)

    print("Optimal number of features : %d" % rfecv.n_features_)

    X_train_new = rfecv.transform(X_train)
    print("Num Features Before:", X_train.shape[1])
    print("Num Features After:", X_train_new.shape[1])

    X_train_new = X_train.iloc[:, rfecv.support_]
    rfecv_features = X_train_new.columns.tolist()
    # save rfecv features
    pickle.dump(rfecv_features, open("./weights/rfecv_features.pkl", "wb"))
    X_train_new = X_train.loc[:, rfecv_features]
    X_test_new = X_test.loc[:, rfecv_features]
    X_val_new = X_val.loc[:,rfecv_features]

    ## MODEL
    
    clf = RandomForestClassifier(random_state=4266)

    params = {
        'bootstrap': [True, False],
        'max_depth': [3, 5, 7, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [1200, 1400, 1600, 1800, 2000]
        }

    num_cores = cpu_count()
    print(f"Gridsearching with {num_cores-2}...")
    gscv_model = GridSearchCV(clf, param_grid = params, verbose = 2, cv=5, scoring = 'roc_auc', n_jobs=num_cores-2)
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