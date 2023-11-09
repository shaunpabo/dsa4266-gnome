import pandas as pd
import numpy as np
import gzip
import json
import pickle
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

#########################
# Function for features #
#########################

def create_pwm_from_sequences(sequences):
    """
    Create a Position Weight Matrix (PWM) from a list of sequences.

    Args:
        sequences (list of str): List of DNA sequences of equal length.

    Returns:
        np.ndarray: The PWM matrix.
    """
    # Check if all sequences have the same length
    seq_length = len(sequences[0])
    if not all(len(seq) == seq_length for seq in sequences):
        raise ValueError("All sequences must have the same length")

    # Initialize the PWM matrix with zeros
    pwm = np.zeros((4, seq_length))  # 4 rows for A, C, G, T

    # Count the occurrences of each nucleotide at each position
    for seq in sequences:
        for i, nucleotide in enumerate(seq):
            if nucleotide == 'A':
                pwm[0, i] += 1
            elif nucleotide == 'C':
                pwm[1, i] += 1
            elif nucleotide == 'G':
                pwm[2, i] += 1
            elif nucleotide == 'T':
                pwm[3, i] += 1

    # Convert counts to probabilities
    pwm = pwm / len(sequences)

    return pd.DataFrame(data=pwm, columns=[1,2,3,4,5,6,7], index=["A","C","G","T"])

def log_odds(x):
    """
    Function to calculate log odds
    """
    if x == 0:
        return 0
    else:
        return np.log2(x/0.25)


def get_log_odds(sequences):
    """To get log odd dictionary which is used to calculate PWM score for each nucleo_seq"""
    ppm = create_pwm_from_sequences(sequences)
    log_odds_pos = ppm.applymap(log_odds)
    log_odds_dict = log_odds_pos.to_dict()

    return log_odds_dict, ppm


def get_PWM_score(seq,log_odds_dict):
    """Calculating PWM score for nucleo_seq based on the log odds"""
    res = 0
    for i in range(len(seq)):
        base = seq[i]
        dic = log_odds_dict[(i+1)]
        res = res + dic[base]
    return res

index_ls = ["AA", "AG", "AT", "AC", "GG", "GA", "GT", "GC", "TT", "TA", "TG", "TC", "CC", "CA", "CT", "CG"]

def get_knf(seq, k, index_ls=["AA", "AG", "AT", "AC", "GG", "GA", "GT", "GC", "TT", "TA", "TG", "TC", "CC", "CA", "CT", "CG"], prob = True, output_dic = True):
    dic = {}
    n_kmers = len(seq) - k + 1
    for i in range(n_kmers):
        kmer = seq[i: i + k]
        if kmer not in dic:
            dic[kmer] = 0
        dic[kmer] += 1
    if prob != True:
        return dic
    if output_dic != True:
        ls = np.zeros(len(index_ls))
        for k, v in dic.items():
            ls[index_ls.index(k)] = v
        return ls/n_kmers
    return {k: v/n_kmers for k, v in dic.items()}

def get_cksnap(seq, k, index_ls=["AA", "AG", "AT", "AC", "GG", "GA", "GT", "GC", "TT", "TA", "TG", "TC", "CC", "CA", "CT", "CG"], output_dic = True):
    dic = {}
    n_ks_mers = len(seq) - k - 1
    for i in range(n_ks_mers):
        kmer = seq[i: i + k + 2]
        ks_mer = kmer[0] + kmer[-1]
        if ks_mer not in dic:
            dic[ks_mer] = 0
        dic[ks_mer] += 1
    if output_dic != True:
        ls = np.zeros(len(index_ls))
        for k, v in dic.items():
            ls[index_ls.index(k)] = v
        return ls/n_ks_mers
    return {k: v/n_ks_mers for k, v in dic.items()}

def dacc(seq1, seq2):
    acc = 0
    seq1_knf = get_knf(seq1, 2, False)
    seq2_knf = get_knf(seq2, 2, False)
    for i in range(len(seq1) - 3):
        acc += seq1_knf[seq1[i:i+2]]*seq2_knf[seq2[i+2: i+4]] + seq1_knf[seq1[i+2: i+4]]*seq2_knf[seq2[i:i+2]]
    return acc



def jaccard_similarity(kmer_ls, sliced):
    a = set(get_knf(kmer_ls[0], sliced, False))
    b = set(get_knf(kmer_ls[1], sliced, False))
    c = set(get_knf(kmer_ls[2], sliced, False))
    intersection_ab = len(a.intersection(b))
    union_ab = len(a.union(b))
    intersection_ac = len(a.intersection(c))
    union_ac = len(a.union(c))
    intersection_bc = len(b.intersection(c))
    union_bc = len(b.union(c))

    return (intersection_ab/union_ab + intersection_ac/union_ac + intersection_bc/union_bc)/3

def eiip(seq):
    eiip_values = {
    'A': 0.1260,
    'C': 0.1340,
    'G': 0.0806,
    'T': 0.1335,
    }

    return np.mean([eiip_values[n] for n in seq])


##########################
# To parse training data #
##########################

def parse_data(info_path, json_path):
    # loads data
    print(f"Loading {json_path}...")
    with open(json_path, "r") as f:
        data = [json.loads(line) for line in f]

    # loads data with label
    print(f"Loading {info_path}...")
    info = pd.read_csv(info_path)

    #transfer information from json dict to list
    print("Transferring data from json to dataframe...")
    res = []
    for row in data:
        for trans_id in row.keys():
          for trans_pos in row[trans_id].keys():
              for nucleo_seq in row[trans_id][trans_pos].keys():
                  temp = list(np.mean(np.array(row[trans_id][trans_pos][nucleo_seq]), axis=0))
                  # to get raw data without aggregation
                  # for features in row[trans_id][trans_pos][nucleo_seq]:
                  res.append([trans_id, int(trans_pos), nucleo_seq] + temp)

    data = pd.DataFrame(res, columns = ['transcript_id', 'transcript_position', 'sequence',
                                       'dwelling_t-1', 'sd_-1', 'mean_-1',
                                       'dwelling_t0', 'sd_0', 'mean_0',
                                       'dwelling_t1', 'sd_1', 'mean_1'
                                       ])
    del res
   
    # Merge json data with labels
    print("Merging dataframes to obtain labels")
    data = pd.merge(data,info, on = ['transcript_id', 'transcript_position'])

    data = data.groupby(['transcript_id', 'transcript_position', 'sequence', "start", "end", "n_reads"]).mean(['dwelling_t-1', 'sd_-1', 'mean_-1',
                                                           'dwelling_t0', 'sd_0', 'mean_0',
                                                           'dwelling_t1', 'sd_1', 'mean_1']).reset_index()

    print("Creating features")
    # Get one hot encoding
    encoder = ce.OneHotEncoder(use_cat_names=True)
    print("one hot encoding")
    data = pd.concat([data,encoder.fit_transform(data['sequence'].str.split('', expand = True)[[1, 2, 3, 5, 6, 7]].rename(columns = {3: 'nucleo_-1', 5: 'nucleo_1',
                                                                                                      1: 'nucleo_-3', 2: 'nucleo_-2',
                                                                                                      6: 'nucleo_2', 7: 'nucleo_3'}))],axis=1)
    # Get pwm
    log_odds_dict, ppm = get_log_odds(data.sequence)
    data["pwm_score"] = data.apply(lambda x: get_PWM_score(x.sequence, log_odds_dict),axis=1)
    # Get knf
    data[["knf_" + i for i in index_ls]] = data['sequence'].apply(lambda x: get_knf(x, 2, output_dic = False)).apply(pd.Series)
    # Get cksnap
    data[["cksnap_" + i for i in index_ls]] = data['sequence'].apply(lambda x: get_cksnap(x, 2, output_dic = False)).apply(pd.Series)
    # Get dacc_bet
    data['dacc_bet'] = data['sequence'].apply(lambda x: dacc(list(get_knf(x, 5))[0], list(get_knf(x, 5))[2]))
    # Get jaccard similarity
    data['js_all'] = data['sequence'].apply(lambda x: jaccard_similarity(list(get_knf(x, 5)), 2))
    # Get eiip
    data["eiip"] = data.apply(lambda x: eiip(x.sequence), axis=1)

    return data

######################
# To parse test data #
######################

def parse_test_data(json_zip_path):
    # loads data
    print(f"Loading {json_zip_path}...")
    with gzip.open(json_zip_path, "r") as f:
        data = [json.loads(line) for line in f]


    #transfer information from json dict to list
    print("Transferring data from json to dataframe...")
    res = []
    for row in data:
        for trans_id in row.keys():
            for trans_pos in row[trans_id].keys():
                for nucleo_seq in row[trans_id][trans_pos].keys():
                    temp = list(np.mean(np.array(row[trans_id][trans_pos][nucleo_seq]), axis=0))
                    # to get raw data without aggregation
                    # for features in row[trans_id][trans_pos][nucleo_seq]:
                    res.append([trans_id, int(trans_pos), nucleo_seq] + temp)

    data = pd.DataFrame(res, columns = ['transcript_id', 'transcript_pos', 'sequence',
                                        'dwell_time_-1', 'sd_-1', 'mean_-1',
                                        'dwell_time_0', 'sd_0', 'mean_0',
                                        'dwell_time_1', 'sd_1', 'mean_1'
                                        ])
    # Get one hot encoding
    encoder = ce.OneHotEncoder(use_cat_names=True)
    data = pd.concat([data,encoder.fit_transform(data['sequence'].str.split('', expand = True)[[1, 2, 3, 5, 6, 7]].rename(columns = {3: 'nucleo_-1', 5: 'nucleo_1',
                                                                                                         1: 'nucleo_-3', 2: 'nucleo_-2',
                                                                                                         6: 'nucleo_2', 7: 'nucleo_3'}))],axis=1)
    # Get pwm
    log_odds_dict, ppm = get_log_odds(data.sequence)
    data["pwm_score"] = data.apply(lambda x: get_PWM_score(x.sequence, log_odds_dict),axis=1)
    # Get the 3 variations of 5mers
    data[["5mer_-1", "5mer_0", "5mer_1"]] = data['sequence'].apply(lambda x: list(get_knf(x, 5))).apply(pd.Series)
    # Get knf
    data[["knf_" + i for i in index_ls]] = data['sequence'].apply(lambda x: get_knf(x, 2, output_dic = False)).apply(pd.Series)
    # Get cksnap
    data[["cksnap_" + i for i in index_ls]] = data['sequence'].apply(lambda x: get_cksnap(x, 2, output_dic = False)).apply(pd.Series)
    # Get dacc_bet
    data['dacc_bet'] = data['sequence'].apply(lambda x: dacc(list(get_knf(x, 5))[0], list(get_knf(x, 5))[2]))
    # Get jaccard similarity
    data['js_all'] = data['sequence'].apply(lambda x: jaccard_similarity(list(get_knf(x, 5)), 2))
    # Get eiip
    data["eiip"] = data.apply(lambda x: eiip(x.sequence), axis=1)


    return data

###########
# Scaling #
###########

def scale_data(data, features, identifiers):
    print("Scaling variables...")
    scaler = StandardScaler()
    new_df_scaled = pd.DataFrame(scaler.fit_transform(data[features]), columns = data[features].columns)
    return pd.concat([data[identifiers], new_df_scaled], axis=1)

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb


def train_test_split(data, split):
    print("Splitting data into train/test...")
    splitter = GroupShuffleSplit(test_size= split, n_splits=2, random_state = 42)
    split = splitter.split(data, groups = data['gene_id'])
    train_idx, test_idx = next(split)

    df_train = data.iloc[train_idx].reset_index(drop=True)
    df_test = data.iloc[test_idx].reset_index(drop=True)

    return df_train, df_test


def feature_select(data_train, data_test, model, identifiers, label, min_features_to_select = 1):
    print("Selecting features...")
    id_cols = data_train[identifiers]
    X_train = data_train.drop(columns = identifiers)
    y_train = data_train[label]


    rfecv = RFECV(estimator = model, cv=5, scoring='roc_auc', n_jobs=-1, verbose=10, step=1, min_features_to_select= min_features_to_select)
    rfecv.fit(X_train, y_train)

    print("Optimal number of features : %d" % rfecv.n_features_)
    if data_test == False:
      return pd.concat([id_cols,X_train.iloc[:, rfecv.support_]], axis= 1), X_train.iloc[:, rfecv.support_].columns

    X_test = data_test.drop(columns = identifiers)
    y_test = data_test[label]
    return X_train.iloc[:, rfecv.support_], y_train, X_test.iloc[:, rfecv.support_], y_test, X_train.iloc[:, rfecv.support_].columns


def train_predict_label(X_train, y_train, X_test, y_test, model):
    print("Training model...")
    model.fit(X_train,y_train)

    print("Predicting labels...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    auc_score = roc_auc_score(y_test, y_pred_proba[:,1])
    ap = average_precision_score(y_test, y_pred_proba[:,1])

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC-ROC:", auc_score)
    print("PR-ROC:", ap)

    return y_pred_proba[:,1]

def get_predict_col(data, model, features):
    print("Getting predictions...")
    feat = data[features]
    y_hat = model.predict_proba(feat)
    print("Finished prediction")
    return pd.DataFrame({"prediction": y_hat[:,1]})



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

    with open("./weights/rfmodelgs.pkl","rb") as f:
        clf_xgb = pickle.load(f)
    scaled_data = scale_data(data, rfecv_features, identifier)

    y_hat = get_predict_col(scaled_data, clf_xgb, rfecv_features)
    # del scaled_data
    print("Concatenating prediction column...")
    df = pd.concat([pd.DataFrame({"cell_line": [cancer] * len(data) }), data, y_hat], axis = 1)

    print("Completed job" + i)
    return df