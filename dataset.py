import os
import shutil
import kagglehub
import pandas as pd
import glob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
# Download latest version
path = kagglehub.dataset_download("dhoogla/cicddos2019")
destination_dir = 'PICSC/cicddos2019/'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for file_name in os.listdir(path):
    full_file_name = os.path.join(path, file_name)
    if os.path.isfile(full_file_name):
        shutil.move(full_file_name, os.path.join(destination_dir, file_name))


# Collecting Training and Testing Dataset File Paths
dfps_train = glob.glob('PICSC/cicddos2019/*-training.parquet')
dfps_test = glob.glob('PICSC/cicddos2019/*-testing.parquet')
    
# Common Prefixes in both lists

train_prefixes = [dfp.split('/')[-1].split('-')[0] for dfp in dfps_train]
test_prefixes = [dfp.split('/')[-1].split('-')[0] for dfp in dfps_test]

common_prefixes = list(set(train_prefixes).intersection(test_prefixes))

# Filter the dataframes to only include the common prefixes
dfps_train = [dfp for dfp in dfps_train if dfp.split('/')[-1].split('-')[0] in common_prefixes]
dfps_test = [dfp for dfp in dfps_test if dfp.split('/')[-1].split('-')[0] in common_prefixes]

train_df = pd.concat([pd.read_parquet(dfp) for dfp in dfps_train], ignore_index=True)
test_df = pd.concat([pd.read_parquet(dfp) for dfp in dfps_test], ignore_index=True)

def dataset_print():
    print("Train data size:", train_df.shape)
    print("Test data size:", test_df.shape)

def train_data_print():
    # print("Train data info:", train_df.info())
    # print("Train data description:", train_df.describe())
    print("Train labels distribution:", train_df["Label"].value_counts())
    
def test_data_print():
    # print("Test data info:", test_df.info())
    # print("Test data description:", test_df.describe())
    print("Test labels distribution:", test_df["Label"].value_counts())
