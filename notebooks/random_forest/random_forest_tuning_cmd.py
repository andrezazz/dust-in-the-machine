import pandas as pd
from sklearn.model_selection import train_test_split
import argparse 
bin_count = 171

# Create cmd arguements for parameters

parser = argparse.ArgumentParser()

parser.add_argument("--n_estimators", dest="n_estimators", default=1000) 
parser.add_argument("--n_jobs", dest="n_jobs", default=16) 
parser.add_argument("--max_depth", dest="max_depth", default=12) 
parser.add_argument("--min_samples_leaf", dest="min_samples_leaf", default = 1)
parser.add_argument("--min_samples_split", dest="min_samples_split", default = 2)
parser.add_argument("--max_features", dest="max_features", default = 'auto')

# Read arguments from command line 
args = parser.parse_args()   

def create_test_train(data_set_path, test_size=0.10):
    """ Splits a given csv file into testing and training. Target column is all the bins."""
    # Make sure the columns are set
    data_set = pd.read_csv(data_set_path)

    # Add column for classifying whether the output has most of the data in the last 10 bins.
    end_average = data_set[[f'Output_Bin_{i}' for i in range(bin_count-10, bin_count)]].sum(axis=1) > 0.9

    data_set['Output_Is_End'] = end_average
    data_set['Output_Is_End'] = data_set['Output_Is_End'].astype(int)

    # Shuffle the data
    data_set = data_set.sample(frac=1, random_state=0)
 
    # Select all except output bins
    data_set_X = data_set.drop([f'Output_Bin_{i}' for i in range(bin_count)] + ['Output_Is_End'], axis=1)
    # Select only the output bins
    data_set_Y = data_set[[f'Output_Bin_{i}' for i in range(bin_count)]+ ['Output_Is_End']]

    #Split into training and test data
    return train_test_split(data_set_X,
                            data_set_Y,
                            test_size=test_size, 
                            random_state=300)

filename= "/project/SDS-capstones-kropko21/uva-astronomy/dust_training_data_all_bins_v2.csv"
X_train, X_test, y_train, y_test = create_test_train(filename, test_size=0.10)

output_is_not_end_train_idx = y_train.Output_Is_End.index[y_train['Output_Is_End'] == 0]

X = X_train.loc[output_is_not_end_train_idx]
y = y_train.loc[output_is_not_end_train_idx].drop(['Output_Is_End'], axis=1)

from sklearn.ensemble import RandomForestRegressor

max_depth = int(args.max_depth)
if max_depth == 0:
    max_depth = None

rf = RandomForestRegressor(max_depth=max_depth, 
                            n_estimators=int(args.n_estimators),
                            n_jobs=int(args.n_jobs),
                            min_samples_leaf=int(args.min_samples_leaf),
                            min_samples_split=int(args.min_samples_split),
                            max_features=str(args.max_features),
                            random_state=0)


import time # Just to compare fit times
start = time.time()
rf.fit(X, y)
end = time.time()
print("Tune Fit Time:", end - start)

output_is_not_end_test_idx = y_test.Output_Is_End.index[y_test['Output_Is_End'] == 0]

X_test_non = X_test.loc[output_is_not_end_test_idx]
y_test_non = y_test.loc[output_is_not_end_test_idx].drop(['Output_Is_End'], axis=1)

score = rf.score(X_test_non, y_test_non)
print("Score: ", score)

from joblib import dump, load
filename = f'rf-model-depth-{args.max_depth}-trees-{args.n_estimators}-min_samps-{args.min_samples_leaf}'
dump(rf, f'/project/SDS-capstones-kropko21/uva-astronomy/{filename}.joblib')


# RandomForestRegressor(max_depth=10, n_estimators=1000, n_jobs=16, random_state=0) Time
# Tune Fit Time: 22337.464445590973
# Score:  0.8217937537582921