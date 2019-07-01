import re
import math
import pandas as pd
import numpy as np
from glob import glob


class DatasetLoader:

    train_columns = ['session_id', 'session_position', 'session_length', 'track_id_clean', 'skip_1', 'skip_2', 'skip_3',
                     'not_skipped', 'context_switch', 'no_pause_before_play', 'short_pause_before_play',
                     'long_pause_before_play', 'hist_user_behavior_n_seekfwd', 'hist_user_behavior_n_seekback',
                     'hist_user_behavior_is_shuffle', 'hour_of_day', 'date', 'premium', 'context_type',
                     'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end']

    def __init__(self, train_path, test_path, track_features_path):
        self.train_path = train_path
        self.test_path = test_path
        self.track_features_path = track_features_path

    def read_from_csv(self, filename, format="dataframe"):
        if format == "numpy":
            return pd.read_csv(filename, names=self.train_columns).to_numpy()
        elif format == "dict":
            return pd.read_csv(filename, names=self.train_columns).to_dict()
        else:
            return pd.read_csv(filename, names=self.train_columns)

    def generate_test_split(self, test_set_size=0.1):
        filenames = sorted(glob(self.train_path + "log_*.csv"))
        for filename in filenames:
            # Opens train_set
            file_id = re.search("log_(.*).csv", filename).group(1)
            train_set = self.read_from_csv(filename)

            # Retrieves all distinct session_id's
            session_ids = np.unique(train_set['session_id'].to_numpy())

            # Count the sessions
            no_sessions = len(session_ids) - 2
            test_size = math.ceil(no_sessions * test_set_size)

            # Generate the split
            test_sessions = session_ids[0:math.ceil((test_size/2))]
            prehist_sessions = session_ids[math.ceil((test_size/2)):test_size]
            train_sessions = session_ids[test_size:no_sessions+1]

            # Build new DataFrames by filtering the original
            test_df = train_set['session_id'].isin(test_sessions)
            prehist_df = train_set['session_id'].isin(prehist_sessions)
            train_df = train_set['session_id'].isin(train_sessions)

            # Writing DataFrames to files
            train_set[test_df].to_csv(path_or_buf=self.test_path + "log_input_{id}.csv".format(id=file_id), index=False)
            train_set[prehist_df].to_csv(path_or_buf=self.test_path + "log_prehist_{id}.csv".format(id=file_id), index=False)
            train_set[train_df].to_csv(path_or_buf=filename, index=False)
