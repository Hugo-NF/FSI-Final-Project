import pandas as pd


class DatasetLoader:
    columns = ['session_id', 'session_position', 'session_length', 'track_id_clean', 'skip_1', 'skip_2', 'skip_3',
               'not_skipped', 'context_switch', 'no_pause_before_play', 'short_pause_before_play',
               'long_pause_before_play', 'hist_user_behavior_n_seekfwd', 'hist_user_behavior_n_seekback',
               'hist_user_behavior_is_shuffle', 'hour_of_day', 'date', 'premium', 'context_type',
               'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end']

    def read(self, filename, format="numpy"):
        if format == "dataframe":
            return pd.read_csv(filename, names=self.columns)
        elif format == "dict":
            return pd.read_csv(filename, names=self.columns).to_dict()
        else:
            return pd.read_csv(filename, names=self.columns).to_numpy()

