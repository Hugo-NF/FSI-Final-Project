import sys
import csv
import pandas as pd
import glob
import numpy as np
from math import exp, ceil

import src.plot_tools as plt_tools

test_path = '../data/test_set/'
test_input = sorted(glob.glob(test_path + "log_input*.csv"))

pred_path = '../data/output/'
pred_input = sorted(glob.glob(pred_path + "*.txt"))

output_fname = '../data/submission_output/'
missing_output_file = '../data/missing_files/missing.txt'

def get_accuracy(predictions):
    # Indicator for if the skip prediction was correct. 1 -> correct predicted
    L = lambda x: 1 if x[0] == x[1] else 0

    # Convert predictions to Dataframe and matriz
    pred_list = predictions.values()
    df_pred = pd.DataFrame(pred_list, index=predictions.keys(), dtype=float)

    # Gets true values from test files
    true_values = pd.DataFrame()
    for file in test_input:
        df_file = pd.read_csv(file)
        df_file.set_index('session_id', inplace=True)

        for pred_row in df_pred.iterrows():
            new_true_row = pd.DataFrame([np.array(df_file.loc[pred_row[0]]['skip_2'], dtype=float)], index=[pred_row[0]])
            true_values = pd.concat([true_values, new_true_row])

    # Get a Dataframe with inidicator (that indicates which predictions is correct and which is not)
    zipped_pred_true = np.dstack((df_pred.as_matrix(), true_values.as_matrix()))
    indicator_list = list(map(lambda x: list(map(L, x)), zipped_pred_true))
    df_indicator = pd.DataFrame(indicator_list, index=predictions.keys(), dtype=float)

    # Gets A(i) = Accuracy at position i of the track (mean of each session correct predictions of track i)
    A = np.zeros(10)
    for i in range(len(A)):
        A[i] = sum(df_indicator[i]) / len(df_pred[i].dropna())

    # Gets Average Acurracy of each session
    AA = np.zeros(len(predictions))
    for i in range(len(df_indicator)):
        indicator_row_accuracy = A * df_indicator.iloc[i]
        AA[i] = sum(indicator_row_accuracy) / len(df_pred.iloc[i].dropna())

    # Gets Mean Average Acurracy
    MAA = sum(AA)/len(AA)

    #------------ Show metrics----------
    method_name = 'Boosting'
    saves_path = '../plots_and_tables'

    # Convert metrics to percent mode and rounded to 2
    A = list(map(lambda a: str(round(a*100,2)), A))
    MAA = round(MAA*100, 2)
    
    with open(saves_path+'/Challenge_metrics_'+method_name+'.txt', 'w') as file:
        file.write('Firt Predictions Accuracy: {A}'.format(A='%, '.join(A)+'%'))
        print('Firt Predictions Accuracy: {A}'.format(A='%, '.join(A)+'%'))
        file.write('MAA: {MAA}'.format(MAA=MAA))
        print('MAA: {MAA}'.format(MAA=MAA))



    # Plots for first session; First music prediction and all predictions
    first_session_true = list(true_values.iloc[0].dropna())
    first_session_pred = list(df_pred.iloc[0].dropna())
    first_track_true = true_values[0].dropna().as_matrix().ravel()
    first_track_pred = df_pred[0].dropna().as_matrix().ravel()
    all_true = true_values.dropna().as_matrix().ravel()
    all_pred = df_pred.dropna().as_matrix().ravel()
    

    # Confusion matrix plots
    plt_tools.plot_confusion_matrix(y_pred=first_session_pred,
                                    y_true=first_session_true,
                                    normalize=True,
                                    save_image=True,
                                    image_path=saves_path,
                                    image_name='Confusion_Matrix_'+method_name+'_first_session'
                                    )

    plt_tools.plot_confusion_matrix(y_pred=first_track_pred,
                                    y_true=first_track_true,
                                    normalize=True,
                                    save_image=True,
                                    image_path=saves_path,
                                    image_name='Confusion_Matrix_'+method_name+'_first_music'
                                    )

    plt_tools.plot_confusion_matrix(y_pred=all_pred,
                                    y_true=all_true,
                                    normalize=True,
                                    save_image=True,
                                    image_path=saves_path,
                                    image_name='Confusion_Matrix_'+method_name
                                    )

    # Normal Metrics plots
    plt_tools.get_and_plot_metrics(y_pred=first_session_pred,
                                   y_true=first_session_true,
                                   plot_table=True,
                                   save_table=True,
                                   file_path=saves_path,
                                   file_name='Normal_Metrics_'+method_name+'_first_session')

    plt_tools.get_and_plot_metrics(y_pred=first_track_pred,
                                   y_true=first_track_true,
                                   plot_table=True,
                                   save_table=True,
                                   file_path=saves_path,
                                   file_name='Normal_Metrics_'+method_name+'_first_music'
                                   )

    plt_tools.get_and_plot_metrics(y_pred=all_pred,
                                   y_true=all_true,
                                   plot_table=True,
                                   save_table=True,
                                   file_path=saves_path,
                                   file_name='Normal_Metrics_'+method_name
                                   )

    return


if __name__ == "__main__":
    predictions = {}
    for file_ in pred_input:
        csvreader = csv.reader(open(file_))
        for row in csvreader:
            if row[1] != "":
                if row[0] not in predictions:
                    last_act = 1.0
                    if row[-1] == 'False':
                        last_act = 0.0
                    pred = [[]]
                    counter = 0
                    for p in row[1:-1]:
                        if counter == 10:
                            counter = 0
                            pred.append([])
                        pred[-1].append(p)
                        counter += 1
                    if '_2.txt' in file_:
                        pred = pred[::-1]
                    final_pred = []
                    len_pred = len(pred)
                    for i, p in enumerate(pred):
                        curr_div = 0.0
                        curr_pred = 0.0
                        first_pred = 0
                        last_pred = 0
                        for j, p2 in enumerate(p):
                            if j < len_pred:
                                curr_div += (len_pred * 2 - 1) - abs(i * 2 - j * 2)
                                curr_pred += float(p2) * ((len_pred * 2 - 1) - abs(i * 2 - j * 2))
                        curr_pred += last_act * len_pred * 2
                        curr_div += len_pred * 2
                        final_pred.append(int(round(curr_pred / curr_div)))
                    predictions[row[0]] = final_pred
                elif (len(predictions[row[0]]) * 10) != len(row[1:-1]):
                    print(row[0], ",", file_)
    get_accuracy(predictions)
