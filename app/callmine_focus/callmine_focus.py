import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as patches
import plotly.graph_objs as go
import plotly.express as px
from sklearn import preprocessing
import itertools
import copy
import time

import config

from callmine_focus.gen2Out.gen2out import gen2Out
from callmine_focus.gen2Out.utils import sythetic_group_anomaly, plot_results

from callmine_focus.LookOut.iForest import iForest, forest_outliers
from callmine_focus.LookOut.helper import get_coverage, generate_frequency_list
from callmine_focus.LookOut.LookOut import LookOut
from callmine_focus.LookOut.ranklist import generate_graph
from callmine_focus.LookOut.structures import Graph


# Global variablesgraph
fs = [6, 5] # Figure size
# keys = ['out_degree', 'in_degree',
#         'core', 'weighted_out_degree',
#         'weighted_in_degree', 'in_median_iat',
#         'in_call_count', 'in_median_measure',
#         'out_median_iat', 'out_call_count',
#         'out_median_measure']
min_max_scaler = preprocessing.MinMaxScaler()


def generate_feature_combinations(key_features, d=1):
    """
    Generate feature combinations with the
    set of informed features
    
    Parameters
    ----------
    key_features: array of str
        features to be combined
    d: int
        dimensionality, i.e., the number of
        features per combination
    """
    combinations = []
    plot_dict = {}
    
    for subset in itertools.combinations(key_features, r=d):
        combinations.append(list(subset)[:])

    for i, c in enumerate(combinations):
        plot_dict[i] = c
    
    return combinations, plot_dict


def runGen2Out(ids, features, option=1):
    """
    Run gen2Out for anomaly detection
    
    Parameters
    ----------
    ids: array of str
        row ids
    features: array of str
        set of features to consider
        while running gen2Out
    option: int
        1 for point anomaly, 2 for group anomaly
    """
    model = gen2Out(lower_bound=9,
                upper_bound=12,
                max_depth=7,
                rotate=True,
                contamination='auto',
                random_state=0)
    
    if option==1: # point anomaly
        scores = model.point_anomaly_scores(X=features)
        tuples = [(ids[i], float(scores[i])) for i in range(0, len(ids))]
        scores = sorted(tuples, key = lambda x: x[1], reverse = True)
    else: # group anomaly
        scores, gindices = model.group_anomaly_scores(X=features,
                                                      x_ideal=0.1, y_ideal=1)
        
        tuples = [(ids[i], float(scores[0])) for i in gindices[0]]
        scores = sorted(tuples, key = lambda x: x[1], reverse = True)
        
    return scores
    


def run(N_val, B_val, P_val, features, dimensionality, detection_option=1):
    """
    Run CallMine-Focus
    
    Parameters
    ----------
    N_val: int
        number of outliers
    B_val: int
        budget, i.e., the number of plots to show
    P_val: int
    features: array of str
        feature keys to consider
    dimensionality: int
        number of dimensions to recommend
    detection_option: int
        1 for point anomaly, 2 for group anomaly
    """

#     rank_matrix: nd-array of float
#         matrix with rows ranked by features
#     plot_combinations: nd-array of str
#         combinations of features
#     outlier_ids: array
#         ids of detected outliers

    all_features = config.df_features.columns[1:]

    print("CallMine:\n>>>>>> Generate feature combinations")
    plot_combinations, plot_dict = generate_feature_combinations(key_features=features,
                                                                 d=int(dimensionality))

    print(">>>>>> Normalize features")

    config.df_features = min_max_scaler.fit_transform(config.df_features[all_features].values)
    config.df_features = pd.DataFrame(data=config.df_features, columns=all_features)
    config.df_features = config.df_features.rename_axis('node_ID').reset_index()

    print(">>>>>>> Rank outliers with feature combinations")
    rank_matrix =[]

    for i, combination in enumerate(plot_combinations):
        print(">>>>>> --- Run Isolation Forest / gen2Out and get Scores (every combination of features) " + str(i+1) + " / " + str(len(plot_combinations)),
                end="\r")
        
        if dimensionality == 1:
            scores = iForest(ids = config.df_features['node_ID'],
                             features = np.asarray(config.df_features[combination],
                             dtype = float))
        else:
            scores = runGen2Out(ids = config.df_features['node_ID'],
                                features = np.asarray(config.df_features[combination],
                                dtype = float),
                                option=detection_option)
        
        rank_matrix.append(scores)

    print(">>>>>>> Run Isolation Forest / gen2Out and get Scores (all features)")
    if dimensionality == 1:
        scores = iForest(ids = config.df_features['node_ID'],
                            features = np.asarray(config.df_features[features], dtype = float))
    else:
        scores = runGen2Out(ids = config.df_features['node_ID'],
                            features = np.asarray(config.df_features[features], dtype = float),
                            option=detection_option)

    print(">>>>>> Get top outliers")
    outlier_ids = forest_outliers(N=N_val, scores=scores)

    print('\t#scores:', len(scores))
    print('\t#df_features:', len(config.df_features))
    print('\toutliers:', outlier_ids)
    outlier_circle_size = 40
    
    # Create graph between outliers and plots
    print("Generating Bipartite Graph")
    scaled_matrix, normal_matrix = generate_graph(P_val, rank_matrix, outlier_ids)
    saved_graph = Graph(scaled_matrix)
    print("Graph Generated Successfully")

    # Run appropriate algorithm to get list of selected graphs
    scatter_plots = len(plot_combinations)
    file = open("" + "log_CallMine-Focus.txt", 'w')

    algos = ["LookOut"]
    plot_objects = []

    for algo in algos:
        print("\nIteration " + algo)
        graph = copy.deepcopy(saved_graph)
        print( "N_val = ", N_val, " Budget = ", B_val )

        start_time = time.time()
        print( "Running " + algo + " Algorithm" )
        plots = LookOut(graph, B_val, algo)
        frequencies = generate_frequency_list(plots, scaled_matrix)
        print('frequencies:', frequencies)
        
        print(algo + " Complete")
        elapsed_time = time.time() - start_time

        print("Saving Plots")
        coverage, max_coverage = get_coverage(plots, N_val, normal_matrix)
        
        print( "\t-> Total Plots Generated = ", end='' ); print(scatter_plots)
        print( "\t-> Total Plots Chosen = ", end='' ); print(len(plots))
        print( "\t-> Coverage = ", end='' ); print("{0:.3f} / {1:.3f}".format(coverage, max_coverage))

        # Inverse norm (for plotting)
        config.df_features = min_max_scaler.inverse_transform(config.df_features[all_features].values)
        config.df_features = pd.DataFrame(data=config.df_features, columns=all_features)
        config.df_features = config.df_features.rename_axis('node_ID').reset_index()
        
        # Save selected plots as png images
        for i, plot in enumerate(plots):
            pair = plot_combinations[plot]
              
            if dimensionality == 1:
                imgname = str(i) + '_' + str(plot) + '_histogram_plot'
                print(str(plot), end=' ')
                for p in pair:
                    imgname += '_' + p
                    print(str(p), end=' ')
                print()
                
                fig_cm_r = plotHistogram(df=config.df_features,
                            c1=pair[0],
                            frequencies=frequencies,
                            plot_number=plot,
                            # figname=imgname+'.png',
                            # show_plots=True
                            )
                plot_objects.append(fig_cm_r)

            elif dimensionality == 2:
                imgname = str(i) + '_' + str(plot) + '_pair_plot'
                print(str(plot), end=' ')
                for p in pair:
                    imgname += '_' + p
                    print(str(p), end=' ')
                print()
                
                fig_cm_r = plotScatterPlot(df=config.df_features,
                            c1=pair[0],
                            c2=pair[1],
                            frequencies=frequencies,
                            plot_number=plot,
                            # figname=imgname+'.png',
                            # show_plots=True
                            )
                plot_objects.append(fig_cm_r)

            elif dimensionality > 2:
                imgname = str(i) + '_' + str(plot) + '_parallel_coordinates'
                print(str(plot), end=' ')
                for p in pair:
                    imgname += '_' + p
                    print(str(p), end=' ')
                print()
                
                fig_cm_r = plot_parallel_coordinates(df_features=config.df_features,
                                          columns=pair,
                                          frequencies=frequencies,
                                          plot_number=plot,
                                        #   figname=imgname+'.html',
                                        #   show_plots=True
                                          )
                plot_objects.append(fig_cm_r)
            
    file.close()
    print( "Finished CallMine" )

    return plot_combinations, plots, plot_objects, outlier_ids


def plotHistogram(df, c1, frequencies, plot_number):
    """
    Plot 1-d histogram
    
    Parameters
    ----------
    df: pandas DataFrame
        features
    c1: string
        feature to plot
    frequencies: nd-array
        frequency of outliers
    plot_number: int
        number of plot to show
    """
    fig = plt.figure(figsize=fs)
    colors = len(df) * ['#808080']
    
    plt.hist(x=np.log10(df[c1]+1), bins=100)
    
    for outlier in frequencies.keys():
        if plot_number == frequencies[outlier][1]:
            plt.vlines(x = np.log10(df[c1].iloc[outlier]+1), ymin=0, ymax=plt.ylim()[1], linestyles='dashed', colors='red')##5C1A1B')
        else:
            plt.vlines(x = np.log10(df[c1].iloc[outlier]+1), ymin=0, ymax=plt.ylim()[1], linestyles='dashed', colors='green')#556270')
    
        plt.xlabel(c1.replace('_', ' ') + ' - log10(x+1)')
        plt.ylabel('count')
    
    plt.yscale('log')
    plt.grid()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    return fig


def plotScatterPlot(df, c1, c2, frequencies, plot_number):
    """
    Plot 2-d histogram
    
    Parameters
    ----------
    df: pandas DataFrame
        features
    c1: string
        first feature to plot
    c2: string
        second feature to plot
    frequencies: nd-array
        frequency of outliers
    plot_number: int
        number of plot to show
    """
    fig = plt.figure(figsize=fs)
    plt.hexbin(np.log10(df[c1]+1), np.log10(df[c2]+1), mincnt=1, cmap='jet', bins='log')
    plt.xlabel(c1.replace('_', ' ') + ' - log10(x+1)')
    plt.ylabel(c2.replace('_', ' ') + ' - log10(x+1)')
    plt.colorbar()
    
    for outlier in frequencies.keys():
        size = frequencies[outlier][0]*10
        
        if plot_number == frequencies[outlier][1]:
            plt.scatter(np.log10(df.iloc[outlier][c1]+1), np.log10(df.iloc[outlier][c2]+1),
                    # c='k',
                    facecolors='none',
                    edgecolor='red', linewidth=1, s = int(size))
        else:
            plt.scatter(np.log10(df.iloc[outlier][c1]+1), np.log10(df.iloc[outlier][c2]+1),
                    facecolors='none',
                    edgecolor='blue', linewidth=1, s = int(size))
        
        plt.xlabel(c1.replace('_', ' ') + ' - log10(x+1)')
        plt.ylabel(c2.replace('_', ' ') + ' - log10(x+1)')
    
    plt.grid()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    return fig


def plot_parallel_coordinates(df_features,
                              columns, 
                              frequencies,
                              plot_number):
    """
    Plot n-d parallel coordinates
    
    Parameters
    ----------
    df: pandas DataFrame
        features
    columns: array of str
        features to plot
    frequencies: nd-array
        frequency of outliers
    plot_number: int
        number of plot to show
    """
    
    colors = len(df_features) * [0]
    
    for outlier in frequencies.keys():
        if plot_number == frequencies[outlier][1]:
            colors[outlier] = 2
        else:
            colors[outlier] = 1
    
    df_features['colors'] = colors
    
    fig_parallel_coordinates = px.parallel_coordinates(
                               df_features[columns],
                               columns,
                               color_continuous_scale=['grey', 'blue', 'red'],
                               color=df_features['colors'], color_continuous_midpoint=1
                            )
    
    fig_parallel_coordinates.update_layout(#width=900,
                        height=550)
    fig_parallel_coordinates.update_layout(
                        font_size=22)
    fig_parallel_coordinates.update_layout(coloraxis_showscale=False)

    
    return fig_parallel_coordinates


