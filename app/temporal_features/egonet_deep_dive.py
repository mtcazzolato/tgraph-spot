import os
import pandas as pd
import numpy as np
import networkx as nx                                                                                                                                                                                                                                                                                                                                     

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# %matplotlib inline

import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt
import matplotlib.patches as patches

NODE_ID = "node_ID"
CALL_COUNT = "call_count"
SUM_MEASURE = "sum_measure"
G = None
all_prefix = []

def get_out_degree(df_slice, df_nodes, prefix=''):
    global G
    
    new_G = nx.from_pandas_edgelist(df_slice,
                                    source=SOURCE,
                                    target=DESTINATION,
                                    create_using=nx.DiGraph())
    
    prefix_ = 'out_degree__' + str(prefix)
    out_degs = [(node, val) for (node, val) in new_G.out_degree()] #type is list 
    out_degs_df = pd.DataFrame(out_degs, columns=[SOURCE, (prefix_)])
    df_nodes = df_nodes.merge(out_degs_df, left_on=NODE_ID, right_on=SOURCE, how='left')
    df_nodes = df_nodes.drop(columns=[SOURCE])
    
    G = nx.compose(G, new_G)
    
    prefix_ = 'csum_out_degree__' + str(prefix)
    out_degs = [(node, val) for (node, val) in G.out_degree()] #type is list 
    out_degs_df = pd.DataFrame(out_degs, columns=[SOURCE, (prefix_)])
    df_nodes = df_nodes.merge(out_degs_df, left_on=NODE_ID, right_on=SOURCE, how='left')
    df_nodes = df_nodes.drop(columns=[SOURCE])

    return df_nodes

def get_in_degree(df_slice, df_nodes, prefix=''):
    global G
    
    new_G = nx.from_pandas_edgelist(df_slice,
                                    source=SOURCE,
                                    target=DESTINATION,
                                    create_using=nx.DiGraph())
    
    prefix_ = 'in_degree__' + str(prefix)
    in_degs = [(node, val) for (node, val) in new_G.in_degree()] #type is list 
    in_degs_df = pd.DataFrame(in_degs, columns=[SOURCE, (prefix_)])
    df_nodes = df_nodes.merge(in_degs_df, left_on=NODE_ID, right_on=SOURCE, how='left')
    df_nodes = df_nodes.drop(columns=[SOURCE])
    
    G = nx.compose(G, new_G)
    
    prefix_ = 'csum_in_degree__' + str(prefix)
    in_degs = [(node, val) for (node, val) in G.in_degree()] #type is list 
    in_degs_df = pd.DataFrame(in_degs, columns=[SOURCE, (prefix_)])
    df_nodes = df_nodes.merge(in_degs_df, left_on=NODE_ID, right_on=SOURCE, how='left')
    df_nodes = df_nodes.drop(columns=[SOURCE])

    return df_nodes

def fill_temporal_features(df_slice, df_nodes, node_direction="source", prefix='out_', prefix_direction='out_'):
    global NODE_ID
    
    # First: sort elements by source and timestamp
    df_aux = df_slice.sort_values(by=[node_direction, TIMESTAMP]).reset_index(drop=True)
    
    if len(df_aux) > 0:
        # Group by source, get the difference (iat) between ts of every hash, convert it to timedelta
        df_aux['diff_ts'] = df_aux.groupby(by=[node_direction])[TIMESTAMP].diff().astype("timedelta64[s]")

        # Creates groups by ahash values, get the iat for every row
        group = df_aux.groupby(by=[node_direction], axis=0)['diff_ts']

        # Aggregate iat per group/hash
        df_iat_measure = pd.DataFrame()
        df_iat_measure[prefix_direction+CALL_COUNT+'__'+prefix]  = group.size()

        # Creates groups by ahash values, get the duration of every row
        group = df_aux.groupby(by=[node_direction], axis=0)[MEASURE]

        df_iat_measure[prefix_direction+SUM_MEASURE+'__'+prefix]     = group.sum()

        df_iat_measure.reset_index(inplace=True)
        df_nodes = df_nodes.merge(df_iat_measure, left_on=NODE_ID, right_on=node_direction, how='left')
        df_nodes = df_nodes.drop(columns=[node_direction])
        df_nodes = df_nodes.fillna(0)
    else:
        df_nodes[prefix_direction+SUM_MEASURE+'__'+prefix] = 0
        df_nodes[prefix_direction+CALL_COUNT+'__'+prefix] = 0

    return df_nodes

def get_degrees(df_data, df_nodes):
    global G, all_prefix

    G = nx.empty_graph(create_using=nx.DiGraph())

    # hours = df_data[TIMESTAMP].dt.hour.unique()
    hours = np.arange(0, 24)
    days = df_data[TIMESTAMP].dt.day.unique()

    n_features_degree = 0
    all_prefix = []

    for d in days:
        df = df_data[df_data[TIMESTAMP].dt.day == d]
        for h in range(min(hours), max(hours)+1):#, h_interval):
            prefix = str(d) + '_' + str(h)
            all_prefix.append(prefix)

            idx = np.where(df[TIMESTAMP].dt.hour == h)[0]

            df_nodes = get_in_degree(df_slice=df.iloc[idx], df_nodes=df_nodes, prefix=prefix)
            
            n_features_degree+=1

    G = nx.empty_graph(create_using=nx.DiGraph())
            
    for d in days:
        df = df_data[df_data[TIMESTAMP].dt.day == d]
        for h in range(min(hours), max(hours)+1):#, h_interval):
            prefix = str(d) + '_' + str(h)

            idx = np.where(df[TIMESTAMP].dt.hour == h)[0]

            df_nodes = get_out_degree(df_slice=df.iloc[idx], df_nodes=df_nodes, prefix=prefix)
            
    df_nodes= df_nodes.fillna(0)

    return df_nodes

def get_temporal_features(df_data, df_nodes):
    hours = np.arange(0, 24)
    days = df_data[TIMESTAMP].dt.day.unique()

    idx_previous=[]
    for d in days:
        df_slice = df_data[df_data[TIMESTAMP].dt.day == d]
        for h in range(min(hours), max(hours)+1):#, h_interval):
            prefix = str(d) + '_' + str(h)
            
            idx = list(np.where(df_slice[TIMESTAMP].dt.hour == h)[0])
            idx.extend(idx_previous)
            
            df_nodes = fill_temporal_features(df_slice=df_slice.iloc[idx],
                                df_nodes=df_nodes,
                                node_direction=SOURCE,
                                prefix=prefix,
                                prefix_direction='out_')
            
    idx_previous=[]
    for d in days:
        df_slice = df_data[df_data[TIMESTAMP].dt.day == d]
        for h in range(min(hours), max(hours)+1):#, h_interval):
            prefix = str(d) + '_' + str(h)

            idx = list(np.where(df_slice[TIMESTAMP].dt.hour == h)[0])
            idx.extend(idx_previous)
            
            df_nodes = fill_temporal_features(df_slice=df_slice.iloc[idx],
                                    df_nodes=df_nodes,
                                    node_direction=DESTINATION,
                                    prefix=prefix,
                                    prefix_direction='in_')
            
    df_nodes= df_nodes.fillna(0)
    return df_nodes

def cum_sum_in_degree(df_nodes):
    global all_prefix

    columns = []
    for p in all_prefix: columns.append('csum_in_degree__'+p)

    idx = df_nodes.sort_values(by=columns[-1], ascending=False).index
    fig = plt.figure(figsize=[8,4])

    for row in idx:
        plt.plot(df_nodes[columns].iloc[row].values+1)

    plt.grid()
    plt.title('Cumulated in degree in the period')
    plt.ylabel('cummulated  in degree')
    plt.xlabel('timestamp (hours)')

    # Add shades in the sleeping hours
    # left, bottom, width, height = (0, 0, 8, plt.gca().get_ylim()[1])
    # rect=patches.Rectangle((left,bottom),width,height, alpha=0.3, facecolor="grey")
    # plt.gca().add_patch(rect)
    
    # plt.xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])

    return fig

def cum_sum_out_degree(df_nodes):
    global all_prefix

    columns = []
    for p in all_prefix: columns.append('csum_out_degree__'+p)

    idx = df_nodes.sort_values(by=columns[-1], ascending=False).index
    fig = plt.figure(figsize=[8,4])

    for row in idx:
        plt.plot(df_nodes[columns].iloc[row].values+1)

    plt.grid()
    plt.title('Cumulated out degree in the period')
    plt.ylabel('cummulated  out degree')
    plt.xlabel('timestamp (hours)')

    # Add shades in the sleeping hours
    # left, bottom, width, height = (0, 0, 8, plt.gca().get_ylim()[1])
    # rect=patches.Rectangle((left,bottom),width,height, alpha=0.3, facecolor="grey")
    # plt.gca().add_patch(rect)
    
    # plt.xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])

    return fig


def cum_sum_in_call_count(df_nodes):
    global all_prefix

    columns = []
    for p in all_prefix: columns.append('in_call_count__' + p)

    idx = df_nodes.sort_values(by=columns[-1], ascending=False).index
    fig = plt.figure(figsize=[8,4])

    for row in idx:
        plt.plot(df_nodes[columns].iloc[row].values+1)

    plt.grid()
    plt.title('Cumulated incoming calls in the period:')
    plt.ylabel('cummulated  in call count')
    plt.xlabel('timestamp (hours)')

    # # Add shades in the sleeping hours
    # left, bottom, width, height = (0, 0, 8, plt.gca().get_ylim()[1])
    # rect=patches.Rectangle((left,bottom),width,height, alpha=0.3, facecolor="grey")
    # plt.gca().add_patch(rect)
    # rect2=patches.Rectangle((left+24,bottom),width,height, alpha=0.3, facecolor="grey")
    # plt.gca().add_patch(rect2)
    # plt.xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])

    return fig


def cum_sum_out_call_count(df_nodes):
    global all_prefix

    columns = []
    for p in all_prefix: columns.append('out_call_count__' + p)

    idx = df_nodes.sort_values(by=columns[-1], ascending=False).index
    fig = plt.figure(figsize=[8,4])

    for row in idx:
        plt.plot(df_nodes[columns].iloc[row].values+1)

    plt.grid()
    plt.title('Cumulated outgoing calls in the period:')
    plt.ylabel('cummulated out call count')
    plt.xlabel('timestamp (hours)')

    # # Add shades in the sleeping hours
    # left, bottom, width, height = (0, 0, 8, plt.gca().get_ylim()[1])
    # rect=patches.Rectangle((left,bottom),width,height, alpha=0.3, facecolor="grey")
    # plt.gca().add_patch(rect)
    # rect2=patches.Rectangle((left+24,bottom),width,height, alpha=0.3, facecolor="grey")
    # plt.gca().add_patch(rect2)
    # plt.xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])

    return fig


def get_cum_call_counts(df_nodes):
    columns = []
    for p in all_prefix: columns.append('in_call_count__' + p)

    for i in range(len(columns)-1):
        df_nodes[columns[i+1]] = df_nodes[columns[i]] + df_nodes[columns[i+1]]
        
    columns = []
    for p in all_prefix: columns.append('out_call_count__' + p)

    for i in range(len(columns)-1):
        df_nodes[columns[i+1]] = df_nodes[columns[i]] + df_nodes[columns[i+1]]

    return cum_sum_in_call_count(df_nodes), cum_sum_out_call_count(df_nodes)


def get_curves(df_nodes, df_raw_data, source, destination, measure, timestamp):
    global MEASURE, TIMESTAMP, SOURCE, DESTINATION

    SOURCE = source
    DESTINATION = destination
    MEASURE = measure
    TIMESTAMP = timestamp

    df_temporal_features = df_nodes[[NODE_ID]]
    df_source = pd.DataFrame(data=df_temporal_features[NODE_ID],
                                     columns=[NODE_ID]).set_index(NODE_ID).join(df_raw_data.set_index(SOURCE))[[DESTINATION, MEASURE, TIMESTAMP]].reset_index()
    
    df_destination = pd.DataFrame(data=df_temporal_features[NODE_ID],
                                     columns=[NODE_ID]).set_index(NODE_ID).join(df_raw_data.set_index(DESTINATION))[[SOURCE, MEASURE, TIMESTAMP]].reset_index()
    
    df_source.columns = [SOURCE, DESTINATION, MEASURE, TIMESTAMP]
    df_destination.columns = [DESTINATION, SOURCE, MEASURE, TIMESTAMP]

    df_egonet = pd.concat([df_source, df_destination], axis=0)[[SOURCE, DESTINATION, MEASURE, TIMESTAMP]].reset_index(drop=True)
    df_egonet.dropna(inplace=True)
    
    df_egonet[TIMESTAMP] = df_egonet[TIMESTAMP].astype('datetime64[s]')
    df_egonet.sort_values(by=TIMESTAMP, inplace=True)
    df_egonet.reset_index(drop=True, inplace=True)

    df_source=[]
    df_destination=[]

    df_temporal_features = get_degrees(df_egonet, df_temporal_features)
    df_temporal_features = get_temporal_features(df_egonet, df_temporal_features)
    fig_cum_sum_in_degree = cum_sum_in_degree(df_temporal_features)
    fig_cum_sum_out_degree = cum_sum_out_degree(df_temporal_features)
    fig_cum_sum_in_call_count, fig_cum_sum_out_call_count = get_cum_call_counts(df_temporal_features)

    return df_temporal_features, fig_cum_sum_in_degree, fig_cum_sum_out_degree, fig_cum_sum_in_call_count, fig_cum_sum_out_call_count


def get_node_curves(df_temporal_features, query_hash):
    global all_prefix

    # Incoming calls
    columns = []
    for p in all_prefix: columns.append('in_sum_measure__' + p)

    # idx = df_temporal_features.sort_values(by=columns[-1], ascending=False).index
    fig_incoming = plt.figure(figsize=[10,6])


    for qhash in [query_hash]:
        row = df_temporal_features[df_temporal_features[NODE_ID] == qhash].index[0]
        plt.bar(x = np.arange(len(columns)),
                height=df_temporal_features[columns].iloc[row].values)

    plt.grid()
    plt.title('Cumulated incoming call duration per hour in 48 hours')
    plt.ylabel('total in call duration (seconds)')
    plt.xlabel('timestamp (hours)')

    # Outgoing calls
    columns = []
    for p in all_prefix: columns.append('out_sum_measure__' + p)

    # idx = df_temporal_features.sort_values(by=columns[-1], ascending=False).index
    fig_outgoing = plt.figure(figsize=[10,6])


    for qhash in [query_hash]:
        row = df_temporal_features[df_temporal_features[NODE_ID] == qhash].index[0]
        plt.bar(x = np.arange(len(columns)),
                height=df_temporal_features[columns].iloc[row].values)

    plt.grid()
    plt.title('Cumulated outgoing call duration per hour in 48 hours')
    plt.ylabel('total out call duration (seconds)')
    plt.xlabel('timestamp (hours)')


    return fig_incoming, fig_outgoing