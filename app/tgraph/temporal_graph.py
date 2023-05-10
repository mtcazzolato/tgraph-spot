import sys
import argparse
import pandas as pd
import networkx as nx
import numpy as np
import fn # feature names

        
class TemporalGraph():
    
    def __init__(self, data_network):
        self.data_network = data_network
        assert fn.TIMESTAMP in self.data_network.headers, "TIMESTAMP column not found"
        
    def graph2vec(self):
        self.get_in_features()
        self.get_out_features()
        
    def get_in_features(self):
        """
        extract features from the destination nodes
        """
        self.fill_features(node_direction=fn.DESTINATION, prefix='in_')

    def get_out_features(self):
        """
        extract features from the source nodes
        """
        self.fill_features(node_direction=fn.SOURCE, prefix='out_')
    
    
    def fill_features(self, node_direction=fn.SOURCE, prefix='out_'):
        """
        extract in/out features of iat from TIMESTAMP column,
        grouped by SOURCE or DESTINATION
        populate df_nodes with iat features per node
        """
        
        # First: sort elements by source and timestamp
        df_aux = self.data_network.df.sort_values(by=[node_direction, fn.TIMESTAMP]).reset_index(drop=True)
        
        # Group by source, get the difference (iat) between ts of every hash, convert it to timedelta
        df_aux['diff_ts'] = df_aux.groupby(by=[node_direction])[fn.TIMESTAMP].diff().astype("timedelta64[s]")
        
        # Creates groups by ahash values, get the iat for every row
        group = df_aux.groupby(by=[node_direction], axis=0)['diff_ts']
        
        # Aggregate iat per group/hash
        df_iat_measure = pd.DataFrame()
        df_iat_measure[prefix+fn.AVG_IAT]     = group.mean()
        df_iat_measure[prefix+fn.MAD_IAT]     = group.mad()
        df_iat_measure[prefix+fn.MEDIAN_IAT]  = group.median()
        df_iat_measure[prefix+fn.STD_IAT]     = group.std(ddof=0)
        df_iat_measure[prefix+fn.MIN_IAT]     = group.min()
        df_iat_measure[prefix+fn.MAX_IAT]     = group.max()
        df_iat_measure[prefix+fn.QUANT25_IAT] = group.quantile(q=0.25)
        df_iat_measure[prefix+fn.QUANT50_IAT] = group.quantile(q=0.5)
        df_iat_measure[prefix+fn.QUANT75_IAT] = group.quantile(q=0.75)

        df_iat_measure.reset_index(inplace=True)
        self.data_network.df_nodes = self.data_network.df_nodes.merge(df_iat_measure, left_on=fn.NODE_ID,
                                                                      right_on=node_direction, how='left')
        self.data_network.df_nodes = self.data_network.df_nodes.drop(columns=[node_direction])
        self.data_network.df_nodes = self.data_network.df_nodes.fillna(0)
        