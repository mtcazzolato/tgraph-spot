import sys
import argparse
import pandas as pd
import networkx as nx
import numpy as np
import fn # feature names


class StaticWeightedGraph():
    
    def __init__(self, data_network):
        self.data_network = data_network
        # assert fn.MEASURE in self.data_network.headers, "MEASURE column not found"
    
    def graph2vec(self):
        self.fill_weighted_degree()
        self.fill_weighted_in_degree()
        self.fill_weighted_out_degree()
        
        if fn.MEASURE in self.data_network.headers:
            self.get_in_measures()
            self.get_out_measures()
    
    def fill_weighted_in_degree(self):
        in_degs = [(node, val) for (node, val) in self.data_network.G.in_degree(weight=fn.MEASURE)]
        in_degs_df = pd.DataFrame(in_degs, columns=[fn.SOURCE, fn.WEIGHTED_IN_DEGREE])
        self.data_network.df_nodes = self.data_network.df_nodes.merge(in_degs_df, left_on=fn.NODE_ID, right_on=fn.SOURCE, how='left')
        self.data_network.df_nodes = self.data_network.df_nodes.drop(columns=[fn.SOURCE])

    def fill_weighted_out_degree(self):
        out_degs = [(node, val) for (node, val) in self.data_network.G.out_degree(weight=fn.MEASURE)]
        out_degs_df = pd.DataFrame(out_degs, columns=[fn.SOURCE, fn.WEIGHTED_OUT_DEGREE])
        self.data_network.df_nodes = self.data_network.df_nodes.merge(out_degs_df, left_on=fn.NODE_ID, right_on=fn.SOURCE, how='left')
        self.data_network.df_nodes = self.data_network.df_nodes.drop(columns=[fn.SOURCE])

    def fill_weighted_degree(self):
        degs = [(node, val) for (node, val) in self.data_network.G.degree(weight=fn.MEASURE)]
        degs_df = pd.DataFrame(degs, columns=[fn.SOURCE, fn.WEIGHTED_DEGREE])
        self.data_network.df_nodes = self.data_network.df_nodes.merge(degs_df, left_on=fn.NODE_ID, right_on=fn.SOURCE, how='left')
        self.data_network.df_nodes = self.data_network.df_nodes.drop(columns=[fn.SOURCE])
    
    def get_in_measures(self):
        """
        extract features from the destination nodes
        """
        self.fill_measure_statistics(node_direction=fn.DESTINATION, prefix='in_')

    def get_out_measures(self):
        """
        extract features from the source nodes
        """
        self.fill_measure_statistics(node_direction=fn.SOURCE, prefix='out_')
        
    def fill_measure_statistics(self, node_direction=fn.SOURCE, prefix='out_'):
        
        # Creates groups by ahash values, get the duration of every row
        group = self.data_network.df.groupby(by=[node_direction], axis=0)[fn.MEASURE]
        
        df_measure = pd.DataFrame()
        df_measure[prefix+fn.AVG_MEASURE]     = group.mean()
        df_measure[prefix+fn.MAD_MEASURE]     = group.mad()
        df_measure[prefix+fn.MEDIAN_MEASURE]  = group.median()
        df_measure[prefix+fn.STD_MEASURE]     = group.std(ddof=0)
        df_measure[prefix+fn.MIN_MEASURE]     = group.min()
        df_measure[prefix+fn.MAX_MEASURE]     = group.max()
        df_measure[prefix+fn.QUANT25_MEASURE] = group.quantile(q=0.25)
        df_measure[prefix+fn.QUANT50_MEASURE] = group.quantile(q=0.5)
        df_measure[prefix+fn.QUANT75_MEASURE] = group.quantile(q=0.75)
        df_measure[prefix+fn.SUM_MEASURE]     = group.sum()
        df_measure[prefix+fn.TOTAL_COUNT]      = group.size()
        
        df_measure.reset_index(inplace=True)
        self.data_network.df_nodes = self.data_network.df_nodes.merge(df_measure, left_on=fn.NODE_ID,
                                                                      right_on=node_direction, how='left')
        self.data_network.df_nodes = self.data_network.df_nodes.drop(columns=[node_direction])
        self.data_network.df_nodes = self.data_network.df_nodes.fillna(0)

    