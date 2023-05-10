import sys
import argparse
import pandas as pd
import networkx as nx
import numpy as np
import fn # feature names

class StaticGraph():

    def __init__(self, data_network):
        self.data_network = data_network
        
    def graph2vec(self):
        self.fill_in_degree()
        self.fill_out_degree()
        self.fill_main_core()
    
    def fill_out_degree(self):
        out_degs = [(node, val) for (node, val) in self.data_network.G.out_degree()] #type is list 
        out_degs_df = pd.DataFrame(out_degs, columns=[fn.SOURCE, fn.OUT_DEGREE])
        self.data_network.df_nodes = self.data_network.df_nodes.merge(out_degs_df, left_on=fn.NODE_ID, right_on=fn.SOURCE,how='left')
        self.data_network.df_nodes = self.data_network.df_nodes.drop(columns=[fn.SOURCE])

    def fill_in_degree(self):
        in_degs = [(node, val) for (node, val) in self.data_network.G.in_degree()]
        in_degs_df = pd.DataFrame(in_degs, columns=[fn.SOURCE, fn.IN_DEGREE])
        self.data_network.df_nodes = self.data_network.df_nodes.merge(in_degs_df, left_on=fn.NODE_ID, right_on=fn.SOURCE, how='left')
        self.data_network.df_nodes = self.data_network.df_nodes.drop(columns=[fn.SOURCE])

    def fill_main_core(self):
        G = self.data_network.G.copy()
        G.add_edges_from(nx.selfloop_edges(self.data_network.G))
        G.remove_edges_from(nx.selfloop_edges(G))
        cores = nx.core_number(G)
        core_df = pd.DataFrame.from_dict(cores,orient='index',columns=[fn.CORE])
        core_df[fn.NODE_ID] = core_df.index
        self.data_network.df_nodes = self.data_network.df_nodes.merge(core_df, on=fn.NODE_ID, how='left')
