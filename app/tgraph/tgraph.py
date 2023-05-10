import sys
import argparse
import pandas as pd
import networkx as nx
import numpy as np
import fn # feature names

import data_network
import static_graph
import static_weighted_graph
import temporal_graph
        
class TGraph():
    def __init__(self, filename, source=fn.SOURCE, destination=fn.DESTINATION, measure=None, timestamp=None):
        
        self.data_network = data_network.DataNetwork(filename=filename, source=source, destination=destination, measure=measure, timestamp=timestamp)
    
        sg = static_graph.StaticGraph(self.data_network)
        sg.graph2vec()

        swg = static_weighted_graph.StaticWeightedGraph(self.data_network)
        swg.graph2vec()

        if fn.TIMESTAMP in self.data_network.headers:
            tg = temporal_graph.TemporalGraph(self.data_network)
            tg.graph2vec()

        # if verbose > 1:
        self.my_print()
        print("\n\n ----")
        self.print_to_csv("allFeatures_nodeVectors.csv")
        print(" check the file nodeVectors.csv ")

        print("---------------")
            
            
    def my_print(self):
        """
        for debugging - just prints the pandas frames
        :return:
        """
        print("")
        print("---- echoing the input ----")
        print(self.data_network.df)

        print("\n")
        print("---- RESULT (placeholder): data frame  of nodes -------------")
        print(self.data_network.df_nodes)
        print(self.data_network.df_nodes.values)
    
    
    def print_to_csv(self, out_file_name):
        """
        prints the df_nodes dataframe as a csv file,
        ready for nd_cloud processing
        :param out_file_name:
        :return:
        """
        self.data_network.df_nodes.to_csv(out_file_name, index=False)


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="analysis for static graphs")
    parser.add_argument("-v", "--verbose",
                        help="level of verbosity (-v [-v ...])",
                        action="count",
                        default=0)
    parser.add_argument("filename", help="input file csv[.gz]")
    parser.add_argument("-s", "--source",
                        help="SOURCE column name",
                        default=fn.SOURCE)
    parser.add_argument("-d", "--destination",
                        help="DESTINATION column name",
                        default=fn.DESTINATION)
    parser.add_argument("-m", "--measure",
                        help="MEASURE column name",
                        default=None)
    parser.add_argument("-t", "--timestamp",
                        help="TIMESTAMP column name",
                        default=None)
    
    args        = parser.parse_args()
    verbose     = args.verbose
    filename    = args.filename
    source      = args.source
    destination = args.destination
    measure     = args.measure
    timestamp   = args.timestamp

    if verbose > 0:
        print("    *** verbose = ", verbose)
        print("    *** filename = ", filename)
    
    # Run tgraph with input parameters
    tgraph = TGraph(filename, source=source,
                    destination=destination, measure=measure, timestamp=timestamp)
    