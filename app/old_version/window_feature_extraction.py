# Author: Mirela Cazzolato
# Date: July 2022
# Goal: Window to select raw data and generate t-graph features
# =======================================================================

import streamlit as st
import pandas as pd

# t-graph modules
import tgraph.static_graph as SG
import tgraph.temporal_graph as TG

NODE_ID="node_ID"

def read_file_header(file):
    """
    Read columns from input file
    
    Parameters
    ----------
    file: str
        path of the input file with raw data
    """

    global df_dataset, nrows, columns
    
    columns = pd.read_csv(file, nrows=1).columns.tolist()
    

def populate_selectbox_columns():
    """
    Populate select box with available columns
    to select input parameters for t-graph
    """

    global source, destination, measure, timestamp, columns

    mcol1_features, mcol2_features = st.columns(2)

    with mcol1_features:
        source = st.selectbox(
                        "Select SOURCE column",
                        options=columns,
                        index=0)
        destination = st.selectbox(
                        "Select DESTINATION column",
                        options=columns,
                        index=1)
    
    with mcol2_features:
        measure = st.selectbox(
                        "Select MEASURE column",
                        options=columns,
                        index=2)
        timestamp = st.selectbox(
                        "Select TIMESTAMP column",
                        options=columns,
                        index=3)
    
def run_t_graph(file, source, destination, measure, timestamp):
    """
    Run t-graph
    
    Parameters
    ----------
    file: str
        path of the input file with raw data
    source: str
        input column for source
    destination: str
        input column for destination
    measure: str
        input column for measure
    timestamp: str
        input column for timestamp
    """

    global df_all_features
    
    print('======================', file)
    
    # Get static features
    sg = SG.StaticGraph(filename=file,
                        source=source,
                        destination=destination,
                        measure=measure)
    sg.my_print()
    
    # Get temporal features
    tg = TG.TemporalGraph(filename=file,
                          source=source,
                          destination=destination,
                          measure=measure,
                          timestamp=timestamp)
    tg.my_print()
    
    # Join static and temporal features
    df_all_features = sg.df_nodes.set_index(NODE_ID).join(tg.df_nodes.set_index(NODE_ID)).reset_index()
    df_all_features.fillna(0)
    
    # Save output features
    print("\n\n ----")
    df_all_features.to_csv("data/allFeatures_nodeVectors.csv", index=False)
    print("Check the file \"data/allFeatures_nodeVectors.csv\"")
    
def launch_w_feature_extraction():
    """
    Launch window to extract t-graph features
    """

    global source, destination, measure, timestamp, columns, df_all_features
    df_all_features = None

    st.write(
        """
        # Feature extraction
        ### Extract features using t-graph
        """
    )

    columns = []
    
    # file_feature = st.file_uploader(label="Select input file",
    #                                 type=['txt', 'csv'])
    
    file_source = st.text_input("Enter input file path:")
    use_example_file = st.checkbox("Use example file",
                                False,
                                help="Use in-built example file to demo the app")

    if use_example_file:
            file_source = "data/sample_raw_data.csv"

    if file_source is not None and file_source != '':
        st.write("Selected file:", file_source)
        print(file_source)
        
        read_file_header(file_source)

        with st.expander(label="t-graph parameters", expanded=True):
            populate_selectbox_columns()
        
            if st.button('Run t-graph'):
                run_t_graph(file_source, source, destination, measure, timestamp)
        
                st.success("Finished extracting features. Check file *\'data/allFeatures_nodeVectors.csv\'*")
                st.write("### Extracted features", df_all_features.head())
