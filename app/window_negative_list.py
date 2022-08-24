# Author: Mirela Cazzolato
# Date: August 2022
# Goal: Window to manage a list of negative (blocked) nodes
# =======================================================================

from os import remove

from pyparsing import col
import streamlit as st
import pandas as pd
import os.path

NODE_ID="node_ID"

def read_file(file):
    """
    Read input negative-list file
    
    Parameters
    ----------
    file: str
        path of the input file with negative-list
    """
    
    global df_negative_list

    if os.path.isfile(file):
        df_negative_list = pd.read_csv(file)
    else:
        print("Negative-list file doesn't exist. Creating a new one.")
        df_negative_list = pd.DataFrame(columns=[NODE_ID])
        df_negative_list.to_csv(file, index=False)
        
    df_negative_list = df_negative_list.astype(str)
    

def add_node_to_negativelist(node_id, file):
    global df_negative_list

    df_negative_list.loc[len(df_negative_list)] = [str(node_id)]
    df_negative_list = df_negative_list.drop_duplicates(keep="first", inplace=False, ignore_index=True)    
    df_negative_list.to_csv(file, index=False)


def remove_node_from_negativelist(node_id, file):
    global df_negative_list

    df_negative_list.drop(df_negative_list[df_negative_list[NODE_ID].astype(str) == str(node_id)].index, inplace=True)
    df_negative_list.to_csv(file, index=False)


def launch_w_negative_list():
    """
    Launch window to manage negative-list of nodes
    """

    global df_negative_list
    
    st.write(
        """
        # Negative-list of nodes
        ### Manage list of nodes that must be ignored by the application
        """
    )
    with st.expander(label="Input file", expanded=True):
        # TODO: add option to load other files
        # file_source = st.text_input("Enter input file path:")

        use_example_file = st.checkbox("Use negative-list file \"data/negative-list.csv\"",
                                    True,
                                    disabled=True,
                                    help="Use in-built example file for negative-list nodes in \"data_sample\/negative-list.csv\"")

    with st.expander(label="Listed negative-nodes", expanded=True):
        if (use_example_file and use_example_file != ""):
            file_source = "data/negative-list.csv"

        if (file_source is not None and file_source != ""):
            st.write("Selected file:", file_source)
            
            read_file(file_source)            
            
            col1, col2 = st.columns([1, 1])

            with col1:
                new_negative_list_item = st.text_input("Enter Node ID to add to the negative-list:")

                if st.button("Add node to list"):
                    add_node_to_negativelist(node_id=new_negative_list_item, file=file_source)
                    st.success("Node added to the negative-list file.")
            
            with col2:
                node_to_remove = st.text_input("Enter Node ID to delete from negative-list:")
                if st.button("Delete node from list"):
                    remove_node_from_negativelist(node_id=node_to_remove, file=file_source)
                    st.success("Node deleted from the negative-list file.")

        st.write(df_negative_list.astype(str))
