# Author: Mirela Cazzolato
# Date: August 2022
# Goal: Window to select raw data and generate t-graph features
# =======================================================================

from os import remove

from pyparsing import col
import streamlit as st
import pandas as pd
import os.path

NODE_ID="node_ID"

def read_file(file):
    """
    Read input blocked-list file
    
    Parameters
    ----------
    file: str
        path of the input file with blocked-list
    """
    
    global df_blocked_list

    if os.path.isfile(file):
        df_blocked_list = pd.read_csv(file)
    else:
        print("Blocked-list file doesn't exist. Creating a new one.")
        df_blocked_list = pd.DataFrame(columns=[NODE_ID])
        df_blocked_list.to_csv(file, index=False)


def add_node_to_blockedlist(node_id, file):
    global df_blocked_list

    df_blocked_list.loc[len(df_blocked_list)] = [node_id]
    df_blocked_list.drop_duplicates(inplace=True)
    df_blocked_list.to_csv(file, index=False)


def remove_node_from_blockedlist(node_id, file):
    global df_blocked_list

    df_blocked_list.drop(df_blocked_list[df_blocked_list[NODE_ID] == node_id].index, inplace=True)
    df_blocked_list.to_csv(file, index=False)


def launch_w_blocked_list():
    """
    Launch window to manage blocked-list of nodes
    """

    global df_blocked_list
    
    st.write(
        """
        # Blocked-list of nodes
        ### Manage list of nodes that must be ignored by the application
        """
    )
    with st.expander(label="Input file", expanded=True):
        # TODO: add option to load other files
        # file_source = st.text_input("Enter input file path:")

        use_example_file = st.checkbox("Use blocked-list file \"data_sample/blocked-list.csv\"",
                                    True,
                                    disabled=True,
                                    help="Use in-built example file for blocked-list nodes in \"data_sample\/blocked-list.csv\"")

    with st.expander(label="Listed blocked-nodes", expanded=True):
        if (use_example_file and use_example_file != ""):
            file_source = "data_sample/blocked-list.csv"

        if (file_source is not None and file_source != ""):
            st.write("Selected file:", file_source)
            
            read_file(file_source)            
            
            col1, col2 = st.columns([1, 1])

            with col1:
                new_blocked_list_item = st.text_input("Enter Node ID to add to the blocked-list:")

                if st.button("Add node to list"):
                    add_node_to_blockedlist(node_id=new_blocked_list_item, file=file_source)
                    st.success("Node added to the blocked-list file.")
            
            with col2:
                node_to_remove = st.text_input("Enter Node ID to delete from blocked-list:")
                if st.button("Delete node from list"):
                    remove_node_from_blockedlist(node_id=node_to_remove, file=file_source)
                    st.success("Node deleted from the blocked-list file.")

        st.write(df_blocked_list)