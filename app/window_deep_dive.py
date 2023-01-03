# Author: Mirela Cazzolato
# Date: July 2022
# Goal: Window to deep dive on selected nodes and check temporal features
# =======================================================================

import streamlit as st

import os.path

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import plotly.graph_objs as go
import plotly.offline as py
from ipywidgets import interactive, HBox, VBox
from streamlit_plotly_events import plotly_events

from collections import Counter
from cross_associations.matrix import Matrix
from cross_associations.plots import plot_shaded, plot_spy
import cross_associations.cluster_search as cs

from temporal_features.egonet_deep_dive import get_curves, get_node_curves

file_negative_list="data/negative-list.csv"
NODE_ID="node_ID"
SOURCE="source"
DESTINATION="destination"
MEASURE="measure"
TIMESTAMP="timestamp"
UNIQUE_DATES=[]
plotly_width="100%"
plotly_height=800
flag_graph_constructed=False
generate_egonet=False
ready_for_deep_dive=False
fig_adj_matrix=None
fig_cross_associations=None
fig_cum_sum_in_degree=None
fig_cum_sum_out_degree=None
fig_cum_sum_in_call_count=None
fig_cum_sum_out_call_count=None
df_temporal_features=None

# TODO: add config file with preset columns/cases
preset_features = ("in_degree, out_degree, core",
                   "weighted_in_degree, weighted_out_degree, core",
                   "out_degree, in_degree, core, in_median_iat, out_median_measure",
                   "out_degree, in_degree, in_median_measure, out_median_measure",
                   "out_call_count, in_call_count, weighted_degree, core")

preset_feature_columns = (["in_degree", "out_degree", "core"],
                          ["weighted_in_degree", "weighted_out_degree", "core"],
                          ["out_degree", "in_degree", "core", "in_median_iat", "out_median_measure"],
                          ["out_degree", "in_degree", "in_median_measure", "out_median_measure"],
                          ["out_call_count", "in_call_count", "weighted_degree", "core"])


def read_files(file_features, file_graph):
    """
    Read files with features and raw data
    
    Parameters
    ----------
    file_features: str
        path of the input file with t-graph features
    file_graph: str
        path of the input file with raw data
    """

    global df, df_graph

    # File with input features
    df = pd.read_csv(file_features)

    # Select nodes with more than 5 calls
    # df = df[(df['in_call_count'] + df['out_call_count']) >= 5].reset_index(drop=True)
    
    # File with raw data (source, destination, measure, timestamp) to generate the graph
    df_graph = pd.read_csv(file_graph)

    filter_negative_list()
    
    df.fillna(0, inplace=True)
    flag_graph_constructed=False


def adjust_input_data():
    """
    Adjust inpute data: cast timestamp column to datetime64,
    remove calls with zero duration,
    and remove calls with same source and destination.
    """
    
    global df_graph, SOURCE, DESTINATION, MEASURE, TIMESTAMP, UNIQUE_DATES

    df_graph[TIMESTAMP] = df_graph[TIMESTAMP].astype('datetime64[s]')
    df_graph.sort_values(by=TIMESTAMP, inplace=True)
    df_graph.reset_index(drop=True, inplace=True)

    df_graph[MEASURE] = pd.to_numeric(df_graph[MEASURE], errors='coerce')
    df_graph.dropna(subset=[MEASURE])
    df_graph = df_graph[df_graph[SOURCE] != df_graph[DESTINATION]]
    df_graph = df_graph[df_graph[MEASURE] > 0]
    df_graph = df_graph.groupby([TIMESTAMP, SOURCE, DESTINATION]).sum().add_suffix('').reset_index()

    UNIQUE_DATES = df_graph[TIMESTAMP].dt.date.unique()


def update_sidebar():
    """
    Add options to the sidebar
    """

    global df, ego_radius, max_nodes_association_matrix
    with st.sidebar:
        ego_radius = st.number_input(
            "EgoNet radius parameter",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            format="%d",
            help="""Radius of the egonet. It
                  may take a while to run. Use with caution."""
        )
    
        max_nodes_association_matrix = st.number_input(
            "Max #nodes for matrix association",
            min_value=1,
            # max_value=5,
            value=500,
            step=20,
            format="%d",
            help="""Maximum number of nodes allowed to generate the
                    matrix association plot. With high #nodes, it
                    may take a while to run. Use with caution."""
        )


def populate_selectbox_graph():
    """
    Populate select box with available columns for the graph
    """

    global df_graph, SOURCE, DESTINATION, MEASURE, TIMESTAMP
    mcol1_graph, mcol2_graph, mcol3_graph, mcol4_graph = st.columns(4)

    with mcol1_graph:
        SOURCE = st.selectbox(
                        "Select SOURCE column",
                        options=df_graph.columns,
                        index=0)
    with mcol2_graph:
        DESTINATION = st.selectbox(
                        "Select DESTINATION column",
                        options=df_graph.columns,
                        index=1)
    with mcol3_graph:
        MEASURE = st.selectbox(
                        "Select MEASURE column",
                        options=df_graph.columns,
                        index=2)
    with mcol4_graph:
        TIMESTAMP = st.selectbox(
                        "Select TIMESTAMP column",
                        options=df_graph.columns,
                        index=3)


def construct_graph():
    """
    Construct graph for the deep dive with selected attributes
    """

    adjust_input_data()

    global G, SOURCE, DESTINATION, MEASURE, flag_graph_constructed, generate_egonet
    G = nx.from_pandas_edgelist(df_graph, source=SOURCE,
                                            target=DESTINATION,
                                            edge_attr=MEASURE,
                                            create_using=nx.DiGraph())

    flag_graph_constructed=True
    generate_egonet = False


def filter_negative_list():
    """
    Filter out nodes included in the negative list
    """

    global df, df_graph

    if os.path.isfile(file_negative_list):
        # Remove nodes in the negative-list
        df_negative_list = pd.read_csv(file_negative_list)

        df = df[~df[NODE_ID].isin(list(df_negative_list[NODE_ID].values))].reset_index(drop=True)
        
        df_graph = df_graph[~df_graph[SOURCE].isin(list(df_negative_list[NODE_ID].values))]
        df_graph = df_graph[~df_graph[DESTINATION].isin(df_negative_list[NODE_ID].values)].reset_index(drop=True)
    else:
        print('No negative list found.')


def plot_scatter_matrix(columns):
    """
    Plot interactive scatter plot with the selected columns as features

    Parameters
    ----------

    columns: list
        list of features to show in the scatter matrix
    """

    global df
    truecolor = '#f95a10'
    falsecolor = 'blue'
    linecolor = 'white'

    dimensions=[]
    for c in columns:
        # Construct dict with column names and labels
        dimensions.append(dict(label=c.replace('_', ' '), values=np.log10(df[c]+1)))

    colors = pd.Series(data=[falsecolor] * len(df))
    
    fig = go.Figure(data=go.Splom(
            dimensions = dimensions,
            customdata = df[NODE_ID],
            hovertemplate="<br>".join([
                        "%{xaxis.title.text}: %{x}",
                        "%{yaxis.title.text}: %{y}",
                        "hash: %{customdata}",
            ]),
            showlegend=False, #Show legend entries later on!
            showupperhalf=False, # remove plots on diagonal
            # text=df_label[LABEL],
            marker=dict(color=list(colors),
                        showscale=False,
                        line_color=linecolor,
                        line_width=0.8,
                        size=8,
                        opacity=0.5
            ),
    ))
    
    fig.update_traces(unselected_marker=dict(opacity=0.1, size=5),
                selected_marker=dict(size=10, opacity=0.9),
                selector=dict(type='splom'),
                diagonal_visible=False)
    
    fig.update_layout(
        title='Scatter matrix with graph information',
        dragmode='select',
        hovermode='closest',
    ) 
    
    return fig


def get_egonet(G, suspicious_nodes, radius=1, column=''):
    """
    Compose a graph with the egonets of a given set of suspicious nodes.
    Return the subgraph of the composed egonets and the index of
    suspicious nodes inside the subgraph

    Parameters
    ----------
    G: nx.Graph
        graph with all nodes to extract the EgoNets from
    suspicious_nodes: list
        list of suspicious nodes
    radius: int
        step of the EgoNet
    """
        
    final_G = nx.empty_graph(create_using=nx.DiGraph())

    for ego_node in suspicious_nodes:
        # create ego network
        hub_ego = nx.ego_graph(G, ego_node, radius=radius, distance='weight', undirected=True)
        final_G = nx.compose(final_G, hub_ego)

    idx_suspicious_nodes = []
    for node in suspicious_nodes:
        idx_suspicious_nodes.append(list(np.where(pd.DataFrame(data=final_G.nodes()) == node)[0])[0])
    
    return final_G, idx_suspicious_nodes


def plot_adj_matrix(G, markersize=2, compute_associations=True):
    """
    Plot adjacency matrix of the given graph

    Parameters
    ----------
    G: nx.Graph
        graph to show
    markersize: int
        size of the marker to show the correspondences in the plot
    compute_associations: boolean
        informs if the matrix associations should be generated
    """

    fig, ax = plt.subplots()
    plt.spy(nx.adjacency_matrix(G), markersize=markersize)
    ax.set_aspect('equal', adjustable='box')
    plt.ylabel('source')
    plt.xlabel('destination')
    # plt.tight_layout()

    if compute_associations: # Compute matrix associations
        fig_cross_associations = plot_cross_associations(nx.adjacency_matrix(G))
    else:
        return fig  # Skip matrix associations and return only the original matrix

    return fig, fig_cross_associations


def plot_cross_associations(sparse_matrix, markersize=2):
    """
    Find cross associations in the given adjacency matrix

    Parameters
    ----------
    sparse_matrix: matrix
        adjacency matrix to find cross associations
    markersize: int
        size of the marker to show the correspondences in the plot
    """

    # Run algorithm to find cross associations
    matrix = Matrix(sparse_matrix)
    cluster_search = cs.ClusterSearch(matrix)
    cluster_search.run()
    col_counter = Counter(cluster_search._matrix.col_clusters)
    row_counter = Counter(cluster_search._matrix.row_clusters)
    height, width = np.shape(cluster_search._matrix.matrix)

    # Plot results
    fig_cross_associations = plt.figure()
    ax1 = fig_cross_associations.add_subplot(111)
    ax1.set_xlim(right=width)
    ax1.set_ylim(top=height)
    ax1.set_ylabel('source')
    ax1.set_xlabel('destination')
    col_offset = 0
    for col, col_len in col_counter.most_common():
        row_offset = 0
        for row, row_len in row_counter.most_common():

            ax1.add_patch(
                patches.Rectangle(
                    (col_offset * 1.0, row_offset * 1.0),
                    (col_offset + col_len) * 1.0,
                    (row_offset + row_len) * 1.0,
                    facecolor=None,
                    color=None,
                    edgecolor="#0000FF",
                    linewidth=0.5,
                    fill=False
                )
            )
            row_offset += row_counter[row]
        col_offset += col_counter[col]

    ax1.spy(cluster_search._matrix.transformed_matrix, markersize=markersize, color='black')
    ax1.set_aspect('equal', adjustable='box')
    return fig_cross_associations


def set_fig_adj_matrix(fig_adj_matrix_):
    """
    Set the figure of adjacency matrix for the global variable
    
    Parameters
    ----------
    fig_adj_matrix_: Figure
        Figure of adjacency matrix
    """
    
    global fig_adj_matrix
    fig_adj_matrix = fig_adj_matrix_


def set_fig_cross_associations(fig_cross_associations_):
    """
    Set the figure of adjacency matrix with cross associations
    for the global variable
    
    Parameters
    ----------
    fig_cross_associations: Figure
        Figure of cross associations
    """
    
    global fig_cross_associations
    fig_cross_associations = fig_cross_associations_


def get_fig_adj_matrix():
    """
    Get the figure of adjacency matrix from the global variable
    """
    
    global fig_adj_matrix
    return fig_adj_matrix


def get_fig_cross_associations():
    """
    Get the figure of adjacency matrix with cross associations
    from the global variable
    """
    
    global fig_cross_associations
    return fig_cross_associations


def set_temporal_figures(fig_cum_sum_in_degree_, fig_cum_sum_out_degree_, fig_cum_sum_in_call_count_, fig_cum_sum_out_call_count_):
    """
    Set the figures of temporal features
    for the global corresponding variables
    
    Parameters
    ----------
    fig_cum_sum_in_degree_:
        Figure with cumulated in degree per node
    fig_cum_sum_out_degree_:
        Figure with cumulated out degree per node
    fig_cum_sum_in_call_count_:
        Figure with cumulated in call count per node
    fig_cum_sum_out_call_count_:
        Figure with cumulated out call count per node
    
    """
    
    global fig_cum_sum_in_degree, fig_cum_sum_out_degree, fig_cum_sum_in_call_count, fig_cum_sum_out_call_count
    fig_cum_sum_in_degree = fig_cum_sum_in_degree_
    fig_cum_sum_out_degree = fig_cum_sum_out_degree_
    fig_cum_sum_in_call_count = fig_cum_sum_in_call_count_
    fig_cum_sum_out_call_count = fig_cum_sum_out_call_count_


def get_fig_cum_sum_in_degree():
    """
    Get figure with cumulated in degree per node
    """
    
    global fig_cum_sum_in_degree
    return fig_cum_sum_in_degree


def get_fig_cum_sum_out_degree():
    """
    Get Figure with cumulated out degree per node
    """
    
    global fig_cum_sum_out_degree
    return fig_cum_sum_out_degree


def get_fig_cum_sum_in_call_count():
    """
    Get figure with cumulated in call count per node
    """
    
    global fig_cum_sum_in_call_count
    return fig_cum_sum_in_call_count


def get_fig_cum_sum_out_call_count():
    """
    Get figure with cumulated out call count per node
    
    """
    
    global fig_cum_sum_out_call_count
    return fig_cum_sum_out_call_count


def set_df_temporal_features(df_temporal_features_):
    """
    Set dataframe with temporal features
    for the global variable
    
    Parameters
    ----------
    df_temporal_features_:
        Dataframe with temporal features
    """
    
    global df_temporal_features
    df_temporal_features=df_temporal_features_


def get_df_temporal_features():
    """
    Get dataframe with temporal features
    from the global variable
    """
    
    global df_temporal_features
    return df_temporal_features


def launch_deep_dive():
    """
    Launch window to visualize features interactively
    and do deep dive on selected nodes
    """
    
    global generate_egonet, ready_for_deep_dive
    global get_cross_associations

    st.write(
        """
        # Deep dive on suspicious nodes
        """
    )

    with st.expander("Input parameters", expanded=True):

        selected_points = []
        col1_file_selection, col2_file_selection = st.columns(2)

        with col1_file_selection:
            file_features = st.file_uploader(label="Select a file with features",
                                    type=['txt', 'csv'])

            use_example_features = st.checkbox("Use example file with features",
                                    False,
                                    help="Use in-built example file with features to demo the app")

        with col2_file_selection:
            file_graph = st.file_uploader(label="Select a file with raw data",
                                    type=['txt', 'csv'])

            use_example_graph = st.checkbox("Use example file with raw data",
                                    False,
                                    help="Use in-built example file with raw data to demo the app")

        if use_example_features and not file_features:
            file_features = "data/allFeatures_nodeVectors.csv"

        if use_example_graph and not file_graph:
            file_graph = "data/sample_raw_data.csv"

        if file_features and file_graph:
            read_files(file_features, file_graph)

            populate_selectbox_graph()

            if st.button('Construct graph'):
                construct_graph()

                fig_adj_matrix = None
                fig_cross_associations = None
    
    if flag_graph_constructed:
        update_sidebar()
        
        df_result = pd.DataFrame()
        
        with st.expander("Interactive selection", expanded=True):

            checkbox_custom_columns = st.checkbox("Custom features",
                    help="Select desired features or unselect this option to show pre-set feature combinations.",
                    value=True)

            if checkbox_custom_columns:
                selected_columns = st.multiselect("Choose features to visualize",
                                                    df.columns[1:].values)
            else:
                preset_columns = st.radio("Pre-set feature combinations",
                                            (preset_features))
                
                # TODO: maybe improve this with a loop
                if (preset_columns == preset_features[0]):
                    selected_columns = preset_feature_columns[0]
                elif (preset_columns == preset_features[1]):
                    selected_columns = preset_feature_columns[1]
                elif (preset_columns == preset_features[2]):
                    selected_columns = preset_feature_columns[2]
                elif (preset_columns == preset_features[3]):
                    selected_columns = preset_feature_columns[3]
                elif (preset_columns == preset_features[4]):
                    selected_columns = preset_feature_columns[4]
                
            if (len(selected_columns) > 2):
                fig = plot_scatter_matrix(selected_columns)
                selected_points = plotly_events(fig, select_event=True,
                                                    override_height=plotly_height,
                                                    override_width=plotly_width,)
            
                if len(selected_points) > 0:
                    st.write("Selected nodes:", len(selected_points))
                    df_selected = pd.DataFrame(selected_points)
                    st.dataframe(df.loc[df_selected["pointNumber"].values])

                    if st.button('Get EgoNet'):
                        generate_egonet = True
                        get_cross_associations = True
                        ready_for_deep_dive = False

        with st.expander(label="Deep dive on selected nodes", expanded=True):
            if (generate_egonet == True and len(df_selected) > 0):
                
                st.write("### Adjacency matrix of the generated EgoNet")

                final_G, idx_suspicious_nodes = get_egonet(G,
                                        suspicious_nodes=df.loc[df_selected["pointNumber"].values][NODE_ID],
                                        radius=ego_radius, #2-step way egonet
                                        column="core")
                
                st.write("EgoNet size:", len(final_G.nodes))

                if get_cross_associations:
                    if len(final_G.nodes) < max_nodes_association_matrix:
                        fig_adj_matrix, fig_cross_associations = plot_adj_matrix(G=final_G)
                        set_fig_adj_matrix(fig_adj_matrix_=fig_adj_matrix)
                        set_fig_cross_associations(fig_cross_associations_=fig_cross_associations)
                    else:
                        fig_adj_matrix = plot_adj_matrix(G=final_G, compute_associations=False)
                        set_fig_adj_matrix(fig_adj_matrix_=fig_adj_matrix)

                    #This is to prevent re-generating matrixes all the time                
                    get_cross_associations = False
                    set_temporal_figures(fig_cum_sum_in_degree_=None,
                                         fig_cum_sum_out_degree_=None,
                                         fig_cum_sum_in_call_count_=None,
                                         fig_cum_sum_out_call_count_=None)

                _, col1, _, col2, _ = st.columns([1, 3, 1, 3, 1])
                col1.pyplot(get_fig_adj_matrix())

                if len(final_G.nodes) < max_nodes_association_matrix:
                    col2.pyplot(get_fig_cross_associations())

                st.write("Nodes in the EgoNet:")
                df_result = pd.DataFrame(data=final_G.nodes, columns=[NODE_ID]).set_index(NODE_ID).join(df.set_index(NODE_ID)).reset_index()
                df_result.columns=df.columns
                st.write(df_result)
            
        with st.expander(label="Check calls over time:", expanded=True):
            if (len(df_result) > 0):
                
                col1_dates, col2_dates = st.columns([3, 1])

                with col1_dates:
                    selected_date = st.selectbox(
                            "Select initial date",
                            options=UNIQUE_DATES,
                            index=0)

                with col2_dates:
                        n_days = st.number_input(
                        "#Days to visualize calls",
                        min_value=1,
                        max_value=5,
                        value=1 ,
                        step=1,
                        format="%d",
                        help="""Maximum number of days to visualize the calls,
                                from the selected date. Use with caution, it may
                                take a while to compute.""",
                        on_change=None
                        )

                if st.button("Visualize calls over time"):
                    ready_for_deep_dive = True
                    set_temporal_figures(fig_cum_sum_in_degree_=None,
                                         fig_cum_sum_out_degree_=None,
                                         fig_cum_sum_in_call_count_=None,
                                         fig_cum_sum_out_call_count_=None)

                    if (get_fig_cum_sum_in_degree() is None):
                        print("Generating temporal features...")
                        df_temporal_features, fig_cum_sum_in_degree, fig_cum_sum_out_degree, fig_cum_sum_in_call_count, fig_cum_sum_out_call_count = get_curves(df_result,
                                                        df_raw_data=df_graph,
                                                        measure=MEASURE,
                                                        timestamp=TIMESTAMP,
                                                        source=SOURCE,
                                                        destination=DESTINATION,
                                                        selected_date=selected_date,
                                                        n_days=n_days)
                        set_df_temporal_features(df_temporal_features)
                        set_temporal_figures(fig_cum_sum_in_degree, fig_cum_sum_out_degree, fig_cum_sum_in_call_count, fig_cum_sum_out_call_count)
                        print("Done generating temporal features.")

                if (get_fig_cum_sum_in_degree()):
                    col1, _, col2 = st.columns([3, 1, 3])                    
                    col1.pyplot(get_fig_cum_sum_in_degree())
                    col2.pyplot(get_fig_cum_sum_out_degree())

                    col3, _, col4 = st.columns([3, 1, 3])
                    col3.pyplot(get_fig_cum_sum_in_call_count())
                    col4.pyplot(get_fig_cum_sum_out_call_count())

        with st.expander("Deep dive on selected node:", expanded=True):
            if (ready_for_deep_dive):

                # Deep dive on selected node
                selected_node = st.selectbox("Select node from EgoNet",
                                                    options=df_result,
                                                    index=0)

                if st.button("Deep dive on selected node"):
                    st.write("Deep dive on "+ str(selected_node))
                    fig_selected_node_incoming, fig_selected_node_outgoing = get_node_curves(get_df_temporal_features(), selected_node)
                    
                    col1_node_plot, _, col2_node_plot = st.columns([3, 1, 3])

                    col1_node_plot.pyplot(fig_selected_node_incoming)
                    col2_node_plot.pyplot(fig_selected_node_outgoing)
