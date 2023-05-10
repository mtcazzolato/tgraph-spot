import config

import sys
import numpy as np
import pandas as pd
import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as patches
import plotly.graph_objs as go
import plotly.express as px

import datashader as ds
import datashader.transfer_functions as tf
from datashader.colors import inferno
from matplotlib.cm import jet

# t-graph modules
# import tgraph.static_graph as SG
# import tgraph.temporal_graph as TG

sys.path.insert(0, 'app/tgraph')
from tgraph import TGraph

# cross-associations modules
from collections import Counter
from cross_associations.matrix import Matrix
from cross_associations.plots import plot_shaded, plot_spy
import cross_associations.cluster_search as cs

from temporal_features.egonet_deep_dive import get_curves, get_node_curves

plt.rcParams.update({'font.size': 12})
figsize=[8, 6]


def read_file_header(file):
    """
    Read columns from input file
    
    Parameters
    ----------
    file: str
        path of the input file with raw data
    """

    config.columns_fextraction = pd.read_csv(file, nrows=1).columns.tolist()


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

    print('======================', file)
    
    # Get static features
    my_tgraph = TGraph(filename=file,
                        source=source,
                        destination=destination,
                        measure=measure,
                        timestamp=timestamp)

    my_tgraph.data_network.df_nodes.to_csv("data/allFeatures_nodeVectors.csv", index=False)

    print(my_tgraph.data_network.df_nodes.head())

    return my_tgraph.data_network.df_nodes.head()

    # # Get static features
    # sg = SG.StaticGraph(filename=file,
    #                     source=source,
    #                     destination=destination,
    #                     measure=measure)
    # sg.my_print()
    
    # # Get temporal features
    # tg = TG.TemporalGraph(filename=file,
    #                       source=source,
    #                       destination=destination,
    #                       measure=measure,
    #                       timestamp=timestamp)
    # tg.my_print()
    
    # # Join static and temporal features
    # df_all_features = sg.df_nodes.set_index(config.NODE_ID).join(tg.df_nodes.set_index(config.NODE_ID)).reset_index()
    # df_all_features.fillna(0)
    
    # # Save output features
    # print("\n\n ----")
    # df_all_features.to_csv("data/allFeatures_nodeVectors.csv", index=False)
    # print("Check the file \"data/allFeatures_nodeVectors.csv\"")

    # return df_all_features.head()


def read_file_features(file):
    """
    Read columns from input file
    
    Parameters
    ----------
    file: str
        path of the input file with features
    """

    config.df_features = pd.read_csv(file)
    config.flag_features_loaded=True


def read_file_label(file):
    """
    Read columns from input file
    
    Parameters
    ----------
    file: str
        path of the input file with labels
    """

    config.df_labels = pd.read_csv(file)
    config.flag_labels_sorted = False
    config.flag_labels_loaded=True


def read_file_graph(file):
    """
    Read source, destination, duration and timestamp
    from input file

    Parameters
    ----------
    file: str
        path of the input file with raw data
    """
    # File with raw data (source, destination,
    # measure, timestamp) to generate the graph

    config.df_raw_data = pd.read_csv(file)
    config.flag_raw_data_loaded=True
    

def construct_graph():
    """
    Construct graph for the deep dive with selected attributes
    """


    config.G = nx.from_pandas_edgelist(config.df_raw_data,
                                       source=config.SOURCE,
                                       target=config.DESTINATION,
                                       edge_attr=config.MEASURE,
                                       create_using=nx.DiGraph())
    
    config.flag_graph_constructed=True
    
    if config.TIMESTAMP:
        config.df_raw_data[config.TIMESTAMP] = config.df_raw_data[config.TIMESTAMP].astype('datetime64[s]')
        config.df_raw_data.sort_values(by=config.TIMESTAMP, inplace=True)
        config.df_raw_data.reset_index(drop=True, inplace=True)

    if config.MEASURE:
        config.df_raw_data[config.MEASURE] = pd.to_numeric(config.df_raw_data[config.MEASURE], errors='coerce')
        config.df_raw_data.dropna(subset=[config.MEASURE])
        config.df_raw_data = config.df_raw_data[config.df_raw_data[config.MEASURE] > 0]

    config.df_raw_data = config.df_raw_data[config.df_raw_data[config.SOURCE] != config.df_raw_data[config.DESTINATION]]

    if config.TIMESTAMP:
        config.df_raw_data = config.df_raw_data.groupby([config.TIMESTAMP, config.SOURCE, config.DESTINATION]).sum().add_suffix('').reset_index()
        config.UNIQUE_DATES = config.df_raw_data[config.TIMESTAMP].dt.date.unique()
    else:
        config.df_raw_data = config.df_raw_data.groupby([config.SOURCE, config.DESTINATION]).sum().add_suffix('').reset_index()
        config.UNIQUE_DATES = None

def sort_labels():
    """
    Sort input labels by NODE_ID, according to the order
    given by the nodes
    
    Parameters
    ----------
    label_column_node_id: str
        column from the lables' file with the NODE_ID
    label_column_name: str
        column from the lables' file with label values
    label_true_value: str
        string with the value corresponding to TRUE for anomaly
    """

    config.NODE_ID_LABEL = config.label_column_node_id
    config.LABEL = config.label_column_name
    config.LABEL_TRUE_VALUE = config.label_true_value
    
    # Get unique node values for True and False
    true_nodes = config.df_labels[config.df_labels[config.LABEL].astype(str) == str(config.LABEL_TRUE_VALUE)][config.NODE_ID_LABEL].unique()
    df_label_unique = pd.DataFrame(data=true_nodes, columns=[config.NODE_ID_LABEL])
    
    config.LABEL_TRUE_VALUE = 'True'
    df_label_unique[config.LABEL] = len(df_label_unique) * [config.LABEL_TRUE_VALUE]

    # Join feature df with unique labeled dataframe
    config.df_labels = config.df_features.set_index(config.NODE_ID).join(df_label_unique.set_index(config.NODE_ID_LABEL), how='left', lsuffix='_left')[[config.LABEL]]
    config.df_labels = config.df_labels.reset_index()
    
    # Replace NaN with False
    config.df_labels[config.LABEL].fillna('False', inplace=True)

    config.NODE_ID_LABEL = config.NODE_ID
    config.flag_labels_sorted = True


def get_label_indexes(label_value):
    """
    Return the indexes of rows containing the informed label value

    Parameters
    ----------
    label_value: str
        label value of the rows to be retrieved
    """

    idx = np.where(config.df_labels[config.LABEL].astype(str) == str(label_value))[0].tolist()
    
    return idx


def plot_hexbin():
    """
    Plot hexbin with selected features
    """

    fig_hexbin, ax = plt.subplots(figsize=figsize)
    img = ax.hexbin(np.log10(config.df_features[config.feature1_hexbin]+1),
                    np.log10(config.df_features[config.feature2_hexbin]+1),
                    cmap=config.cmap, mincnt=1, bins='log')
    
    cb = plt.colorbar(img, ax=ax)
    cb.set_label("log10(N)")
    ax.set_xlabel(config.feature1_hexbin.replace('_', ' ') + ' — log10(x+1)')
    ax.set_ylabel(config.feature2_hexbin.replace('_', ' ') + ' — log10(x+1)')
    ax.grid(True)

    return fig_hexbin


def plot_hexbin_labeled():
    """
    Plot hexbin with selected features and available labels
    """
    
    # Create subplots with shared x and y axis
    fig_hexbin_labeled, ax = plt.subplots(nrows=1,
                           ncols=config.num_labels,
                           sharex=True,
                           sharey=True,
                           figsize=[figsize[0]*2, figsize[1]])

    img = None

    for i, l in enumerate(config.labels):
        idx_label = get_label_indexes(label_value=l)
        if (len(idx_label) > 0):
            img = ax[i].hexbin(x = np.log10(config.df_features[config.feature1_hexbin].loc[idx_label]+1),
                               y = np.log10(config.df_features[config.feature2_hexbin].loc[idx_label]+1),
                               cmap=config.cmap, mincnt=1, bins='log')
        
        ax[i].set_xlabel(config.feature1_hexbin.replace('_', ' ') + ' — log10(x+1)')
        ax[i].set_ylabel(config.feature2_hexbin.replace('_', ' ') + ' — log10(x+1)')
        ax[i].set_title('Label={}'.format(l))
        cb = plt.colorbar(img, ax=ax[i])
        cb.set_label("log10(N)")
        ax[i].set_title("Label={}".format(l))
        ax[i].grid(True)
    
    plt.tight_layout()
    return fig_hexbin_labeled
   

def plot_lasso():
    """
    Plot interactive scatter plot with the selected pair of features
    """
    
    fig_lasso = go.FigureWidget([go.Scatter(x = np.log10(config.df_features[config.columns_matrix_lasso[0]]+1),
                                    y = np.log10(config.df_features[config.columns_matrix_lasso[1]]+1),
                                    mode = 'markers')])
    fig_lasso.layout.xaxis.title = config.columns_matrix_lasso[0].replace('_', ' ') + ' — log10(x+1)'
    fig_lasso.layout.yaxis.title = config.columns_matrix_lasso[1].replace('_', ' ') + ' — log10(x+1)'
    # f.update_layout(width=plotly_width,height=plotly_height)

    scatter = fig_lasso.data[0]
    scatter.marker.opacity = 0.5

    fig_lasso.update_traces(
                unselected_marker=dict(opacity=0.05, size=7),
                selected_marker=dict(size=12, opacity=0.9))

    return fig_lasso


def plot_lasso_scatter_matrix():
    """
    Plot interactive scatter plot with the selected columns as features

    """

    truecolor = '#f95a10'
    falsecolor = 'blue'
    linecolor = 'white'

    dimensions=[]
    for c in config.columns_matrix_lasso:
        # Construct dict with column names and labels
        dimensions.append(dict(label=c.replace('_', ' ') + ' — log10(x+1)',
                          values=np.log10(config.df_features[c]+1)))

    colors = pd.Series(data=[falsecolor] * len(config.df_features))
    
    fig_matrix_lasso = go.Figure(data=go.Splom(
            dimensions = dimensions,
            customdata = config.df_features[config.NODE_ID],
            hovertemplate="<br>".join([
                        "%{xaxis.title.text}: %{x}",
                        "%{yaxis.title.text}: %{y}",
                        "hash: %{customdata}",
            ]),
            showlegend=False, #Show legend entries later on!
            showupperhalf=False, # remove plots in the diagonal
            marker=dict(color=list(colors),
                        showscale=False,
                        line_color=linecolor,
                        line_width=0.8,
                        size=8,
                        opacity=0.5
            ),
    ))
    
    fig_matrix_lasso.update_traces(
                unselected_marker=dict(opacity=0.1, size=5),
                selected_marker=dict(size=10, opacity=0.9),
                selector=dict(type='splom'),
                diagonal_visible=False)
    
    fig_matrix_lasso.update_layout(
        # title='Scatter matrix with graph information',
        dragmode='select',
        hovermode='closest',
    ) 
    
    return fig_matrix_lasso



def get_egonet(suspicious_nodes):
    """
    Compose a graph with the egonets of a given set of suspicious nodes.
    Return the subgraph of the composed egonets and the index of
    suspicious nodes inside the subgraph

    Parameters
    ----------
    suspicious_nodes: list
        list of indexes of suspicious nodes
    """
        
    final_egonet = nx.empty_graph(create_using=nx.DiGraph())

    for ego_node in suspicious_nodes:
        # create ego network
        hub_ego = nx.ego_graph(config.G, ego_node,
                               radius=config.egonet_radius,
                               distance='weight',
                               undirected=True)
        final_egonet = nx.compose(final_egonet, hub_ego)

    idx_suspicious_nodes = []
    for node in suspicious_nodes:
        idx_suspicious_nodes.append(list(np.where(pd.DataFrame(data=final_egonet.nodes()) == node)[0])[0])
    
    config.flag_egonet_constructed = True
    return final_egonet, idx_suspicious_nodes


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

    fig_adj, ax = plt.subplots()
    plt.spy(nx.adjacency_matrix(G), markersize=markersize)
    ax.set_aspect('equal', adjustable='box')
    plt.ylabel('source')
    plt.xlabel('destination')
    # plt.tight_layout()

    if compute_associations: # Compute matrix associations
        fig_cross_associations = plot_cross_associations(nx.adjacency_matrix(G))
    else:
        return fig_adj  # Skip matrix associations and return only the original matrix

    config.flag_update_matrices = True
    return fig_adj, fig_cross_associations


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


def plot_interactive_egonet(G, suspicious_nodes=[]):
    """
    # TODO
    """
    
    edge_x = []
    edge_y = []
    pos = nx.layout.spring_layout(G)
    
    
    for edge in G.edges():
        x0, y0 = pos.get(edge[0])
        x1, y1 = pos.get(edge[1])
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos.get(node)
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(x=node_x, y=node_y,
        mode='markers', hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlGnBu', reversescale=True, color=[], size=15,
                    colorbar=dict(
                        thickness=15,
                        title='Node Connections',
                        xanchor='left',
                        titleside='right'
                    ),
            line_width=2))

    node_adjacencies = []
    node_text = []

    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1])) # number of adjacencies to use as color
        node_text.append('Node ID: ' + str(list(G.nodes())[node]) + ', #connections: ' + str(len(adjacencies[1])))
    
    for s in suspicious_nodes:
        node_adjacencies[s] = 'red'

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    config.fig_plotly_graph_spring_layout = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        #title='Deep dive on suspicious nodes',
                        titlefont_size=16, showlegend=False, hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            showarrow=True,
                            xref="paper",
                            yref="paper",
                            x=0.005, y=-0.002 ) ],
                    # width=1200,
                    height=600,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
        

def get_temporal_curves(df_result, selected_date, n_days):
    """
    # TODO
    """

    (config.df_temporal_features,
     config.fig_cum_sum_in_degree,
     config.fig_cum_sum_out_degree,
     config.fig_cum_sum_in_total_count,
     config.fig_cum_sum_out_total_count) = get_curves(df_result,
                df_raw_data=config.df_raw_data,
                measure=config.MEASURE,
                timestamp=config.TIMESTAMP,
                source=config.SOURCE,
                destination=config.DESTINATION,
                selected_date=selected_date,
                n_days=n_days)

def get_temporal_node_curves(selected_node):
    """
    # TODO
    """   

    (config.fig_selected_node_incoming,
     config.fig_selected_node_outgoing) = get_node_curves(config.df_temporal_features,
                                                          selected_node)


def plot_parallel_coordinates(df_features, columns):
    """
    Plot parallel ploting of features and the given columns


    Parameters
    ----------
    df_features: DataFrame
        features to plot
    columns: list
        columns to use as coordinates
    """
    
    config.fig_parallel_coordinates = px.parallel_coordinates(df_features,
                                  dimensions=columns,
                                  color_continuous_scale=px.colors.diverging.Tealrose,
                                  color_continuous_midpoint=2)
    config.fig_parallel_coordinates.update_layout(#width=900,
                      height=550)
    config.fig_parallel_coordinates.update_layout(
        font_size=22
    )


def get_sample(budget=1_000):
    sample_idx = np.random.randint(low=0, high=len(config.df_features)-1, size=budget, dtype=int)
    
    return sample_idx


def gen_ds_image(x_range, y_range, plot_width, plot_height, c1, c2):

    if x_range is None or y_range is None or plot_width is None or plot_height is None:
        return None
    
    cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=plot_height, plot_width=plot_width)
    agg_scatter = cvs.points(np.log10(config.df_features[[c1, c2]]+1), 
                         c1, c2
                             # , ds.count_cat('core')
                            )
    img = tf.shade(agg_scatter, cmap=jet, how='log')
    img = tf.dynspread(img, threshold=0.95, max_px=1, shape='circle')
    
    return img.to_pil()


def plot_sample_with_complete_background(column1, column2, outlier_ids=[], include_sample=False):
    # TODO
    # Plot sample and static background for reference

    if include_sample:
        sample_idx = get_sample(budget=1_000)

    x_range=[0, np.log10(config.df_features[column1].max()+1)]
    y_range=[0, np.log10(config.df_features[column2].max()+1)]
    # plot_height=1200
    plot_width=1000

    initial_img = gen_ds_image(x_range, y_range,
                                plot_width=plot_width, plot_height=config.plotly_height,
                                c1=column1, c2=column2)

    config.fig_sample_image_background = go.FigureWidget(data=[{'x': x_range, 
                           'y': y_range, 
                           'mode': 'markers',
                           'marker': {'opacity': 0}}], # invisible trace to init axes and to support autoresize
                    layout={'width': plot_width, 'height': config.plotly_height})

    config.fig_sample_image_background.layout.images = [go.layout.Image(
        source = initial_img,  # plotly now performs auto conversion of PIL image to png data URI
        xref = "x",
        yref = "y",
        x = x_range[0],
        y = y_range[1],
        sizex = x_range[1] - x_range[0],
        sizey = y_range[1] - y_range[0],
        sizing = "stretch",
        layer = "below"
        )
    ]

    def update_ds_image(layout, x_range, y_range, plot_width, plot_height):
        img = config.fig_sample_image_background.layout.images[0]
        
        # Update with batch_update so all updates happen simultaneously
        with config.fig_sample_image_background.batch_update():
            img.x = x_range[0]
            img.y = y_range[1]
            img.sizex = x_range[1] - x_range[0]
            img.sizey = y_range[1] - y_range[0]
            img.source = gen_ds_image(x_range, y_range, plot_width, plot_height)

    # Install callback to run exactly once if one or more of the following properties changes
    #  - xaxis range
    #  - yaxis range
    #  - figure width
    #  - figure height
    config.fig_sample_image_background.layout.on_change(update_ds_image, 'xaxis.range', 'yaxis.range', 'width', 'height')
    config.fig_sample_image_background.layout.dragmode = 'zoom'


    if include_sample:
        config.fig_sample_image_background.add_trace(
                go.Scatter(
                    x=np.log10(config.df_features[column1].iloc[sample_idx]+1),
                    y=np.log10(config.df_features[column2].iloc[sample_idx]+1),
                    mode='markers',
                    showlegend=False,
                    marker=dict(
                        # symbol='x',
                        symbol='circle',
                        opacity=0.3,
                        color='white',
                        size=8,
                        line=dict(width=1),
                    )
                ),
        )
    
    # Include outliers
    config.fig_sample_image_background.add_trace(
            go.Scatter(
                x=np.log10(config.df_features[column1].iloc[outlier_ids]+1),
                y=np.log10(config.df_features[column2].iloc[outlier_ids]+1),
                mode='markers',
                showlegend=False,
                marker=dict(
                    # symbol='x',
                    symbol='circle',
                    opacity=0.8,
                    color='red',
                    size=5,
                    line=dict(width=1),
                )
            ),
    )
    

    config.fig_sample_image_background.layout.xaxis.title = column1.replace('_', ' ') + ' — log10(x+1)'
    config.fig_sample_image_background.layout.yaxis.title = column2.replace('_', ' ') + ' — log10(x+1)'
        
    # config.fig_sample_image_background.update_layout(
    #             autosize=True,
    #             dragmode='select',
    #             hovermode='closest',
    # )

    config.fig_sample_image_background.update_traces(
                unselected_marker=dict(opacity=0.1, size=5),
                selected_marker=dict(size=15, opacity=0.7, color='black'),
                selector=dict(type='splom'),
                diagonal_visible=False
    )

    # return config.fig_sample_image_background