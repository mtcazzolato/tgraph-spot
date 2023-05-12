# Expected column names
NODE_ID="node_ID"
LABEL=""
NODE_ID_LABEL=""
LABEL_TRUE_VALUE=""
MEASURE=""
TIMESTAMP=""
SOURCE=""
DESTINATION=""
UNIQUE_DATES=[]

# Global dataframes
df_dataset_fextraction=False
df_features=None
df_negative_list=None
df_selected=None
df_labels=None
df_raw_data=None
df_temporal_features=None

# Useful flags
flag_features_loaded=False
flag_labels_loaded=False
flag_labels_sorted=False
flag_raw_data_loaded=False
flag_graph_constructed=False
flag_egonet_constructed=False
flag_update_matrices=False
flag_use_negative_list=False

# Graph for deep dive
G = None

num_labels=2
labels=['False', 'True']
cmap="rainbow"
cmap_colorshade="jet"
plotly_width="100%"
plotly_height=800

columns_fextraction=[]

# Columns to show in the hexbin plot
feature1_hexbin=""
feature2_hexbin=""
check_use_labels_hexbin=False

# Columns to show in the lasso plots
columns_matrix_lasso=[]
# Columns to show in the parallel coordinates
columns_parallel_coordinates=[]
# Selected points with lasso for deep dive
selected_points_lasso=[]
# Selected points with lasso from the density-biased sampling
selected_biased_sampling=[]
# Selected features (keys) for callmine_focus
columns_callmine_focus=[]

callmine_combinations = []
callmine_plot_dict = []

# Columns to control labels in charts
label_column_node_id=""
label_column_name=""
label_true_value=""

max_nodes_association_matrix=500
egonet_radius=1

fig_adj_matrix=None
fig_cross_associations=None
fig_plotly_graph_spring_layout=None
fig_cum_sum_in_degree=None
fig_cum_sum_out_degree=None
fig_cum_sum_in_total_count=None
fig_cum_sum_out_total_count=None
fig_selected_node_incoming=None
fig_selected_node_outgoing=None
fig_parallel_coordinates=None

# Filenames
feature_file_path="data/allFeatures_nodeVectors.csv"
raw_data_file_path="data/sample_raw_data.csv"
negative_list_file_path="data/negative-list.csv"


preset_features = ("in_degree, out_degree, core",
                   "weighted_in_degree, weighted_out_degree, core",
                   "out_degree, in_degree, core, in_median_iat, out_median_measure",
                   "out_degree, in_degree, in_median_measure, out_median_measure",
                   "out_total_count, in_total_count, weighted_degree, core")


# Preset combinations of features
preset_feature_columns = (["in_degree", "out_degree", "core"],
                          ["weighted_in_degree", "weighted_out_degree", "core"],
                          ["out_degree", "in_degree", "core", "in_median_iat", "out_median_measure"],
                          ["out_degree", "in_degree", "in_median_measure", "out_median_measure"],
                          ["out_total_count", "in_total_count", "weighted_degree", "core"])

