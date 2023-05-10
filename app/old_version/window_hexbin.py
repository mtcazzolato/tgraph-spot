import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

NODE_ID="node_ID"
LABEL=""
NODE_ID_LABEL=""
LABEL_TRUE_VALUE=""

num_labels=2
labels=['False', 'True']
flag_labels_sorted=False

nrows=20_000

def read_file(file):
    """
    Read columns from input file
    
    Parameters
    ----------
    file: str
        path of the input file with features
    """

    global df, nrows
    df = pd.read_csv(file)


def read_file_label(file):
    """
    Read columns from input file
    
    Parameters
    ----------
    file: str
        path of the input file with labels
    """

    global df_label, nrows, flag_labels_sorted
    df_label = pd.read_csv(file)
    flag_labels_sorted = False
    

def sort_labels(label_column_node_id, label_column_name, label_true_value):
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

    global df, df_label, NODE_ID_LABEL, LABEL, LABEL_TRUE_VALUE, flag_labels_sorted

    NODE_ID_LABEL = label_column_node_id
    LABEL = label_column_name
    LABEL_TRUE_VALUE = label_true_value
    
    # Get unique node values for True and False
    true_nodes = df_label[df_label[LABEL].astype(str) == str(LABEL_TRUE_VALUE)][NODE_ID_LABEL].unique()
    
    df_label_unique = pd.DataFrame(data=true_nodes, columns=[NODE_ID_LABEL])
    
    LABEL_TRUE_VALUE = 'True'
    df_label_unique[LABEL] = len(df_label_unique) * [LABEL_TRUE_VALUE]

    # Join feature df with unique labeled dataframe
    df_label = df.set_index(NODE_ID).join(df_label_unique.set_index(NODE_ID_LABEL), how='left')[[LABEL]]
    df_label = df_label.reset_index()
    
    # Replace NaN with False
    df_label[LABEL].fillna('False', inplace=True)

    NODE_ID_LABEL = NODE_ID

    flag_labels_sorted = True


def get_label_indexes(label_value):
    """
    Return the indexes of rows containing the informed label value

    Parameters
    ----------
    label_value: str
        label value of the rows to be retrieved
    """

    global df_label
    
    idx = np.where(df_label[LABEL].astype(str) == str(label_value))[0].tolist()
    
    return idx


def update_sidebar():
    """
    Add options to the sidebar
    """
    
    global df, nrows

    # TODO

    # with st.sidebar:
    #     nrows = st.slider(
    #         "# points in the hexbin scatter plot",
    #         min_value=1, max_value=int(len(df)), value=min(len(df), 20_000)
    #     )

    # # Add option to sort data
    # add_sort_field()


def add_sort_field():
    """
    Add field to sort values in the sidebar
    """

    global df

    with st.sidebar:
        selectbox_sorting_column = st.sidebar.selectbox(
            "Sort data by (descending order):",
            options=df.columns,
            index=0
        )
        
        if st.button('Sort features'):
            df = df.sort_values(by=selectbox_sorting_column, ascending=False)


def populate_selectbox():
    """
    Populate select box with available features
    """
    
    global df, feature1, feature2
    mcol1_features, mcol2_features = st.columns(2)

    with mcol1_features:
        feature1 = st.selectbox("Select first feature",
                                options=df.columns[1:],
                                index=0)
    with mcol2_features:
        feature2 = st.selectbox("Select second feature",
                                options=df.columns[1:],
                                index=1)

def plot_hexbin():
    """
    Plot hexbin with selected features
    """

    global feature1, feature2, nrows, df, df_label
    
    fig, ax = plt.subplots()
    img = ax.hexbin(np.log10(df[feature1]+1),
                    np.log10(df[feature2]+1),
                    cmap='jet', mincnt=1, bins='log')
    
    cb = plt.colorbar(img, ax=ax)
    cb.set_label("log10(N)")
    ax.set_xlabel(feature1.replace('_', ' ') + ' -- log10(x+1)')
    ax.set_ylabel(feature2.replace('_', ' ') + ' -- log10(x+1)')
    
    return fig
    

def plot_hexbin_labeled():
    """
    Plot hexbin with selected features and available labels
    """

    global df, feature1, feature2, nrows

    # Create subplots with shared x and y axis
    fig, ax = plt.subplots(nrows=1,
                           ncols=num_labels,
                           sharex=True,
                           sharey=True,
                           figsize=[10,4])

    for i, l in enumerate(labels):
    
        idx_label = get_label_indexes(label_value=l)
        print(l)
        print(len(idx_label))
        print(idx_label[:20])

        if (len(idx_label) > 0):
            img = ax[i].hexbin(x = np.log10(df[feature1].loc[idx_label]+1),
                                        y = np.log10(df[feature2].loc[idx_label]+1),
                                        cmap='jet', mincnt=1, bins='log')
        
        ax[i].set_xlabel(feature1.replace('_', ' ') + ' -- log10(x+1)')
        ax[i].set_ylabel(feature2.replace('_', ' ') + ' -- log10(x+1)')
        ax[i].set_title('Label={}'.format(l))
        cb = plt.colorbar(img, ax=ax[i])
        cb.set_label("log10(N)")
        ax[i].set_title("Label={}".format(l))
        ax[i].grid(True)
    
    plt.tight_layout()
    return fig
    

def launch_w_hexbin():
    """
    Launch window to visualize hecbin from features
    """

    global nrows, df, df_label, flag_labels_sorted

    st.write(
        """
        # Hexbin scatter plot
        ### Visualize pairs of features generated by t-graph
        """
    )

    # Show options to select input files with feature and labels
    col1_data_selection, col2_data_selection = st.columns([1, 1])

    with col1_data_selection:
        file = st.file_uploader(label="Select a file with features",
                                type=['txt', 'csv'])

        use_example_file = st.checkbox(
            "Use example file", False, help="Use in-built example file to demo the app"
        )

    with col2_data_selection:
        file_labels = st.file_uploader(label="Select a file with labels",
                                type=['txt', 'csv'])

        use_example_file_labels = st.checkbox(
            "Use example file", False, help="Use in-built example file with labels to demo the app"
        )

    if use_example_file and not file:
        file = "data/allFeatures_nodeVectors.csv"

    if use_example_file_labels and not file_labels:
        file_labels = "data/sample_raw_data.csv"


    if file:
        read_file(file)
        update_sidebar()
        populate_selectbox()


        with st.expander(label="Features (top-20 rows)", expanded=False):
            st.dataframe(df.head(20))
        
        with st.expander(label="HexBin visualization", expanded=True):
            _, col_hexbin, _ = st.columns([1, 5, 1])

            with col_hexbin:
                # Add histogram scatter plot
                fig_hexbin = plot_hexbin()
                st.pyplot(fig_hexbin)


    # Show options for label if the user selects a file
    if file_labels:
        with st.expander(label="Setup label options", expanded=True):
            read_file_label(file_labels)

            mcol1_label_options, mcol2_label_options = st.columns([1, 1])

            with mcol1_label_options:
                label_column_node_id = st.selectbox("Select column with NODE ID",
                                        options=df_label.columns,
                                        index=0)
                label_column_name = st.selectbox("Select column with LABEL",
                        options=df_label.columns,
                        index=1)
        
            with mcol2_label_options:
                label_true_value = st.selectbox("Select TRUE LABEL VALUE",
                                        options=df_label[label_column_name].unique(),
                                        index=0,
                                        help="Only rows with selected value will be considered fraud (True).  "
                                              + "The remaining rows will be considered as not fraud (False)")
            
                st.write("True value for fraud:", label_true_value)
                    
            if st.button('Set label settings'):
                # If this is not the first time setting the label columns, reload the file
                if flag_labels_sorted:
                    read_file_label(file_labels)
                
                sort_labels(label_column_node_id, label_column_name, str(label_true_value))


    # If the labels were read and sorted, show the top rows and corresponding plot
    if file_labels and flag_labels_sorted:
        with st.expander(label="Labels (top-20 rows)", expanded=False):
            st.dataframe(df_label.head(20))

        with st.expander(label="HexBin visualization - labeled data", expanded=True):
            # Add histogram scatter plot
            fig_hexbin_label = plot_hexbin_labeled()
            st.pyplot(fig_hexbin_label)
