# Author: Mirela Cazzolato
# Date: May 2023
# Goal: Window to visualize and manage t-graph features
# =======================================================================

import config
import util
import callmine_focus.callmine_focus as callmine_focus

import pandas as pd
import streamlit as st
from streamlit_plotly_events import plotly_events


def fextraction_tab():
    with st.expander("Select file with raw data (mandatory)", expanded=True):
        form_fextraction_input_file = st.form(key='form_fextraction_load_file')
        file_source_extraction = form_fextraction_input_file.text_input("Enter input file path:")
        use_example_file = form_fextraction_input_file.checkbox("Use example file",
                                    False,
                                    help="Use in-built example file to demo the app")

        if use_example_file:
                file_source_extraction = "data/sample_raw_data.csv"

        form_fextraction_input_file.form_submit_button("Load File Headers")

        if file_source_extraction is not None and file_source_extraction != '':
            st.success("Selected file: " + str(file_source_extraction))
            print('Selected file for feature extraction:', file_source_extraction)
            
            util.read_file_header(file_source_extraction)

    with st.expander(label="Select t-graph parameters and extract features", expanded=True):
        cc0, cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 1, 1])
        with cc0:
            st.write("Select columns to load:")
        with cc1:
            hasSource = st.checkbox(label="SOURCE", value=True, disabled=True, help="The SOURCE column is mandatory")
        with cc2:
            hasDestination = st.checkbox(label="DESTINATION", value=True, disabled=True, help="The DESTINATION column is mandatory")
        with cc3:
            hasMeasure = st.checkbox(label="MEASURE", value=False, help="The MEASURE column for weighted graphs is optional")
        with cc4:
            hasTimestamp = st.checkbox(label="TIMESTAMP", value=False, help="The TIMESTAMP column for time-evolving graphs is optional")

        form_fextraction_input_columns = st.form(key='form_fextraction_load_columns')
        
        if hasSource:
            source = form_fextraction_input_columns.selectbox(
                            "Select SOURCE column for t-graph",
                            options=config.columns_fextraction,
                            index=0)

        if hasDestination:
            destination = form_fextraction_input_columns.selectbox(
                            "Select DESTINATION column for t-graph",
                            options=config.columns_fextraction,
                            index=1)
        
        if hasMeasure:
            measure = form_fextraction_input_columns.selectbox(
                            "Select MEASURE column for t-graph",
                            options=config.columns_fextraction,
                            index=2)
        else:
            measure=None

        if hasTimestamp:
            timestamp = form_fextraction_input_columns.selectbox(
                            "Select TIMESTAMP column for t-graph",
                            options=config.columns_fextraction,
                            index=3)
        else:
            timestamp = None

        run_tgraph_fextraction = form_fextraction_input_columns.form_submit_button("Run t-graph")

        if run_tgraph_fextraction:
            with st.spinner('Running t-graph and extracting features...'):
                df_head_features = util.run_t_graph(file_source_extraction, source, destination, measure, timestamp)

            st.success("Finished extracting features. Check file *\'data/allFeatures_nodeVectors.csv\'*")
            st.write("### Extracted features (top rows)", df_head_features)


def input_tab():
    with st.expander("t-graph Features (mandatory)", expanded=True):
        file_features = st.file_uploader(label="Select a file with features",
                                type=['txt', 'csv'])

        use_example_file_features = st.checkbox("Use example file with features",
                                False,
                                help="Use in-built example file with features to demo the app")

        config.flag_use_negative_list = st.checkbox(label="Remove nodes from the negative-list",
                                                    value=True,
                                                    help="Select this option to ignore nodes that are in the negative-list.")

        if use_example_file_features and not file_features:
            file_features = config.feature_file_path            
            
        if file_features:
            with st.spinner('Reading features...'):
                util.read_file_features(file_features)
            st.success("Features loaded.")

    with st.expander("Raw data (mandatory for deep dives and attention routing)", expanded=True):
        file_graph = st.file_uploader(label="Select a file with raw data",
                                    type=['txt', 'csv'])
                                    
        use_example_graph = st.checkbox("Use example file with raw data",
                                False,
                                help="Use in-built example file of raw data to demo the app")

        if use_example_graph and not file_graph:
            file_graph = config.raw_data_file_path

        if file_graph:
            cc0, cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 1, 1])
            
            with cc0:
                st.write("Select columns to load:")
            with cc1:
                hasSource = st.checkbox(label="SOURCE", value=True, disabled=True, key="sourceLoad", help="The SOURCE column is mandatory")
            with cc2:
                hasDestination = st.checkbox(label="DESTINATION", value=True, disabled=True, key="destinationLoad", help="The DESTINATION column is mandatory")
            with cc3:
                hasMeasure = st.checkbox(label="MEASURE", value=False, key="measureLoad", help="The MEASURE column for weighted graphs is optional")
            with cc4:
                hasTimestamp = st.checkbox(label="TIMESTAMP", value=False, key="timestampLoad", help="The TIMESTAMP column for time-evolving graphs is optional")

            form_load_raw_data_columns = st.form(key='form_load_files_columns_raw_data')

            mcol1_graph, mcol2_graph, mcol3_graph, mcol4_graph = form_load_raw_data_columns.columns(4)

            util.read_file_graph(file_graph)
            st.success("Raw data loaded.")

            with mcol1_graph:
                if hasSource:
                    config.SOURCE = form_load_raw_data_columns.selectbox(
                                    "Select SOURCE column",
                                    options=config.df_raw_data.columns,
                                    index=0)
            with mcol2_graph:
                if hasDestination:
                    config.DESTINATION = form_load_raw_data_columns.selectbox(
                                    "Select DESTINATION column",
                                    options=config.df_raw_data.columns,
                                    index=1)
            with mcol3_graph:
                if hasMeasure:
                    config.MEASURE = form_load_raw_data_columns.selectbox(
                                    "Select MEASURE column",
                                    options=config.df_raw_data.columns,
                                    index=2)
                else:
                    config.MEASURE = None

            with mcol4_graph:
                if hasTimestamp:
                    config.TIMESTAMP = form_load_raw_data_columns.selectbox(
                                    "Select TIMESTAMP column",
                                    options=config.df_raw_data.columns,
                                    index=3)
                else:
                    config.TIMESTAMP = None


            construct_graph_load_columns = form_load_raw_data_columns.form_submit_button("Construct Graph",
                                                help="Load raw data and construct graph for deep dive.")

            if construct_graph_load_columns:
                with st.spinner('Constructing graph...'):
                    util.construct_graph()
                st.success("Graph constructed.")

    with st.expander("Labels (optional)", expanded=True):
        if config.flag_features_loaded:
            file_labels = st.file_uploader(label="Select a file with labels",
                                type=['txt', 'csv'])

            use_example_file_labels = st.checkbox(
                "Use example file", False, help="Use in-built example file with labels to demo the app"
            )

            if use_example_file_labels and not file_labels:
                file_labels = "data/sample_raw_data.csv"
                
            if file_labels:
                util.read_file_label(file_labels)
                populate_selectbox_labels()

            if st.button('Set label settings'):
                # If this is not the first time setting the label columns, reload the file
                if config.flag_labels_sorted:
                    util.read_file_label(file_labels)
            
            if ((config.flag_features_loaded) and (config.flag_labels_loaded) and (not config.flag_labels_sorted)):
                util.sort_labels()

                st.success("Labels loaded.")
    

def hexbin_tab():
    with st.expander("Select features to visualize", expanded=True):
        form_hexbin_feature_selection = st.form(key='form_feature_selection')

        populate_selectbox_features(form_hexbin_feature_selection)
        # If labels are loaded and sorted
        if config.flag_labels_sorted:
            config.check_use_labels_hexbin = form_hexbin_feature_selection.checkbox("Use labels", False,
                            help="Select this option to visualize features with the loaded labels.")

        plot_hexbin = form_hexbin_feature_selection.form_submit_button("Plot HexBin")

    if plot_hexbin:
        with st.expander(label="HexBin visualization", expanded=True):
            with st.spinner('Generating HexBin...'):
                _, col_hexbin, _ = st.columns([1, 5, 1])
                
                with col_hexbin:
                    # Add histogram scatter plot
                    fig_hexbin = util.plot_hexbin()
                    st.pyplot(fig_hexbin)

        if config.check_use_labels_hexbin:
            with st.expander("Labeled HexBin", expanded=True):
                # Add histogram scatter plot
                fig_hexbin_label = util.plot_hexbin_labeled()
                st.pyplot(fig_hexbin_label)


def populate_selectbox_features(form):
    """
    Populate select box with available features
    """
    
    mcol1_features, mcol2_features = form.columns(2)

    with mcol1_features:
        config.feature1_hexbin = form.selectbox("Select first feature",
                                options=config.df_features.columns[1:],
                                index=0)
    with mcol2_features:
        config.feature2_hexbin = form.selectbox("Select second feature",
                                options=config.df_features.columns[1:],
                                index=1)
    

def populate_selectbox_labels():
    """
    Populate select box with available labels
    """

    mcol1_label_options, mcol2_label_options, mcol3_label_options = st.columns([1, 1, 1])

    with mcol1_label_options:
        config.label_column_node_id = st.selectbox("Select column with NODE ID",
                                            options=config.df_labels.columns,
                                            index=0)

    with mcol2_label_options:
        config.label_column_name = st.selectbox("Select column with LABEL",
                                         options=config.df_labels.columns,
                                         index=1)

    with mcol3_label_options:
        config.label_true_value = st.selectbox("Select TRUE LABEL VALUE",
                                        options=config.df_labels[config.label_column_name].unique(),
                                        index=0,
                                        help="Only rows with selected value will be considered fraud (True).  "
                                                + "The remaining rows will be considered as not fraud (False)")
    
        st.write("True value for fraud:", config.label_true_value)


def deep_dive_tab():
    """
    Options for lasso selection and deep dive
    """
    

    with st.expander("Select features to visualize", expanded=True):
        checkbox_custom_columns = st.checkbox("Custom features",
                help="Select desired features or unselect this option to show pre-set feature combinations.",
                value=True)

        form_deep_dive_features = st.form(key='select_deep_dive_features')

        if checkbox_custom_columns:
            config.columns_matrix_lasso = form_deep_dive_features.multiselect("Choose features to visualize",
                                                config.df_features.columns[1:].values)
        
        else:
            preset_columns = form_deep_dive_features.radio("Pre-set feature combinations",
                                        (config.preset_features))
            
            if (preset_columns == config.preset_features[0]):
                config.columns_matrix_lasso = config.preset_feature_columns[0]
            elif (preset_columns == config.preset_features[1]):
                config.columns_matrix_lasso = config.preset_feature_columns[1]
            elif (preset_columns == config.preset_features[2]):
                config.columns_matrix_lasso = config.preset_feature_columns[2]
            elif (preset_columns == config.preset_features[3]):
                config.columns_matrix_lasso = config.preset_feature_columns[3]
            elif (preset_columns == config.preset_features[4]):
                config.columns_matrix_lasso = config.preset_feature_columns[4]
        
       
        submit_deep_dive_features = form_deep_dive_features.form_submit_button("Visualize Features")

    with st.expander(label="Interactive visualization — select nodes of interest", expanded=True):
        if (len(config.columns_matrix_lasso) > 1):
            if (len(config.columns_matrix_lasso) > 2):
                fig_lasso = util.plot_lasso_scatter_matrix()
            elif (len(config.columns_matrix_lasso) == 2):
                fig_lasso = util.plot_lasso()

            # Add interactive scatter plot  
            config.selected_points_lasso = plotly_events(fig_lasso, select_event=True,
                                            override_height=config.plotly_height,
                                            override_width=config.plotly_width,)

            if len(config.selected_points_lasso) > 0:
                config.df_selected = pd.DataFrame(config.selected_points_lasso)
                st.write("Selected nodes:", len(config.selected_points_lasso))
                st.markdown("**Features of select nodes:**")
                st.dataframe(config.df_features.loc[config.df_selected["pointNumber"].values])

            if (not(config.flag_graph_constructed)):
                st.error("Please, construct graph in the \"Input Data\" tab to continue.")
            

    with st.expander(label="Deep dive on selected nodes", expanded=True):
        if len(config.selected_points_lasso) > 0 and config.flag_graph_constructed:
            with st.spinner('Deep-diving into the selected nodes...'):
                egonet_suspicious_nodes, index_suspicious_nodes = util.get_egonet(
                                            suspicious_nodes=config.df_features.loc[config.df_selected["pointNumber"].values][config.NODE_ID]
                                        )
                st.write("EgoNet size:", len(egonet_suspicious_nodes.nodes))
            
                # Show features of nodes in the EgoNet
                st.write("**Nodes in the EgoNet:**")
                df_result = pd.DataFrame(data=egonet_suspicious_nodes.nodes, columns=[config.NODE_ID]).set_index(config.NODE_ID).join(config.df_features.set_index(config.NODE_ID)).reset_index()
                df_result.columns=config.df_features.columns
                st.write(df_result)
                
                if (config.flag_update_matrices):
                    if st.button("Update EgoNet visualizations"):
                        config.flag_update_matrices = False

        if len(config.selected_points_lasso) > 0 and config.flag_graph_constructed:
            st.markdown("**Adjacency matrix of the generated EgoNet:**")
            # This flag is true for the first sleection and
            # for every time the user asks to update the EgoNet
            if (config.flag_update_matrices == False):
                with st.spinner('Generating adjacency matrices...'):
                    if len(egonet_suspicious_nodes.nodes) < config.max_nodes_association_matrix:
                        config.fig_adj_matrix, config.fig_cross_associations = util.plot_adj_matrix(G=egonet_suspicious_nodes)
                        
                    else:
                        config.fig_adj_matrix = util.plot_adj_matrix(G=egonet_suspicious_nodes, compute_associations=False)
                
                with st.spinner('Generating interactive EgoNet...'):
                    # Generate the visualization of the graph/egonet
                    util.plot_interactive_egonet(egonet_suspicious_nodes, index_suspicious_nodes)
            
            _, col1, _, col2, _ = st.columns([1, 3, 1, 3, 1])
            col1.pyplot(config.fig_adj_matrix)

            if len(egonet_suspicious_nodes.nodes) < config.max_nodes_association_matrix:
                col2.pyplot(config.fig_cross_associations)

            # Plot the visualization of the graph/egonet
            if (config.fig_plotly_graph_spring_layout):
                st.plotly_chart(config.fig_plotly_graph_spring_layout,
                                use_container_width=True)
            
            st.markdown("**Parallel coordinates for multiple features:**")

            form_parallel_coordinates = st.form(key='form_parallel_coordinates')
            config.columns_parallel_coordinates = form_parallel_coordinates.multiselect("Choose coordinates to visualize",
                                            config.df_features.columns[1:].values)
            
            submit_coordinates = form_parallel_coordinates.form_submit_button("Plot coordinates")

            if (len(config.columns_parallel_coordinates) > 1):
                with st.spinner('Generating parallel coordinates...'):
                    util.plot_parallel_coordinates(df_features=df_result, columns=config.columns_parallel_coordinates)
                    st.plotly_chart(config.fig_parallel_coordinates,
                                    use_container_width=True)

    
    with st.expander(label="Deep dive on the EgoNet: totals over time", expanded=True):
        if len(config.selected_points_lasso) > 0 and config.flag_egonet_constructed and (config.TIMESTAMP!=None) and (config.MEASURE!=None):
            
            form_temporal = st.form(key='form_temporal_deep_dive')
            col1_dates, col2_dates = form_temporal.columns([3, 1])
            
            with col1_dates:
                selected_date = form_temporal.selectbox(
                        "Select initial date",
                        options=config.UNIQUE_DATES,
                        index=0)

            with col2_dates:
                    n_days = form_temporal.number_input(
                        "#Days to visualize the data",
                        min_value=1,
                        max_value=5,
                        value=1 ,
                        step=1,
                        format="%d",
                        help="""Maximum number of days to visualize the data,
                                from the selected date. Use with caution, it may
                                take a while to compute.""",
                        on_change=None)
            
            submit_temporal = form_temporal.form_submit_button("Visualize data over time")
            
            if submit_temporal:
                with st.spinner('Generating temporal plots...'):
                    util.get_temporal_curves(df_result, selected_date=selected_date, n_days=n_days)
                    
            if (config.fig_cum_sum_in_degree):
                col1_temporal, _, col2_temporal = st.columns([3, 1, 3])                    
                col1_temporal.pyplot(config.fig_cum_sum_in_degree)
                col2_temporal.pyplot(config.fig_cum_sum_out_degree)

                col3_temporal, _, col4_temporal = st.columns([3, 1, 3])
                col3_temporal.pyplot(config.fig_cum_sum_in_total_count)
                col4_temporal.pyplot(config.fig_cum_sum_out_total_count)
            

            form_selected = st.form(key='form_selected_node_deep_dive')
            # Deep dive on selected node
            selected_node = form_selected.selectbox("Select node from EgoNet",
                                                options=df_result,
                                                index=0)
            submit_selected_node = form_selected.form_submit_button("Deep dive on selected node")
            if submit_selected_node:
                with st.spinner('Deep-diving into the selected node...'):
                    st.write("Deep dive on node "+ str(selected_node))
                    util.get_temporal_node_curves(selected_node)
            
            if (config.fig_selected_node_incoming):
                col1_node_plot, _, col2_node_plot = st.columns([3, 1, 3])

                col1_node_plot.pyplot(config.fig_selected_node_incoming)
                col2_node_plot.pyplot(config.fig_selected_node_outgoing)

                print("Done generating temporal features.")


def negative_list_tab():
    """
    Manage list of negative (blocked) nodes
    that must be ignored by the application
    """

    st.write("Ongoing work.")
    

    with st.expander(label="List of nodes that must be ignored by the application", expanded=True):
        use_example_file = st.checkbox(str("Use negative-list file \""+config.negative_list_file_path+"\""),
                                    True,
                                    disabled=True,
                                    help="Use in-built example file for negative-list nodes in \""+config.negative_list_file_path+"\"")

    with st.expander(label="Listed negative-nodes", expanded=True):
        if (use_example_file and use_example_file != ""): # Always true
            file_source = config.negative_list_file_path

        if (file_source is not None and file_source != ""):
            st.write("Selected file:", file_source)
            
            util.initialize_negative_list()            
            
            col1, col2 = st.columns([1, 1])

            with col1:
                new_negative_list_item = st.text_input("Enter Node ID to add to the negative-list:")

                if st.button("Add node to list"):
                    util.add_node_to_negativelist(node_id=new_negative_list_item)
                    st.success("Node added to the negative-list file.")
            
            with col2:
                node_to_remove = st.text_input("Enter Node ID to delete from negative-list:")
                if st.button("Delete node from list"):
                    util.remove_node_from_negativelist(node_id=node_to_remove)
                    st.success("Node deleted from the negative-list file.")

        st.write(config.df_negative_list.astype(str))
    


def update_sidebar():
    """
    Add options to the sidebar
    """

    with st.sidebar:
    
        config.ego_radius = st.number_input(
            "EgoNet radius parameter",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            format="%d",
            help="""Radius of the egonet. It
                  may take a while to run. Use with caution."""
        )

        config.max_nodes_association_matrix = st.number_input(
            "Max #nodes for matrix cross-association",
            min_value=1,
            # max_value=5,
            value=500,
            step=20,
            format="%d",
            help="""Maximum number of nodes allowed to generate the
                  matrix coss-association plot. With high #nodes, it
                  may take a while to run. Use with caution."""
        )


def launch_w_dashboard():
    """
    Launch window with all functionalities:
    static and interactive visualization,
    and deep dive and attention-routing
    """
    
    update_sidebar()

    tab_feature_extraction, tab_input_data, tab_HexBin, tab_DeepDive, tab_negative_list = st.tabs([
                                                                 "Feature Extraction",
                                                                 "Input Data (mandatory)",
                                                                 "HexBin",
                                                                 "Deep Dive",
                                                                 "Negative-List"])


    with tab_feature_extraction:
        fextraction_tab()
    with tab_input_data:
        input_tab()
    with tab_HexBin:
        if config.flag_features_loaded:
            hexbin_tab()
    with tab_DeepDive:
        if config.flag_features_loaded:
            deep_dive_tab()
    with tab_negative_list:
        if config.flag_features_loaded:
            negative_list_tab()

