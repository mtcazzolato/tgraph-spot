# TgraphSpot
Fast and Effective Anomaly Detection for Time-Evolving Graphs

**Authors:** Mirela T. Cazzolato<sup>1,2</sup>, Saranya Vijayakumar<sup>1</sup>, Xinyi Zheng<sup>1</sup>, Namyong Park<sup>1</sup>, Meng-Chieh Lee<sup>1</sup>, Pedro Fidalgo<sup>3,4</sup>, Bruno Lages<sup>3</sup>, Agma J. M. Traina<sup>2</sup>, Christos Faloutsos<sup>1</sup>.  

**Affiliations:** <sup>1</sup> Carnegie Mellon University (CMU), <sup>2</sup> University of São Paulo (USP), <sup>3</sup> Mobileum, <sup>4</sup> ISCTE-IUL  

*Work submitted for review.*  

## Requirements

Check file `requirements.txt`

To create and use a virtual environment, type:

    python -m venv tgraph_venv
    source tgraph_venv/bin/activate
    pip install -r requirements.txt
 
 
## Running the app

Run the app with the following command on your Terminal:  

    make
or  

    streamlit run app/tgraphspot.py --server.maxUploadSize 5000

 - Parameter `[--server.maxUploadSize 5000]` is optional, and it is used to increase the size limit of input files.

## Data Sample

We provide a toy sample dataset on folder *data/*. Check file *sample_raw_data.csv*  

## Acknowledgement

**Matrix cross-associations**
The code for generating matrix cross-associations is originally from [this Github repository](https://github.com/clifflyon/fully-automatic-cross-associations).  
The work was proposed in [this](https://www.cs.cmu.edu/~christos/PUBLICATIONS/kdd04-cross-assoc.pdf) paper:  

> Deepayan Chakrabarti, S. Papadimitriou, D. Modha, C. Faloutsos.
> **Fully automatic cross-associations**. Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data
> mining. 2004. DOI:10.1145/1014052.1014064.


----------------------


# TgraphSpot: Video tutorial

Step-by-step tutorial on how to use TgraphSpot to generate features and visualize the results


## Step 1: Feature Extraction  

Inform the path of a file containing columns corresponding to source, destination, and measure (e.g., call duration). We provide a sample file in the repository as an example. Click and check the option "Use example file " to use it in the application." After loading the file, click on "Run t-graph" and wait until de task is done. The application saves the file with generated features in the folder "data/."



https://user-images.githubusercontent.com/8514761/186825700-87bc77c7-1995-4c20-b2fc-2ada4ea14683.mov



## Step 2: HexBin scatter plot

Load the file with the extracted features (from Step 1), and select pairs of features to visualize. The chart is automatically updated. Labels can also be loaded and visualized separately.



https://user-images.githubusercontent.com/8514761/186825713-97bf963e-fc2f-4f4b-8d65-8dd6e933a8a0.mov



## Step 3: Lasso selection and parallel coordinates

Load the extracted features and the file with phone calls. Then select a pair of features to visualize. The application allows the user to make a lasso selection of points of interest. The selected points are listed below the chart. From the selected nodes, the application generates the corresponding EgoNet and plots the adjacency matrix and the cross-associations found. Generating the cross-associations can take some time. The user can control the maximum size of the EgoNet to generate the corresponding visualization (see the parameter in the left panel). Finally, at the bottom of the page, the application shows a plot with parallel coordinates, allowing the user to visualize many features at once.



https://user-images.githubusercontent.com/8514761/186825735-13d99ae2-a932-410d-9df6-1660f8190f68.mov



## Step 4: Interactive scatter matrix

The interactive scatter matrix allows the user to visualize many scatter plots simultaneously, combining many features of interest. There are pre-set feature combinations as well, defined by experts to assist in finding abnormal behavior on logs of phone calls. As mentioned in Step 3, the user can also select desired points, and generate the EgoNet and the matrix visualizations.



https://user-images.githubusercontent.com/8514761/186825759-19c1fd02-b517-47ab-b09a-9acdc590d336.mov



## Step 5: Deep dive

In the deep dive module, the user can visualize the incoming and outgoing behavior of the nodes from the generated EgoNet over time. In the selected period, the user can further select a node and visualize the total duration of incoming and outgoing calls per hour.



https://user-images.githubusercontent.com/8514761/186825795-891ed0e4-1c84-4ba9-a1cf-697c06fbd695.mov



## Step 6: Manage negative-list

The negative-list can be used to remove numbers (or nodes) that usually receive or make many calls but should be ignored during the analysis. Examples of such cases are emergency and service numbers.


https://user-images.githubusercontent.com/8514761/186825829-5bfa05c5-01a4-4482-adc1-1a3202c32070.mov





