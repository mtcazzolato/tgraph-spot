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

