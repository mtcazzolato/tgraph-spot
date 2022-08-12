# tgraphSpot
Anomaly detection for time-evolving graphs


## Requirements

Check file `requirements.txt`

To create and use a virtual environment, type:

    python -m venv wcw_venv
    source wcw_venv/bin/activate
    pip install -r requirements.txt
 
 
## Running the app

Run the app with the following command on your Terminal:

    streamlit run app/tgraphspot.py --server.maxUploadSize 5000

 - Parameter `[--server.maxUploadSize 5000]` is optional, and it is used to increase the size limit of input files.
