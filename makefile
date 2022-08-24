# goal: run demo for tgraphspot

INPUT_DIR=data
SAMPLE_FILE=sample_raw_data.csv

run:
	streamlit run app/tgraphspot.py --server.maxUploadSize 5000

prep:
	pip3 install -r requirements.txt
