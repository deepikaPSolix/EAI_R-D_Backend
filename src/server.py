from dotenv import load_dotenv
from flask import Flask, request, jsonify
import os

from cluster_classify import ClusterAndClassify
from attribute_extractor import AttributeExtractor
from doc_file import DocFile
import pandas as pd

app = Flask(__name__)
load_dotenv()

# Ensure the folder for saving uploaded files exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/")
def home():
    return "<p>Welcome to EAI!!!</p>"

# @app.route("/sendquery", methods=['POST'])
# def send_query():
#     return "Query result"

# @app.route("/updateRecord", methods=['POST'])
# def update_record():
#     return "Query result"


@app.route("/uploadandtrain", methods=['POST'])
def cluster_and_classify():
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    saved_files = []

    for file in files:
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Save each file to the upload folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print("Filename: ", file.filename)
        saved_files.append(file_path)

    print(saved_files)

    parsed_files = [DocFile(file_path= path) for path in saved_files] # Extract data from file and put in DocFile Object

    # Cluster Data
    cc = ClusterAndClassify()
    cc_res = cc.cluster_classify(data = parsed_files)
    labels = cc.generate_cluster_labels(cc_res)

    # Extract attributes
    attr_ext = AttributeExtractor()
    attr_res = attr_ext.extract_from_files(parsed_files)
    labels.to_csv('labels.csv', index=False)
    attr_res.to_csv('attr.csv', index=False)

    result = pd.merge(labels, attr_res, on='file_name', how='left') 
    result['attributes'] = result[attr_res.columns.difference(['file_name'])].apply(lambda row: row.to_json(), axis=1)
    result = result[['file_name', 'data', 'cluster_labels','label','sensitivity', 'attributes']]
    result.to_csv('result.csv', index=False)

    return jsonify({"labels": labels.to_json(orient='records'), "attributes": attr_res.to_json()}), 201

@app.route("/classify", methods=['POST'])
def classify():
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    saved_files = []

    for file in files:
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Save each file to the upload folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print("Filename: ", file.filename)
        saved_files.append(file_path)

    parsed_files = [DocFile(file_path= path) for path in saved_files] # Extract data from file and put in DocFile Object
    cc = ClusterAndClassify()
    labels = cc.classify_data(parsed_files)

    return jsonify({"labels": labels.to_json(orient='records')}), 201

# @app.route("/retrain", methods=['POST'])
# def cluster_and_classify():
#     return "Query result"



if __name__ == '__main__':
    app.run(debug=True)
