import os
from dotenv import load_dotenv

from attribute_extractor import AttributeExtractor
from cluster_classify import ClusterAndClassify
from doc_file import DocFile

def main():
    load_dotenv()  # take environment variables from .env.

    input_folder = 'input'
    output_folder = 'output'

    file_paths = []
    for filename in os.listdir(input_folder):
        full_path = os.path.join(input_folder, filename)
        if os.path.isfile(full_path):  # Check if it's a file
            file_paths.append(full_path)

    files = [DocFile(file_path= path) for path in file_paths] # Extract data from file and put in DocFile Object


    # Cluster Data
    cc = ClusterAndClassify()
    cc_res = cc.cluster_classify(data = files)
    labels = cc.generate_cluster_labels(cc_res)

    # Extract attributes
    attr_ext = AttributeExtractor()
    attr_res = attr_ext.extract_from_files(files)