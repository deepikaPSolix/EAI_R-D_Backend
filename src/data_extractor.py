class DataExtractor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.all_data = []

    def initialize_model(self):
        if not os.getenv("TOGETHER_API_KEY"):
            os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter your Together API key: ")
        model = ChatTogether(temperature=0.1)
        self.model = model

    def extract_data(self):
        self.initialize_model()
        files = self.get_files()
        for file_path in files:
            file_name = os.path.basename(file_path)
            chunks = self.extract_data_chunks(file_path)
            data = self.extract_data_from_chunks(chunks)
            combined_json = self.combine_data(data, file_name)
            self.all_data.append(combined_json)

    def extract_data_chunks(self, file_path):
        elements = partition(filename=file_path)
        chunks = chunk_elements(elements, overlap=50, max_characters=2000)
        return chunks

    def extract_data_from_chunks(self, chunks):
        res = []

        for i in range(0, len(chunks), 10):
            query = '''
            Extract attributes described in the output format from the text below.
            - If you can't find an attribute, just leave it blank. Do not put null.
            - If there are multiple values for the same attribute, select the most relevant one based on the context provided.
            - Make sure to check is the text is medical data.
            - Make sure the output data complies with the output format.
            - Only output the JSON and absolutely nothing else.
            Here is the text:
            '''

            for j in range(i, min(i + 10, len(chunks))):
                query += chunks[j].text + "\n"

            attr_res = self.infer_model(query, ExtractedData)
            if attr_res:
                res.append(attr_res)

        return res

    def combine_data(self, data, file_name):
        combine_data_query = f'''
       You have multiple JSON objects containing extracted data. Your task is to combine these into a single JSON object. Follow these guidelines:

- Add a `file_name` attribute with the value `{file_name}`.
- Include all unique keys from the input objects.
- For attributes appearing multiple times, select the most relevant value based on context.
- Merge arrays into a single string if an attribute appears multiple times.
- Replace empty arrays with an empty string.
- If any boolean attribute is true on one object, set it to true on the combined object.
- Provide only the final JSON object without any additional text or formatting.
- Ensure the output is a JSON object, not an array of JSON objects.

Here are the JSON objects:
        '''

        for item in data:
            # print(item)
            try:
                combine_data_query += item.json() + "\n"
            except:
                print(item)
                # combine_data_query += json.dumps(item) + "\n"


        combine_data_query_res = self.model.invoke(combine_data_query)
        return combine_data_query_res.content

    def infer_model(self, query, data_model):
        output_parser = PydanticOutputParser(pydantic_object=data_model)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | self.model | output_parser
        try:
            res = chain.invoke({"query": query})
            return res
        except Exception as e:
            print(e)
            return None

    def convert_to_csv(self, output_csv_path):
        keys = set().union(*(d.keys() for d in self.all_data))

        with open(output_csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=keys)
            writer.writeheader()
            for entry in self.all_data:
                writer.writerow(entry)

    def get_files(self):
        """Get a list of all PDF files in the specified folder."""
        return [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith(".pdf")]


