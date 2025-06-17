import os
from vanna.chromadb import ChromaDB_VectorStore
from vanna.openai import OpenAI_Chat

class MyVanna(OpenAI_Chat, ChromaDB_VectorStore):
    document_store = {}
    def __init__(self,
        config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o",
            "path": "../vanna-chroma"
        }):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

    def add_document(self, db_id, doc_id):
        MyVanna.document_store[db_id] = doc_id

    def get_document(self, db_id):
        if db_id in MyVanna.document_store:
            return MyVanna.document_store[db_id]
        else:
            return None
        
    def delete_document(self, db_id):
        if db_id in MyVanna.document_store:
            del MyVanna.document_store[db_id]

    def delete_all(self):
        MyVanna.document_store = {}
        
    def list_documents(self):
        return list(MyVanna.document_store.keys())
    
    def list_document_names(self):
        return set(MyVanna.document_store.values())
    
    def get_sql_prompt(
        self,
        initial_prompt : str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        """
        Example:
        ```python
        vn.get_sql_prompt(
            question="What are the top 10 customers by sales?",
            question_sql_list=[{"question": "What are the top 10 customers by sales?", "sql": "SELECT * FROM customers ORDER BY sales DESC LIMIT 10"}],
            ddl_list=["CREATE TABLE customers (id INT, name TEXT, sales DECIMAL)"],
            doc_list=["The customers table contains information about customers and their sales."],
        )

        ```

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """
        
        if initial_prompt is None:
            initial_prompt = f"You are a {self.dialect} expert. " + \
            "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt += (
             "===Response Guidelines \n"
            "1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n"
            "2. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
            "3. If the provided context is insufficient, based on your knowlwdge choose the columns that makes more sense closely related to columns that tou have asked for. \n"
            "4. Please use the most relevant table(s). \n"
            "5. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
            "6. If you do not see any primary and foreign key relationships(joins) in the DDL, Take the same column names as joins and gemerate the sql queries. \n"
            f"7. Ensure that the output SQL is {self.dialect}-compliant and executable, and free of syntax errors. \n"
            "8. For any string comparison in a WHERE clause, wrap both the column and the literal in LOWER() (or UPPER()) to make the match case-insensitive. \n"
        )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log