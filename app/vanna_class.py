import os
from typing import Tuple, Union
from flask import current_app
from vanna.chromadb import ChromaDB_VectorStore
from vanna.openai import OpenAI_Chat
from vanna.ollama import Ollama
import pandas as pd
import plotly
import tiktoken
from app.db_utils import open_db_connection


# class MyVanna(OpenAI_Chat, ChromaDB_VectorStore):
#     document_store = {}
#     def __init__(self,
#         config = {
#             "api_key": os.getenv("OPENAI_API_KEY"),
#             "model": "gpt-4o",
#             "path": "../vanna-chroma"
#         }):
#         ChromaDB_VectorStore.__init__(self, config=config)
#         OpenAI_Chat.__init__(self, config=config)

class MyVanna(Ollama, ChromaDB_VectorStore):

    # def count_tokens_for_model(self, text: str) -> int:
    #     MODEL_NAME = "gpt-4o"
    #     encoding = tiktoken.encoding_for_model(MODEL_NAME)
    #     return len(encoding.encode(text))
    
    document_store = {}
    def __init__(self, 
        config = {
            "ollama_host": os.getenv("OLLAMA_HOST", "http://192.168.1.116:11434"),
            "model": "deepseek-r1:14b",
            "path": "../vanna-chroma"
        }):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

    # def chat_completion(self, messages, **kwargs):
    #     """
    #     Wrapper kept only for legacy code that still expects
    #     `self.chat_completion(messages)["content"]`.

    #     Under the hood it just forwards to `submit_prompt()` and wraps the
    #     returned string in the tiny dict old callers expect.
    #     """
    #     content = self.submit_prompt(messages, **kwargs)

    #     # Count tokens in the LLM response
    #     tokens = self.count_tokens_for_model(content)
    #     current_app.logger.info(f"RESPONSE TOKENS FROM LLM: {tokens}")
    #     return {"content": content}

    def chat_completion(self, messages, **kwargs):
        """
        Wrapper kept only for legacy code that still expects
        `self.chat_completion(messages)["content"]`.

        Under the hood it just forwards to `submit_prompt()` and wraps the
        returned string in the tiny dict old callers expect.
        """
        return {"content": self.submit_prompt(messages, **kwargs)}

    def split_document_into_chunks(self, document: str, max_chunk_size: int = 500) -> List[str]:
        """
        Splits a document into chunks based on max_chunk_size (in characters).
        You can enhance this to split by section, paragraph, or token count.
        """
        chunks = []
        words = document.split()
        current_chunk = []

        for word in words:
            if sum(len(w) + 1 for w in current_chunk) + len(word) + 1 > max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            current_chunk.append(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    
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
             """1. You MUST reference ONLY tables / columns that exist in the supplied DDL / metadata.
                    • For each requested attribute, find an exact or clear-synonym column name.
                    • If no match exists, you MUST NOT invent a name.
                        - Instead, either:
                        a) produce an `intermediate_sql` query to discover the correct column,  OR
                        b) return `NULL AS "<Friendly-Name>"  -- UNMAPPED`.
                    • Violating this rule is considered an error; regenerate until compliant.\n"""
            "2. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n"
            "3. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
            "4. Please use the most relevant table(s). \n"
            "5. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
            "6. If you do not see any primary and foreign key relationships(joins) in the DDL, Take the same column names as joins and gemerate the sql queries. \n"
            f"7. Ensure that the output SQL is {self.dialect}-compliant and executable, and free of syntax errors. \n"
            "8. For any string comparison in a WHERE clause, wrap both the column and the literal in LOWER() (or UPPER()) to make the match case-insensitive. \n"
            "9. Use ONLY the tables, columns, and values that exist in the provided schema context. Do not use columns or values that do not exist in the context. \n"\
            "10. Make sure all rows are distinct and there are no duplicates. \n"
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
        current_app.logger.info(f"from VANNA : {message_log}")

        # Count tokens in the message log
        # text = message_log
        # tokens = self.count_tokens_for_model(text)
        # current_app.logger.info(f"Tokens in the string: {tokens}")

        return message_log
    
    # ------------------------------- pre-processing metadadata -------------------------------

    def train_with_fallback_doc(self, ddl_list: list[str] | None = None) -> str:
       
        ddl_list = ddl_list or []

        samples = self._extract_sample_data_from_db()
        if samples is None:
            return None
        context_md = self.build_metadata(ddl_list, samples)

        table_docs: list[str] = []

        current_app.logger.info(f"Making LLM call to generate metadata from DDL and sample Tables: {len(samples)} ",)

        for table, meta in samples.items():
            context_md = self.build_metadata(ddl_list, {table: meta})
            sys_prompt =  (
                            """ You are a senior data  analyst. Given the SQL DDL and a small sample of table data, analyze the structure and content to extract standardized fro each column,
                              output metadata in Markdown starting with: Table: <table_name>
                                Then generate a 5-column table with headers:
                                | Column Name | Data Type | Column Alias Name | Description |
                                 Populate the table as follows
                                 Column Name: The column name as defined in the table.
                                 Data Type: As defined in the DDL.
                                 Column Alias Name: Inferred standardized name for use in relationship analysis across tables. Use actual data patterns and naming similarities to assign meaningful, consistent aliases (e.g., user_id, created_at, product_code).
                                 Description: A concise explanation of the column's meaning, based on the column name and actual sample data.
                                 Use — if any information is missing or cannot be inferred from the available data.."""
                        )

            messages = [
                self.system_message(sys_prompt),
                self.user_message(context_md)
            ]
            doc_text = self.chat_completion(messages)["content"]
            table_docs.append(doc_text)

        generated_doc = "\n\n".join(table_docs)
        current_app.logger.info(f"Generated metadata document: {generated_doc}")

        db_id = self.train(documentation=generated_doc)
        current_app.logger.info(f"Generated metadata document is added to vanna chroma: {db_id}")
        self.add_document(db_id=db_id, doc_id=db_id)

        return db_id

    
    def _extract_sample_data_from_db(self, limit: int = 50) -> dict:
        try:
            conn = open_db_connection()
            cur  = conn.cursor() 
            if not conn:
                current_app.logger.error("Failed to connect to the database")
                raise Exception("Failed to connect to the database")
            cur = conn.cursor()
            cur.execute(""" SELECT table_name FROM   information_schema.tables WHERE  table_schema = 'public'; """)
            samples = {}
            table_names = cur.fetchall()

            for (table,) in table_names:
                cur.execute(f"SELECT * FROM {table} LIMIT {limit}")
                rows = cur.fetchall()
                cols = [c.name for c in cur.description]
                samples[table] = {"columns": cols, "rows": rows}
            cur.close()
            conn.close()
        except Exception as e:
            current_app.logger.error("Failed to connect to the database")
            return None
        return samples

    def build_metadata(self, ddl_list: list[str], samples: dict) -> str:
        parts = ["## All DDL"]
        parts += [f"```sql\n{ddl.strip()}\n```" for ddl in ddl_list]

        for table, meta in samples.items():
            parts.append(f"\n## {table}")
            parts.append("Columns: " + ", ".join(meta["columns"]))
            if meta["rows"]:
                parts.append("Sample rows:")
                for r in meta["rows"]:
                    parts.append("  - " + ", ".join(map(str, r)))
            else:
                parts.append("*no rows*")
        return "\n".join(parts)

 # ------------------------------- END pre-processing metadadata -------------------------------


    def ask(
        self,
        question: Union[str, None] = None,
        print_results: bool = True,
        auto_train: bool = True,
        visualize: bool = True,  # if False, will not generate plotly code
        allow_llm_to_see_data: bool = False,
    ) -> Union[
        Tuple[
            Union[str, None],
            Union[pd.DataFrame, None],
            Union[plotly.graph_objs.Figure, None],
        ],
        None,
    ]:
        """
        **Example:**
        ```python
        vn.ask("What are the top 10 customers by sales?")
        ```

        Ask Vanna.AI a question and get the SQL query that answers it.

        Args:
            question (str): The question to ask.
            print_results (bool): Whether to print the results of the SQL query.
            auto_train (bool): Whether to automatically train Vanna.AI on the question and SQL query.
            visualize (bool): Whether to generate plotly code and display the plotly figure.

        Returns:
            Tuple[str, pd.DataFrame, plotly.graph_objs.Figure]: The SQL query, the results of the SQL query, and the plotly figure.
        """

        if question is None:
            question = input("Enter a question: ")

        try:
            sql = self.generate_sql(question=question, allow_llm_to_see_data=allow_llm_to_see_data)
        except Exception as e:
            print(e)
            return None, None, None

        if print_results:
            try:
                Code = __import__("IPython.display", fromList=["Code"]).Code
                display(Code(sql))
            except Exception as e:
                print(sql)

        if self.run_sql_is_set is False:
            print(
                "If you want to run the SQL query, connect to a database first."
            )

            if print_results:
                return None
            else:
                return sql, None, None

        try:
            df = self.run_sql(sql)

            if print_results:
                try:
                    display = __import__(
                        "IPython.display", fromList=["display"]
                    ).display
                    display(df)
                except Exception as e:
                    print(df)

            if len(df) > 0 and auto_train:
                self.add_question_sql(question=question, sql=sql)
            # Only generate plotly code if visualize is True
            if visualize:
                try:
                    plotly_code = self.generate_plotly_code(
                        question=question,
                        sql=sql,
                        df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                    )
                    fig = self.get_plotly_figure(plotly_code=plotly_code, df=df)
                    if print_results:
                        try:
                            display = __import__(
                                "IPython.display", fromlist=["display"]
                            ).display
                            Image = __import__(
                                "IPython.display", fromlist=["Image"]
                            ).Image
                            img_bytes = fig.to_image(format="png", scale=2)
                            display(Image(img_bytes))
                        except Exception as e:
                            fig.show()
                except Exception as e:
                    # Print stack trace
                    print("Couldn't run plotly code: ", e)
                    if print_results:
                        return None
                    else:
                        return sql, df, None
            else:
                return sql, df, None

        except Exception as e:
            print("Couldn't run sql: ", e)
            if print_results:
                return None
            else:
                return sql, e, None
        return sql, df, fig




