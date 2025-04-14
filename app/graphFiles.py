
from flask import current_app
import networkx as nx
import igraph as ig
from collections import defaultdict
import leidenalg
import os
from random import sample
from app.llm_model import LLMModel
from flask import current_app
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from urllib.parse import urlparse
from pyvis.network import Network

class GraphFiles():
    def __init__(self):
        self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = LLMModel.from_together()


        
    def build_similarity_graph(self, chunks, threshold=0.75):
        current_app.logger.info(f"CHUNKS are created")
        embeddings = self.st_model.encode(chunks)
        G_nx = nx.Graph()

        for idx, chunk in enumerate(chunks):
            G_nx.add_node(idx, text=chunk,)

        sim_matrix = cosine_similarity(embeddings)

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                if sim_matrix[i][j] > threshold:
                    G_nx.add_edge(i, j, weight=sim_matrix[i][j])

        G_ig = ig.Graph.TupleList(G_nx.edges(), directed=False)
        partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
        current_app.logger.info(f"Chunk partition is created...")
        for idx, community in enumerate(partition.membership):
            G_nx.nodes[idx]['cluster'] = community
        self.G_nx = G_nx
        self.embeddings = embeddings
        self.chunks = chunks
        return G_nx
    def get_main_node_label_from_url(self, url):
        netloc = urlparse(url).netloc
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc.split('.')[0].capitalize()

    def get_main_node_label_from_files(self, filenames):
        if len(filenames) == 1:
            return os.path.splitext(os.path.basename(filenames[0]))[0]
        elif len(filenames) <= 3:
            return ", ".join([os.path.splitext(os.path.basename(f))[0] for f in filenames])
        else:
            return f"{len(filenames)} Uploaded Files"
    def render_graph_html(self, G_nx, min_cluster_size,MAX_LABEL_NODES):
        clusters = defaultdict(list)
        for node, data in G_nx.nodes(data=True):
            cluster = data.get("cluster", -1)
            clusters[cluster].append(node)

        # Change as needed: only show clusters with at least 5 nodes
        filtered_clusters = {c: nodes for c, nodes in clusters.items() if len(nodes) >= min_cluster_size}
        current_app.logger.info(f"Filtered clusters: {list(filtered_clusters.keys())}")

        G_radial = nx.Graph()
        if hasattr(self, "source_url"):
            main_label = self.get_main_node_label_from_url(self.source_url)
        elif hasattr(self, "file_names"):
            main_label = self.get_main_node_label_from_files(self.file_names)
        else:
            main_label = "Graph Source"
        # Add a central node (e.g., representing the main query or topic)
        main_node = "Main"
        G_radial.add_node(main_node, label='MAIN NODE', title=main_label, color="orange", size=30)
        # For each filtered cluster, add a cluster node and then add all sentence nodes
        for cluster, nodes in filtered_clusters.items():
            cluster_node = f"Cluster_{cluster}"
            sample_nodes = sample(nodes, min(len(nodes), MAX_LABEL_NODES))
            current_app.logger.info(f"Maximum modes in each cluster: {len(sample_nodes)}")
            def get_clean_text(node_id):
                raw = G_nx.nodes[node_id].get("text", "")
                return raw["text"] if isinstance(raw, dict) else raw

            cluster_text = "\n".join([get_clean_text(n) for n in sample_nodes])
            label = self.generate_cluster_label(cluster_text)
            # Add the cluster node with styling
            G_radial.add_node(cluster_node, label=f"{label} ({len(nodes)})", title=label, color="red", size=15, shape="box")
            # Connect the main node to the cluster node
            weight = float(data.get('weight', 1))
            G_radial.add_edge(main_node, cluster_node)
            
            # Add each sentence node within the cluster
            for n in sample_nodes:
                sentence =get_clean_text(n)
                label = sentence[:100] + "..." if len(sentence) > 100 else sentence
                G_radial.add_node(n, label=" ", title=label, color="lightblue")
                # weight = float(data.get('weight', 1))
                G_radial.add_edge(cluster_node, n)
        net = Network(height="800px", width="100%", directed=False)
        net.from_nx(G_radial)
        net.repulsion(node_distance=150, central_gravity=0.3)
        html_content = net.generate_html()
        return html_content


    def generate_cluster_label(self, cluster_text: str) -> str:
        prompt = (
            "Generate a clear, concise, descriptive label (max. 2 or 3 words) "
            "for the following cluster of sentences. Just give me the label without additional details."
        )
        full_prompt = f"{cluster_text}\n\n{prompt}"
        response = self.llm.model.invoke(full_prompt).content.strip()
        current_app.logger.info(f"Generated cluster label: {response}")
        return response
    def query_graph(self, query: str):
        query_vec = self.st_model.encode([query])[0]
        similarities = cosine_similarity([query_vec], self.embeddings)[0]

        top_k = max(5, int(len(self.chunks) * 0.01))
        top_k_idx = np.argsort(similarities)[-top_k:][::-1]

        top_clusters = {
            self.G_nx.nodes[i]['cluster']
            for i in top_k_idx
            if 'cluster' in self.G_nx.nodes[i]
        }

        cluster_nodes = [
            n for n in self.G_nx.nodes
            if 'cluster' in self.G_nx.nodes[n] and self.G_nx.nodes[n]['cluster'] in top_clusters
        ]

        if not top_clusters or not cluster_nodes:
            current_app.logger.info("⚠️ No clusters found. Using only top-k similar chunks.")
            context_node_ids = list(top_k_idx)
        else:
            subgraph = self.G_nx.subgraph(cluster_nodes)
            centrality_scores = nx.pagerank(subgraph)
            ranked_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
            central_nodes = [node for node, _ in ranked_nodes  [:max(5, int(0.2 * len(ranked_nodes)))]]
            context_node_ids = list(set(central_nodes + list(top_k_idx)))

        context_data = [self.G_nx.nodes[n].get("text", "") for n in context_node_ids]

        # Make sure every chunk is a string, just in case
        context_chunks = [str(chunk) for chunk in context_data if chunk]

        # Join them safely for LLM context
        context = "\n---\n".join(context_chunks)

        current_app.logger.info(f'CONTEXT DATA SENT')
        return self.generate_query_answer(query, context)
    
    def generate_query_answer(self, query, context):
        full_prompt = f"""You are a technical assistant. Based on the context below, provide a **detailed** and **step-by-step** answer to the user's question.
            Context:
            {context}
            User Question:
            {query}
            Instructions:
            - Provide **detailed steps** for the process, including any parameters, options, or settings the user needs to be aware of. use line breaks wherever necessary.
            - Ensure that each step is explained with as much information as possible, including specific UI elements, selections, and actions. use line breaks wherever necessary.
            - Focus on providing comprehensive, actionable, and explicit instructions.
            - if the query: {query} ; involves a process or procedure, break down each phase, highlighting key actions, choices, or configurations.
            - if the query: {query} ; does not have any relevant information in the context, just say "I Don't have Relevant Information".
            
            Answer:"""
        llm_response = self.llm.model.invoke(full_prompt).content.strip()
        return llm_response
    def query_graph_link_response(self, query: str) -> dict:
        """Query using top chunks to return LLM-generated response + related link"""
        query_vec = self.st_model.encode([query])[0]
        similarities = cosine_similarity([query_vec], self.embeddings)[0]

        top_k = max(5, int(len(self.chunks) * 0.01))
        top_k_idx = np.argsort(similarities)[-top_k:][::-1]

        # Retrieve chunk info
        retrieved_units = []
        for i in top_k_idx:
            retrieved_units.append({
                "unit_id": i,
                "text": self.G_nx.nodes[i]['text'],
                "url": self.G_nx.nodes[i].get("url", "No Link Available"),
                "section_title": self.G_nx.nodes[i].get("section_title", "Untitled")
            })
        product_links = [unit["url"] for unit in retrieved_units]
        # Remove placeholders like "No Link Available"
        valid_links = [url for url in product_links if url != "No Link Available"]

        retrieved_texts = "\n---\n".join([
            f"[Section: {unit['section_title']}]\n{unit['text']}" for unit in retrieved_units
        ])
        current_app.logger.info(f"chunks length {len(retrieved_texts)}")

        # Format prompt
        prompt = self._format_prompt(query, valid_links, retrieved_texts)

        # Get LLM response
        llm_response = self.llm.model.invoke(prompt).content.strip()

        return llm_response
    def _format_prompt(self, query, product_links, retrieved_texts):
        max_input_length = 6000
        trimmed_texts = retrieved_texts[:max_input_length]

        return (
            f"You are a helpful assistant. Use the information below to answer the user's question.\n\n"
            f"Context:\n{trimmed_texts}\n\n"
            f"Available Links:\n" + "\n".join(product_links) + "\n\n"
            f"User Question:\n{query}\n\n"
            f"Instructions:\n"
            "- Read the context carefully.\n"
            "- If the context includes a real link relevant to your answer, include **only one link** on a new line at the end.\n"
            "- Do **NOT** include fake links like 'No Link Available' or placeholder text.\n"
            "- If no link is available or needed, **just skip it** — only return the answer.\n"
            "- If there's no relevant info in the context, return exactly: 'I don't have enough information.'\n"
            "- Stick strictly to the context provided. Don't make things up.\n"
            "- Format:\n"
            "  [Answer]\n"
            "  [Link — only if real and useful]\n"
        )

