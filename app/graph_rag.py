import json
import random
import spacy
import networkx as nx
import igraph as ig
import leidenalg
import numpy as np
import faiss
import os
from urllib.parse import urlparse
from app.graph_builder import GraphBuilder
from app.llm_model import LLMModel
from sentence_transformers import SentenceTransformer
from pyvis.network import Network
from typing import Dict, List
from flask import current_app
import logging
logger = logging.getLogger(__name__)
class GraphProcessor:
    def __init__(self, builder: GraphBuilder):
        if not hasattr(builder, 'data_dir'):
            raise ValueError("GraphBuilder must be initialized with data_dir")
        self.builder=builder
        self.nlp = spacy.load("en_core_web_sm")
        self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = LLMModel.from_together()
        self.index = None
        self.builder=builder
        self.text_units = []
        self.communities = {}
        self.community_summaries = {}

    def process_scraped_data(self, visited: Dict):

        visited = self.builder.load_scraped_data()
        if not visited:
            raise ValueError("No scraped data available - run processing first")
        self.text_units = self._create_text_units(visited)

        self.text_units = self._extract_entities(self.text_units)
        
        G = self._build_semantic_graph(self.text_units)
        
        self.communities = self._perform_leiden_clustering(G)
   
        self.community_summaries = self._generate_community_summaries()

        self.index = self._build_faiss_index()
  
        return self._visualize_graph_static(G)

    def query_graph(self, query: str):
        """Complete query processing pipeline"""
        
        query_embedding = self.st_model.encode([query])
        distances, indices = self.index.search(query_embedding, 5)
        retrieved_units = [self.text_units[i] for i in indices[0] if i <len(self.text_units)]
        comm_ids = [self.communities.get(unit["unit_id"], -1) for unit in retrieved_units]
        top_comm = max(set(comm_ids), key=comm_ids.count)
        comm_summary = self.community_summaries.get(top_comm, "")   
        product_links=[unit.get("url","No Link Available") for unit in retrieved_units]
        retrieved_texts = "\n---\n".join([unit["text"] for unit in retrieved_units])
        prompt = self._format_prompt(query, product_links, retrieved_texts)
        
        response = self.llm.model.invoke(prompt,max_tokens=2000).content.strip()
        print("RESPNSE: ", response)
        return response

    def _create_text_units(self, visited):
        text_units = []
        for url, text in visited.items():
            segments = [seg.strip() for seg in text.split("\n\n") if len(seg.strip()) > 20]
            for i, seg in enumerate(segments):
                text_units.append({
                    "url": url,
                    "unit_id": f"{url}__{i}",
                    "text": seg
                })
        return text_units

    def _extract_entities(self, text_units):
        for unit in text_units:
            doc = self.nlp(unit["text"])
            unit["entities"] = list(set(ent.text for ent in doc.ents))
        return text_units

    def _build_semantic_graph(self, text_units):
        G = nx.Graph()
        entity_map = {}
        
        for unit in text_units:
            G.add_node(unit["unit_id"], **unit)
            for entity in unit["entities"]:
                entity_map.setdefault(entity, []).append(unit["unit_id"])
        
        for units in entity_map.values():
            for i in range(len(units)):
                for j in range(i+1, len(units)):
                    G.add_edge(units[i], units[j])
        return G

    def _perform_leiden_clustering(self, G):
        ig_nodes = list(G.nodes())
        node_to_index = {n: i for i, n in enumerate(ig_nodes)}
        edges = [(node_to_index[u], node_to_index[v]) for u, v in G.edges()]
        
        ig_graph = ig.Graph()
        ig_graph.add_vertices(len(ig_nodes))
        ig_graph.add_edges(edges)
        partition = leidenalg.find_partition(ig_graph, leidenalg.RBConfigurationVertexPartition)
        logger.info(f"Partitions:  {partition}")
        return {node: partition.membership[i] for i, node in enumerate(ig_nodes)}

    def _generate_community_summaries(self):
        comm_groups = {}
        for unit in self.text_units:
            comm = self.communities.get(unit["unit_id"], -1)
            comm_groups.setdefault(comm, []).append(unit["unit_id"])
        
        summaries = {}
        for comm, unit_ids in comm_groups.items():
            combined = " ".join(u["text"] for u in self.text_units if u["unit_id"] in unit_ids)
            summaries[comm] = combined[:500] + "..." if len(combined) > 500 else combined
        return summaries

    def _build_faiss_index(self):
        embeddings = self.st_model.encode([u["text"] for u in self.text_units])
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype("float32"))
        return index
    def _format_prompt(self, query, product_links, retrieved_texts):
        max_input_length = 6000
        trimmed_texts = retrieved_texts[:max_input_length]
        # current_app.logger.info(f"**************************** query response: {trimmed_texts}",exc_info=True)
        prompt = (
            f"Context:\n{trimmed_texts}\n\n"
        f"Available Product Links:\n" + "\n".join(product_links) + "\n\n"
        f"Question: {query}\n\n"
        "Provide only one of these responses:\n"
       "- If product information is available: '[clear  and relaveant answer to query]\n[Relevant product link]'\n"
        "- If no relevant information is found: 'I don't have enough information."
        f"Answer the question using ONLY the provided Context above.\n"
        "If relevant product information is available, provide a clear and meaningful answer.\n"
        "Include ONLY ONE product link if available.\n"
        "If no relevant information is found, return exactly: 'I don't have enough information.'\n"
        "DO NOT repeat the answer or the link. DO NOT add extra text, notes, or disclaimers.\n\n"


    )
        return prompt        


    def _visualize_graph_static(self, G, output_file="graph_static.html"):
        """Generate interactive graph visualization with custom JS"""
        num_nodes = len(G.nodes)
        if num_nodes <= 50:
            spring_length = 200
            gravity = -1500
        elif num_nodes <= 150:
            spring_length = 400
            gravity = -2500
        else:
            spring_length = 700
            gravity = -3500
        
        pos = nx.spring_layout(G, k=0.8, seed=42)
        net = Network(
            height="800px", 
            width="100%", 
            bgcolor="#ffffff", 
            font_color="black",
            notebook=False, 
            directed=True
        )
        
        net.set_options(f"""
    {{
        "physics": {{
            "enabled": false,
            "forceAtlas2Based": {{
                "gravitationalConstant": {gravity},
                "centralGravity": 0.005,
                "springLength": {spring_length},
                "springConstant": 0.02,
                "damping": 0.4,
                "avoidOverlap": 1.0
            }},
            "solver": "forceAtlas2Based"
        }},
        "interaction": {{
            "hover": true
        }},
        "layout": {{
            "hierarchical": {{
                "enabled": false,
                "direction": "UD",
                "sortMethod": "hubsize"
            }}
        }}
    }}
    """)

        for node, data in G.nodes(data=True):
            x = pos[node][0] * 1000
            y = pos[node][1] * 1000
            url = data.get("url", "#")
            full_text = data.get("text", "")
            parsed_url = urlparse(url)
            last_part = parsed_url.path.rstrip('/').split('/')[-1] if parsed_url.path else "unknown"
            label_text = full_text[:50] + "..." if len(full_text) > 50 else full_text
            tooltip = f"URL: {url}\nLabel: {label_text}"
            random_color = "#%06x" % random.randint(0, 0xFFFFFF)
            
            net.add_node(
                node, 
                label=last_part, 
                title=tooltip, 
                color=random_color, 
                shape="dot",
                size=20, 
                x=x, 
                y=y, 
                fixed=True,
                borderWidth=3,
                shadow=True, 
                href=url
            )
        for edge in G.edges(data=True):
            source, target, edge_data = edge
            weight = edge_data.get("weight", 1)
            edge_color = "#%06x" % random.randint(0, 0xFFFFFF)
            net.add_edge(source, target, color=edge_color, width=weight * 0.5, 
                        arrowsize=0.3, smooth=True, dashes=random.choice([True, False]))
        output_path = os.path.join(self.builder.data_dir, output_file)
        net.save_graph(output_path)
        self._add_custom_js(output_path)
        return self._read_html(output_path)

    def _add_custom_js(self, file_path: str):
        """Inject custom JavaScript for node click handling"""
        custom_js = """
        <script type="text/javascript">
        setTimeout(function(){
            network.on("click", function(params) {
            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                var nodeData = network.body.data.nodes.get(nodeId);
                if (nodeData.href && nodeData.href !== "#") {
                window.open(nodeData.href, "_blank");
                }
            }
            });
        }, 1000);
        </script>
"""
        with open(file_path, "r+", encoding="utf-8") as f:
            html = f.read()
            html = html.replace("</body>", custom_js + "\n</body>")
            f.seek(0)
            f.write(html)
            f.truncate()

    def _read_html(self, file_path: str) -> str:
        """Read generated HTML content"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()