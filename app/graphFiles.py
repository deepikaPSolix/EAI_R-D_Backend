from flask import current_app
import networkx as nx
import igraph as ig
from collections import defaultdict
import leidenalg
import os
import psycopg2
from app.graph_storage_pg import GraphPostgresStorage
from app.storage_redis import GraphRedisStorage
from random import sample
from app.llm_model import LLMModel
from flask import current_app
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import ast
from collections import Counter
from urllib.parse import urlparse
from pyvis.network import Network
from app.crew_cluster_labeler import generate_unique_label

dns_host = os.getenv("DNS_HOST")
dns_dbname = os.getenv("DNS_DBNAME")
dns_user = os.getenv("DNS_USER")
dns_password = os.getenv("DNS_PASSWORD")
dns_port = os.getenv("DNS_PORT")
dns = f"host={dns_host} dbname={dns_dbname} user={dns_user} password={dns_password} port={dns_port}"
class GraphFiles():
    def __init__(self):
        self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = LLMModel.from_together()
        self.all_chunks=None
        self.all_embeddings = None
   
    def _ensure_all_chunks_are_dicts(self):
        """
        Normalize self.all_chunks to a list of dicts with 'text' and 'source'.
        Converts malformed strings, stringified dicts, and fills in missing keys.
        """
        valid_chunks = []
        for idx, chunk in enumerate(self.all_chunks):
            try:
                if isinstance(chunk, dict) and "text" in chunk and "source" in chunk:
                    valid_chunks.append(chunk)
                elif isinstance(chunk, dict):
                    valid_chunks.append({
                        "text": chunk.get("text", ""),
                        "source": chunk.get("source", "No Source Available")
                    })
                elif isinstance(chunk, str):
                    try:
                        parsed = ast.literal_eval(chunk)
                        if isinstance(parsed, dict):
                            valid_chunks.append({
                                "text": parsed.get("text", ""),
                                "source": parsed.get("source", "No Source Available")
                            })
                        else:
                            valid_chunks.append({
                                "text": str(chunk),
                                "source": "No Source Available"
                            })
                    except Exception:
                        valid_chunks.append({
                            "text": str(chunk),
                            "source": "No Source Available"
                        })
                else:
                    valid_chunks.append({
                        "text": str(chunk),
                        "source": "No Source Available"
                    })
            except Exception as e:
                current_app.logger.error(f"⚠️ Failed to normalize chunk at index {idx}: {repr(chunk)} - {e}")
                valid_chunks.append({
                    "text": str(chunk),
                    "source": "No Source Available"
                })

        self.all_chunks = valid_chunks


    def split_source_and_text(self, raw: str):
        """Split 'source||text' and clean both parts, also remove embedded filename lines."""
        raw = raw.replace("\x00", "").strip()
        source = None
        text = raw

        if "||" in raw:
            source, text = raw.split("||", 1)
            source = source.strip()
            text = text.strip()

        if source:
            filename = os.path.splitext(os.path.basename(source))[0].lower()
            cleaned_lines = []
            for line in text.splitlines():
                line_lower = line.strip().lower()
                if filename in line_lower:
                    continue  
                if all(part in line_lower for part in filename.split()):
                    continue  
                cleaned_lines.append(line.strip())
            text = "\n".join(cleaned_lines)

        return source, text.strip()


    def build_similarity_graph(self,
                               chunks,
                               threshold: float = 0.85,
                               max_chunks_for_graph: int = 1500):
        current_app.logger.info(f"CHUNKS received: {len(chunks)}")
        
        G_nx = nx.Graph(); processed_texts = []
        for idx, chunk in enumerate(chunks):
            try:
                source = None
                text = ""

                if isinstance(chunk, str):
                    try:
                        parsed = ast.literal_eval(chunk)
                        if isinstance(parsed, dict):
                            text = parsed.get("text", "").strip()
                            source = parsed.get("url", "").strip()
                        else:
                            source, text = self.split_source_and_text(chunk)
                    except Exception:
                        source, text = self.split_source_and_text(chunk)

                elif isinstance(chunk, dict):
                    text = chunk.get("text", "").strip()
                    source = chunk.get("url", "").strip()

                else:
                    source, text = self.split_source_and_text(str(chunk))

            except Exception as e:
                current_app.logger.warning(f"⚠️ Failed to process chunk {idx}: {e}")
                source, text = self.split_source_and_text(str(chunk))

            G_nx.add_node(idx, text=text.strip(), source=source)
            processed_texts.append(text.strip())

        current_app.logger.info(f"NODES added to full graph: {G_nx.number_of_nodes()}")

        self.all_chunks = [{"text": G_nx.nodes[n].get("text", ""), "source": G_nx.nodes[n].get("source", "No Source Available")} for n in G_nx.nodes()]
        self._ensure_all_chunks_are_dicts() 
        self.all_embeddings = self.st_model.encode([c["text"] for c in self.all_chunks])

        mean_vec = np.mean(self.all_embeddings, axis=0, keepdims=True)
        sims     = cosine_similarity(mean_vec, self.all_embeddings).flatten()
        top_idx  = np.argsort(sims)[-max_chunks_for_graph:]
        emb_trim = self.all_embeddings[top_idx]

        simm_mat = cosine_similarity(emb_trim)
        for i in range(len(emb_trim)):
            for j in range(i+1, len(emb_trim)):
                if simm_mat[i,j] >= threshold:
                    u, v = top_idx[i], top_idx[j]
                    G_nx.add_edge(u, v, weight=float(simm_mat[i,j]))

        current_app.logger.info(f"""
        "🔍 About to run Leiden on full graph: "
        "{G_nx.number_of_nodes()} nodes, "
        "{G_nx.number_of_edges()} edges"
    """)
        G_ig      = ig.Graph.TupleList(G_nx.edges(), directed=False)
        partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)

        # After clustering, log cluster membership counts
        from collections import Counter
        counts = Counter(partition.membership)
        # for cid, cnt in counts.items():
        #     current_app.logger.info( f"✅ Leiden produced {len(counts)} clusters: , .join(# {cid}→{cnt}" )  
        # for idx, com in enumerate(partition.membership):
        #     G_nx.nodes[top_idx[idx]]['cluster'] = int(com)
        
        for idx, com in enumerate(partition.membership):
            G_nx.nodes[top_idx[idx]]['cluster'] = int(com)
        current_app.logger.info("✅ Leiden partition done")

        pg = GraphPostgresStorage(dns=dns)
        self.graph_id = pg.save_graph(G_nx)
        current_app.logger.info(f"Saved graph {self.graph_id} to Postgres")

        redis_store = GraphRedisStorage()
        redis_store.store_graph(G_nx)
        current_app.logger.info("Saved graph to RedisGraph")

        self.G_nx           = G_nx
        self.chunk_node_ids = list(G_nx.nodes())
        self.chunks         = [G_nx.nodes[n]["text"] for n in self.chunk_node_ids]
        self.embeddings     = self.st_model.encode(self.chunks)

        return self.graph_id, G_nx


    def _redisgraph_expand(self, seed_ids, max_depth=2, limit=20):
        if seed_ids is None or len(seed_ids) == 0:
            return []

        store     = GraphRedisStorage()
        graph     = store.graph
        seed_list = ",".join(str(i) for i in seed_ids)
        q = f"""
        MATCH (c:Chunk)-[:SIMILAR*1..{max_depth}]->(x:Chunk)
        WHERE c.idx IN [{seed_list}]
        RETURN DISTINCT x.idx, x.text, x.source
        LIMIT {limit}
        """
        result = graph.query(q)
        return [
            {"idx": int(r[0]), "text": r[1], "source": r[2]}
            for r in result.result_set
        ]

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
    def render_graph_html(self, G_nx, min_cluster_size, MAX_LABEL_NODES, threshold):
        clusters = defaultdict(list)
        existing_labels = set()
        for node, data in G_nx.nodes(data=True):
            cluster = data.get("cluster", -1)
            clusters[cluster].append(node)

        # current_app.logger.info(f"All clusters: {[(c, len(n)) for c, n in clusters.items()]}")
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        # current_app.logger.info(f"📊 Cluster sizes: {[(c, len(n)) for c, n in sorted_clusters]}")

        MAX_CLUSTERS = 8
        filtered_clusters = {
            c: nodes for c, nodes in sorted_clusters
            if len(nodes) >= min_cluster_size and c != -1
        }
        filtered_clusters = dict(list(filtered_clusters.items())[:MAX_CLUSTERS])

        current_app.logger.info(f"Filtered clusters: {list(filtered_clusters.keys())}")

        G_radial = nx.Graph()
        if hasattr(self, "source_url"):
            main_label = self.get_main_node_label_from_url(self.source_url)
        elif hasattr(self, "file_names"):
            main_label = self.get_main_node_label_from_files(self.file_names)
        else:
            main_label = "Graph Source"

        main_node = "Main"
        G_radial.add_node(main_node, label='MAIN NODE', title=main_label, color="orange", size=30)
        all_labels = {}

        def get_clean_text(G_nx, node_id):
                
                raw = G_nx.nodes[node_id].get("text", "")
                return raw.get("text", "").strip() if isinstance(raw, dict) else str(raw).strip()
        def format_tooltip(text: str) -> str:

            lines = text.strip().splitlines()
            formatted = []
            for line in lines:
                if line.strip().endswith(":") or line.strip().istitle():
                    formatted.append(f"<b>{line.strip()}</b>")
                else:
                    formatted.append(line.strip())
            return "<br>".join(formatted)

        for cluster_id, nodes in clusters.items():
            if cluster_id == -1:
                continue  
            cluster_text = "\n".join([get_clean_text(G_nx, node) for node in nodes])
            label = generate_unique_label(cluster_text, cluster_id, existing_labels)
            existing_labels.add(label.lower())

    # Store to DB using GraphFiles self.graph_id
            self.store_cluster_label_to_db(cluster_id, label)
            all_labels[cluster_id] = label

        for cluster_id, nodes in filtered_clusters.items():
            cluster_node_f = f"Cluster_{cluster_id}"
            sample_nodes = sample(nodes, min(len(nodes), MAX_LABEL_NODES))
            label = all_labels.get(cluster_id, f"Cluster {cluster_id}")
            G_radial.add_node(cluster_node_f, label=f"{label} ({len(nodes)})", title=label, color="red", size=15, shape="box")
            G_radial.add_edge(main_node, cluster_node_f)

            for n in sample_nodes:
                sentence = get_clean_text(G_nx,n)
                tooltip = format_tooltip(sentence[:100] + "..." if len(sentence) > 100 else sentence)
                G_radial.add_node(n, label=" ", title=tooltip, color="lightblue")
                G_radial.add_edge(cluster_node_f, n)

        self.add_all_semantic_edges(G_nx, G_radial, threshold)

        G_cleaned = nx.Graph()
        for n, d in G_radial.nodes(data=True):
            safe_n = str(n) if not isinstance(n, (int, str)) else n
            G_cleaned.add_node(safe_n, **d)

        for u, v, d in G_radial.edges(data=True):
            safe_u = str(u) if not isinstance(u, (int, str)) else u
            safe_v = str(v) if not isinstance(v, (int, str)) else v
            G_cleaned.add_edge(safe_u, safe_v, **d)

        net = Network(height="800px", width="100%", directed=False)
        net.from_nx(G_cleaned)
        net.repulsion(node_distance=150, central_gravity=0.3)

        # current_app.logger.info(f"✅ Render complete. Cluster labels: {list(all_labels.values())}")
        return net.generate_html()
    def store_cluster_label_to_db(self, cluster_id: int, label: str):
        try:
            import psycopg2
            conn = psycopg2.connect(dns)
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO cluster_labels (graph_id, cluster_id, label)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (graph_id, cluster_id) DO UPDATE SET label = EXCLUDED.label
                    """, (self.graph_id, cluster_id, label))
            current_app.logger.info(f"✅ Label stored: {label} (graph_id={self.graph_id}, cluster_id={cluster_id})")
        except Exception as e:
            current_app.logger.error(f"❌ DB label storage failed for cluster {cluster_id}: {e}", exc_info=True)

    def add_all_semantic_edges(self, source_graph: nx.Graph, target_graph: nx.Graph, threshold):
        """
        Connect all node pairs in target_graph based on semantic similarity stored in source_graph.
        This connects across clusters as well.
        """
        edge_count = 0
        nodes = list(target_graph.nodes())
        for i, n1 in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                n2 = nodes[j]
                if source_graph.has_edge(n1, n2):
                    weight = source_graph[n1][n2]['weight']
                    if weight >= threshold:
                        target_graph.add_edge(n1, n2, color="gray", width=1)
                        edge_count += 1
        current_app.logger.info(f"🌐 Global semantic edges added: {edge_count}")
    def render_combined_clusters_to_single_graph(self, graph_cluster_map: dict, min_cluster_size=1, MAX_LABEL_NODES=7, threshold=0.84):
            combined_G = nx.Graph()
            node_id_map = {}
            cluster_labels = {}

            conn = psycopg2.connect(dns)
            cur = conn.cursor()

            for graph_id, cluster_ids in graph_cluster_map.items():
                cluster_ids = [int(cid) for cid in cluster_ids if int(cid) >= 0]

                cur.execute("""
                    SELECT gn.node_idx, gn.cluster_id, gn.text, gn.source, cl.label
                    FROM graph_nodes gn
                    LEFT JOIN cluster_labels cl ON cl.graph_id = gn.graph_id AND cl.cluster_id = gn.cluster_id
                    WHERE gn.graph_id = %s AND gn.cluster_id = ANY(%s)
                """, (graph_id, cluster_ids))
                rows = cur.fetchall()

                for node_idx, cluster_id, text, source, label in rows:
                    new_id = f"{graph_id}_{node_idx}"
                    combined_G.add_node(new_id, text=text, cluster=cluster_id, graph_id=graph_id, source=source)
                    node_id_map[(graph_id, node_idx)] = new_id
                    cluster_labels[(graph_id, cluster_id)] = label or f"Cluster {cluster_id}"

                if rows:
                    selected_ids = list(set([r[0] for r in rows]))
                    cur.execute("""
                        SELECT source_idx, target_idx, weight
                        FROM graph_edges
                        WHERE graph_id = %s
                        AND source_idx = ANY(%s)
                        AND target_idx = ANY(%s)
                    """, (graph_id, selected_ids, selected_ids))
                    for source_idx, target_idx, weight in cur.fetchall():
                        sid = node_id_map.get((graph_id, source_idx))
                        tid = node_id_map.get((graph_id, target_idx))
                        if sid and tid:
                            combined_G.add_edge(sid, tid, weight=weight)

            cur.close()
            conn.close()

            self.G_nx = combined_G
            self.graph_id = None
            self.chunk_node_ids = list(combined_G.nodes())
            self.all_chunks = [
                    {
                        "text": combined_G.nodes[n].get("text", ""),
                        "source": combined_G.nodes[n].get("source", "No Source Available")
                    }
                    for n in self.chunk_node_ids
                ]

            self._ensure_all_chunks_are_dicts()

            self.all_embeddings = self.st_model.encode([
                chunk["text"] for chunk in self.all_chunks if chunk["text"]
            ])

            net = Network(height="800px", width="100%", directed=False)
            main_node = "Main"
            net.add_node(main_node, label="MAIN NODE", color="orange", size=30)

            cluster_map = defaultdict(list)
            for node, data in combined_G.nodes(data=True):
                cluster_map[(data['graph_id'], data['cluster'])].append(node)

            def format_tooltip(text: str) -> str:
                lines = str(text).strip().splitlines()
                formatted = []
                for line in lines:
                    if line.strip().endswith(":") or line.strip().istitle():
                        formatted.append(f"<b>{line.strip()}</b>")
                    else:
                        formatted.append(line.strip())
                return "<br>".join(formatted)

            for (graph_id, cluster_id), nodes in cluster_map.items():
                label = cluster_labels.get((graph_id, cluster_id), f"Cluster {cluster_id}")
                cluster_node = f"Cluster_{graph_id}_{cluster_id}"
                net.add_node(cluster_node, label=f"{label} ({len(nodes)})", title=label, color="red", shape="box", size=15)
                net.add_edge(main_node, cluster_node)

                sample_nodes = sample(nodes, min(len(nodes), MAX_LABEL_NODES))
                for n in sample_nodes:
                    if not combined_G.has_node(n):
                        continue
                    text = combined_G.nodes[n].get("text", "")
                    tooltip = format_tooltip(text[:100] + "..." if len(text) > 100 else text)
                    net.add_node(n, label=" ", title=tooltip, color="lightblue")
                    net.add_edge(cluster_node, n)

            edge_count = 0
            for i, n1 in enumerate(list(combined_G.nodes)):
                for j in range(i + 1, len(combined_G.nodes)):
                    n2 = list(combined_G.nodes)[j]
                    if combined_G.has_edge(n1, n2):
                        weight = combined_G[n1][n2].get("weight", 0)
                        if weight >= threshold and n1 in net.get_nodes() and n2 in net.get_nodes():
                            net.add_edge(n1, n2, color="gray", width=1)
                            edge_count += 1

            net.repulsion(node_distance=150, central_gravity=0.3)
            return net.generate_html()

    # def generate_cluster_label(self, cluster_text: str, cluster_id: int) -> str:
    #     conn = psycopg2.connect(dns)
    #     cur = conn.cursor()

    #     try:
    #         cur.execute("""
    #             SELECT label FROM cluster_labels
    #             WHERE graph_id = %s AND cluster_id = %s
    #         """, (self.graph_id, cluster_id))
    #         row = cur.fetchone()
    #         if row and row[0]:
    #             label = row[0].strip()
    #             current_app.logger.info(f"🟢 Using cached label for cluster {cluster_id}: {label}")
    #             return label

    #         cur.execute("""
    #             SELECT label FROM cluster_labels
    #             WHERE graph_id = %s
    #         """, (self.graph_id,))
    #         existing_labels = [r[0].strip() for r in cur.fetchall() if r[0]]
    #         normalized_existing = {label.lower() for label in existing_labels}

    #         if not cluster_text.strip():
    #             label = f"Cluster {cluster_id}"
    #             current_app.logger.warning(f"⚠️ Cluster {cluster_id} is empty. Using default label.")
    #         else:
    #             prompt = f"""
    #                     You are an expert language model tasked with labeling semantic clusters in a knowledge graph.
    #                     🧠 Cluster #{cluster_id} Content:
    #                     {cluster_text.strip()}
    #                     🆕 Existing Label:{existing_labels}
    #                     Each cluster is a group of related topics or concepts. Your goal is to generate a **clear, concise, and completely Unique Label** (2–3 words max) that best summarizes the main idea of the cluster **without duplicating or imitating any existing labels**.
    #                     📌 Existing labels in this graph which are given below:
    #                     {chr(10).join(f"- {label}" for label in sorted(existing_labels)) or 'None'}
    #                     ❗ VERY IMPORTANT:
    #                     - DO NOT use the same words, synonyms, or vague rephrasings of Existing Label.
    #                     - Only return the label. Do not include explanations, punctuation, or extra text.
    #                         """.strip()

    #             response = self.llm.model.invoke(prompt).content.strip()
    #             candidate = response or f"Cluster {cluster_id}"
    #             normalized = candidate.lower()

    #             if normalized in normalized_existing:
    #                 current_app.logger.warning(f"⚠️ LLM returned duplicate label '{candidate}'. Using fallback.")
    #                 label = f"Cluster {candidate}"
    #             else:
    #                 label = candidate

    #         cur.execute("""
    #             INSERT INTO cluster_labels (graph_id, cluster_id, label)
    #             VALUES (%s, %s, %s)
    #             ON CONFLICT (graph_id, cluster_id) DO UPDATE SET label = EXCLUDED.label
    #         """, (self.graph_id, cluster_id, label))
    #         conn.commit()

    #     except Exception as e:
    #         current_app.logger.error(f"❌ Error in label generation for cluster {cluster_id}: {str(e)}")
    #         label = f"Cluster {cluster_id}"

    #     finally:
    #         cur.close()
    #         conn.close()

    #     return label

  
    def query_graph_link_response(self, query: str) -> dict:
        current_app.logger.info(f"📝 Received query: {query!r}")

        q_emb = self.st_model.encode([query])[0]

        if not self.all_chunks or self.all_embeddings is None:
            return {"answer": "I don’t have enough information.", "link": None, "file": None}

        self._ensure_all_chunks_are_dicts()

        sims = cosine_similarity([q_emb], self.all_embeddings).flatten()
        k = max(5, int(len(self.all_chunks) * 0.01))
        seeds = sims.argsort()[-k:][::-1].tolist()

        expanded = self._redisgraph_expand(seeds, max_depth=2, limit=20)
        combined_idxs = list(seeds)
        for e in expanded:
            if e["idx"] not in combined_idxs:
                combined_idxs.append(e["idx"])

        expanded_map = {e["idx"]: e for e in expanded}
        u_list = []
        for idx in combined_idxs:
            if idx in expanded_map:
                u = expanded_map[idx]
            else:
                cu = self.all_chunks[idx]
                u = {
                    "text": cu.get("text", ""),
                    "source": cu.get("source", "")
                }
            u_list.append(u)

        texts = [u["text"] for u in u_list]
        embs2 = self.st_model.encode(texts)
        sims2 = cosine_similarity([q_emb], embs2).flatten()
        top5 = np.argsort(sims2)[-5:][::-1]

        units = []
        link_candidates = []
        for pos in top5:
            u = u_list[pos]
            src = u.get("source", "")
            if src and isinstance(src, str) and "http" in src:
                link_candidates.append(src)
            units.append(f"**Source:** {src}\n{u['text']}")

        context = "\n---\n".join(units)
        unique_links = list(dict.fromkeys(link_candidates)) 

        prompt = (
        "You are a helpful assistant. Respond to the user's question using only the context below.\n\n"
        f"📘 Context:\n{context}\n\n"
        f"🔗 Available Links (for your reference only):\n" + "\n".join(f"- {link}" for link in unique_links) + "\n\n"
        f"❓ Question:\n{query}\n\n"
        "**Instructions:**\n"
        "- Do NOT include or mention any hyperlinks, clickable labels, or '🔗' symbols in the response body.\n"
        "- Do NOT include or echo any lines like 'Schedule a Job for Task Group' or 'Lookup Category'.\n"
        "- Do NOT mention file names or PDFs in your answer.\n"
        "- At the end of your response, include only **one** relevant link (from the list above) on its own line. Nothing else.\n"
        "- If no relevant link exists, skip the link completely.\n"
        "- If there is not enough information to answer the question from the context, respond exactly with:\n"
        "  [NO_ANSWER]"
    )
        try:
            raw_response = self.llm.model.invoke(prompt)
            out = str(raw_response.content).strip()
        except Exception as e:
            current_app.logger.error(f"❌ LLM call failed: {str(e)}")
            return {"answer": "There was an error while generating the response.", "link": None, "file": None}

        current_app.logger.info(f"✅ LLM response: {out}")

        used_link = None
        if isinstance(out, str):
            for link in unique_links:
                if link in out:
                    used_link = link
                    break

        clean_response = out.replace(used_link, "").strip() if used_link else out

        if "[NO_ANSWER]" in clean_response:
            return {
                "answer": "I don’t have enough information.",
                "link": None,
                "sources": None
            }

        file_used = used_link or (u_list[top5[0]].get("source") if len(top5) > 0 else None)

        return {
            "answer": clean_response,
            "link": used_link,
            "sources": file_used
        }
  

    def load_graph_from_db(self, graph_id: int):
        storage = GraphPostgresStorage(dns=dns)
        G_nx = storage.load_graph(graph_id)

        self.graph_id = graph_id
        self.G_nx = G_nx
        self.chunk_node_ids = list(G_nx.nodes())
        self.chunks = [G_nx.nodes[n].get("text", "") for n in self.chunk_node_ids]

        self.all_chunks = []
        for n in self.chunk_node_ids:
            text = G_nx.nodes[n].get("text", "")
            source = G_nx.nodes[n].get("source")

            if source and isinstance(source, str):
                filename = os.path.splitext(os.path.basename(source))[0].strip()
                if filename:
                    text_lines = [
                        line for line in text.splitlines()
                        if filename.lower() not in line.lower()
                    ]
                    text = "\n".join(text_lines).strip()

            if not source or str(source).strip().lower() in {"", "none", "null"}:
                source = getattr(self, "source_url", "No Source Available")

            self.all_chunks.append({
                "text": text,
                "source":source})

        self.all_embeddings = self.st_model.encode(
            [chunk["text"] for chunk in self.all_chunks if chunk["text"]]
        )
        self._ensure_all_chunks_are_dicts()


    def load_graph_from_redis(self):
        """Load the entire semantic_graph from RedisGraph into memory."""
        store = GraphRedisStorage()
        graph = store.graph

        q_nodes = "MATCH (n:Chunk) RETURN n.idx, n.text, n.url, n.file_name, n.cluster"
        result  = graph.query(q_nodes)
        G_nx    = nx.Graph()

        for idx, text, url, fn, cluster in result.result_set:
            idx = int(idx)
            G_nx.add_node(
                idx,
                text    = text,
                url     = url or "",
                file_name = fn or "",
                cluster = int(cluster)
            )

        q_edges = "MATCH (a:Chunk)-[r:SIMILAR]->(b:Chunk) RETURN a.idx, b.idx, r.weight"
        result  = graph.query(q_edges)
        for u, v, w in result.result_set:
            G_nx.add_edge(int(u), int(v), weight=float(w))

        self.G_nx           = G_nx
        self.chunk_node_ids = list(G_nx.nodes())
        self.all_chunks     = [
            {
              "text":      G_nx.nodes[n]["text"],
              "url":       G_nx.nodes[n]["url"],
              "file_name": G_nx.nodes[n]["file_name"]
            }
            for n in self.chunk_node_ids
        ]
        texts = [c["text"] for c in self.all_chunks]
        self.all_embeddings = self.st_model.encode(texts)

        current_app.logger.info(f"🔄 Loaded graph from Redis: {len(self.all_chunks)} nodes, {G_nx.number_of_edges()} edges")
