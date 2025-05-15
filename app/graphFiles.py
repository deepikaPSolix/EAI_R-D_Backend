
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
from urllib.parse import urlparse
from pyvis.network import Network

class GraphFiles():
    def __init__(self):
        self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = LLMModel.from_together()
        self.all_chunks=None
        self.all_embeddings = None
   
    def build_similarity_graph(self,
                               chunks,
                               threshold: float = 0.85,
                               max_chunks_for_graph: int = 1500):
        current_app.logger.info(f"CHUNKS received: {len(chunks)}")
        def clean_str(v): return str(v).replace("\x00","") if isinstance(v,str) else v

        # 1) Build full NetworkX graph
        G_nx = nx.Graph(); processed_texts = []
        for idx, chunk in enumerate(chunks):
            try:
                if isinstance(chunk, dict):
                    text = clean_str(chunk["text"].strip())
                    url  = clean_str(chunk.get("url",""))
                    G_nx.add_node(idx, text=chunk, url=url, file_name=chunk.get("file_name",""))
                    processed_texts.append(text)
                else:
                    parsed = None
                    try: parsed = ast.literal_eval(chunk)
                    except: pass
                    if isinstance(parsed, dict):
                        text = clean_str(parsed["text"].strip())
                        url  = clean_str(parsed.get("url",""))
                        G_nx.add_node(idx, text=parsed, url=url, file_name=parsed.get("file_name",""))
                        processed_texts.append(text)
                    else:
                        parts = str(chunk).split("||",1)
                        if len(parts)==2:
                            fn, txt = parts
                            G_nx.add_node(idx, text=txt.strip(), url="", file_name=fn.strip())
                            processed_texts.append(txt.strip())
                        else:
                            ct = clean_str(str(chunk).strip())
                            G_nx.add_node(idx, text=ct, url="", file_name="")
                            processed_texts.append(ct)
            except Exception as e:
                current_app.logger.warning(f"Node {idx} failed: {e}")
                ct = clean_str(str(chunk))
                G_nx.add_node(idx, text=ct, url="", file_name="")
                processed_texts.append(ct)

        # 2) Embed all for RAG
        self.all_chunks     = chunks
        self.all_embeddings = self.st_model.encode(processed_texts)

        # 3) Trim to top-K by mean similarity
        mean_vec = np.mean(self.all_embeddings, axis=0, keepdims=True)
        sims     = cosine_similarity(mean_vec, self.all_embeddings).flatten()
        top_idx  = np.argsort(sims)[-max_chunks_for_graph:]
        emb_trim = self.all_embeddings[top_idx]

        # 4) Add semantic edges
        simm_mat = cosine_similarity(emb_trim)
        for i in range(len(emb_trim)):
            for j in range(i+1, len(emb_trim)):
                if simm_mat[i,j] >= threshold:
                    u, v = top_idx[i], top_idx[j]
                    G_nx.add_edge(u, v, weight=float(simm_mat[i,j]))

        # 5) Leiden clustering
        G_ig      = ig.Graph.TupleList(G_nx.edges(), directed=False)
        partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
        for idx, com in enumerate(partition.membership):
            G_nx.nodes[top_idx[idx]]['cluster'] = int(com)
        current_app.logger.info("✅ Leiden partition done")

        # 6) Persist to Postgres
        pg = GraphPostgresStorage(dsn=os.getenv("DSN"))
        self.graph_id = pg.save_graph(G_nx)
        current_app.logger.info(f"Saved graph {self.graph_id} to Postgres")

        # 7) Persist to RedisGraph
        redis_store = GraphRedisStorage()
        redis_store.store_graph(G_nx)
        current_app.logger.info("Saved graph to RedisGraph")

        # Wrap up for PyVis & querying
        self.G_nx           = G_nx
        self.chunk_node_ids = list(G_nx.nodes())
        self.chunks         = [G_nx.nodes[n]["text"] for n in self.chunk_node_ids]
        self.embeddings     = self.st_model.encode(self.chunks)

        return self.graph_id, G_nx


    def _redisgraph_expand(self, seed_ids, max_depth=2, limit=20):
        if not seed_ids:
            return []
        store     = GraphRedisStorage()
        graph     = store.graph
        seed_list = ",".join(str(i) for i in seed_ids)
        q = f"""
        MATCH (c:Chunk)-[:SIMILAR*1..{max_depth}]->(x:Chunk)
        WHERE c.idx IN [{seed_list}]
        RETURN DISTINCT x.idx, x.text, x.url, x.file_name
        LIMIT {limit}
        """
        result = graph.query(q)
        return [
            {"idx": int(r[0]), "text": r[1], "url": r[2], "file_name": r[3]}
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
        for node, data in G_nx.nodes(data=True):
            cluster = data.get("cluster", -1)
            clusters[cluster].append(node)

        current_app.logger.info(f"All clusters: {[(c, len(n)) for c, n in clusters.items()]}")
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        current_app.logger.info(f"📊 Cluster sizes: {[(c, len(n)) for c, n in sorted_clusters]}")

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
                continue  # optional: skip labeling unknown clusters
            cluster_text = "\n".join([get_clean_text(G_nx,node) for node in nodes])
            label = self.generate_cluster_label(cluster_text, cluster_id)
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

        # ✅ Sanitize node IDs before passing to Pyvis
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

        current_app.logger.info(f"✅ Render complete. Cluster labels: {list(all_labels.values())}")
        return net.generate_html()

    


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

            conn = psycopg2.connect(os.getenv("DSN"))
            cur = conn.cursor()

            for graph_id, cluster_ids in graph_cluster_map.items():
                cluster_ids = [int(cid) for cid in cluster_ids if int(cid) >= 0]

                cur.execute("""
                    SELECT gn.node_idx, gn.cluster_id, gn.text, gn.file_name, cl.label
                    FROM graph_nodes gn
                    LEFT JOIN cluster_labels cl ON cl.graph_id = gn.graph_id AND cl.cluster_id = gn.cluster_id
                    WHERE gn.graph_id = %s AND gn.cluster_id = ANY(%s)
                """, (graph_id, cluster_ids))
                rows = cur.fetchall()

                for node_idx, cluster_id, text, file_name, label in rows:
                    new_id = f"{graph_id}_{node_idx}"
                    combined_G.add_node(new_id, text=text, cluster=cluster_id, graph_id=graph_id, file_name=file_name)
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
            self.all_chunks = [combined_G.nodes[n]["text"] for n in self.chunk_node_ids]
            self.all_embeddings = self.st_model.encode(self.all_chunks)

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





    def generate_cluster_label(self, cluster_text: str, cluster_id: int) -> str:
        conn = psycopg2.connect(os.getenv("DSN"))
        cur = conn.cursor()

        try:
            cur.execute("""
                SELECT label FROM cluster_labels
                WHERE graph_id = %s AND cluster_id = %s
            """, (self.graph_id, cluster_id))
            row = cur.fetchone()
            if row and row[0]:
                label = row[0].strip()
                current_app.logger.info(f"🟢 Using cached label for cluster {cluster_id}: {label}")
                return label

            cur.execute("""
                SELECT label FROM cluster_labels
                WHERE graph_id = %s
            """, (self.graph_id,))
            existing_labels = [r[0].strip() for r in cur.fetchall() if r[0]]
            normalized_existing = {label.lower() for label in existing_labels}

            if not cluster_text.strip():
                label = f"Cluster {cluster_id}"
                current_app.logger.warning(f"⚠️ Cluster {cluster_id} is empty. Using default label.")
            else:
                prompt = f"""
You are an expert language model tasked with labeling semantic clusters in a knowledge graph.
🧠 Cluster #{cluster_id} Content:
{cluster_text.strip()}
🆕 Existing Label:{existing_labels}
Each cluster is a group of related topics or concepts. Your goal is to generate a **clear, concise, and completely Unique Label** (2–3 words max) that best summarizes the main idea of the cluster **without duplicating or imitating any existing labels**.
📌 Existing labels in this graph which are given below:
{chr(10).join(f"- {label}" for label in sorted(existing_labels)) or 'None'}
❗ VERY IMPORTANT:
- DO NOT use the same words, synonyms, or vague rephrasings of Existing Label.
- Only return the label. Do not include explanations, punctuation, or extra text.
    """.strip()

                response = self.llm.model.invoke(prompt).content.strip()
                candidate = response or f"Cluster {cluster_id}"
                normalized = candidate.lower()

                if normalized in normalized_existing:
                    current_app.logger.warning(f"⚠️ LLM returned duplicate label '{candidate}'. Using fallback.")
                    label = f"Cluster {candidate}"
                else:
                    label = candidate

            cur.execute("""
                INSERT INTO cluster_labels (graph_id, cluster_id, label)
                VALUES (%s, %s, %s)
                ON CONFLICT (graph_id, cluster_id) DO UPDATE SET label = EXCLUDED.label
            """, (self.graph_id, cluster_id, label))
            conn.commit()

        except Exception as e:
            current_app.logger.error(f"❌ Error in label generation for cluster {cluster_id}: {str(e)}")
            label = f"Cluster {cluster_id}"

        finally:
            cur.close()
            conn.close()

        return label

    # def query_graph_link_response(self, query: str) -> dict:
    #     """Query using top chunks to return LLM-generated response + related link"""
    #     query_vec = self.st_model.encode([query])[0]
    #     if not hasattr(self, 'all_chunks') or self.all_chunks is None:
    #         raise ValueError("❌ all_chunks not initialized. Please call build_similarity_graph first.")
    #     if self.all_embeddings is None:
    #         self.all_embeddings = self.st_model.encode([chunk['text'] for chunk in self.all_chunks if chunk.get('text')])

    #     similarities = cosine_similarity([query_vec], self.all_embeddings)[0]


    #     top_k = max(5, int(len(self.all_chunks) * 0.01))
    #     top_k_idx = np.argsort(similarities)[-top_k:][::-1]
    #     expanded = self._redisgraph_expand(top_k_idx, max_depth=2, limit=20)
    #     retrieved_units = []
    #     for i in top_k_idx:
    #         chunk = self.all_chunks[i]
    #         if isinstance(chunk, str) and "||" in chunk:
    #             filename, text = chunk.split("||", 1)
    #             retrieved_units.append({
    #                 "unit_id": i,
    #                 "text": text.strip(),
    #                 "url": "No Link Available",
                   
    #                 "file_name": filename.strip()
    #             })
    #         elif isinstance(chunk, dict):
    #             url = chunk.get("url") or "No Link Available" 
    #             retrieved_units.append({
    #                 "unit_id": i,
    #                 "text": chunk.get("text", ""),
    #                 "url": url,
                
    #                 "file_name": chunk.get("file_name")
    #             })
    #         else:
    #             current_app.logger.warning(f"Unexpected chunk format at index {i}: {chunk}")


    #     product_links = [unit["url"] for unit in retrieved_units]
    #     valid_links = [url for url in product_links if url != "No Link Available"]

    #     retrieved_texts = "\n---\n".join([
    #         f"[retrieved text: {unit['text']}" for unit in retrieved_units
    #     ])
    #     current_app.logger.info(f"len_chunks{len(retrieved_texts)}")

    #     prompt = self._format_prompt(query, valid_links, retrieved_texts)


    #     llm_response = self.llm.model.invoke(prompt).content.strip()

    #     file_name = next((unit.get("file_name") for unit in retrieved_units if unit.get("file_name")), None)
    #     current_app.logger.info(f"FILENAME:{file_name}")
    #     return {
    #         "answer": llm_response,
    #         "file": file_name 
    #     }
    def query_graph_link_response(self, query: str) -> dict:
        current_app.logger.info(f"📝 Received query: {query!r}")

        # 1) Embed the query
        q_emb = self.st_model.encode([query])[0]
        current_app.logger.info("🔢 Query embedded into vector.")

        # 2) Guard against empty state
        if not self.all_chunks or self.all_embeddings is None or len(self.all_embeddings) == 0:
            current_app.logger.warning("⚠️ No chunks or embeddings available; returning fallback.")
            return {"answer": "I don’t have enough information.", "link": None, "file": None}

        # 3) Seed search (flat cosine)
        Y = np.atleast_2d(self.all_embeddings)
        sims = cosine_similarity([q_emb], Y).flatten()
        k = max(5, int(len(self.all_chunks) * 0.01))
        seeds = sims.argsort()[-k:][::-1].tolist()
        current_app.logger.info(f"🎯 Seed selection: top {k} indices = {seeds}")

        # 4) Graph expansion
        expanded = self._redisgraph_expand(seeds, max_depth=2, limit=20)
        if expanded:
            current_app.logger.info(f"🌐 RedisGraph expansion returned {len(expanded)} chunks.")
        else:
            current_app.logger.info("🔄 No RedisGraph expansion; will fallback to seeds only.")

        # 5) Merge seeds + expanded for reranking
        combined_idxs = list(seeds)
        for e in expanded:
            if e["idx"] not in combined_idxs:
                combined_idxs.append(e["idx"])
        current_app.logger.info(f"🔗 Combined seeds+expanded = {combined_idxs}")

        # 6) Build unified list of chunk‐dicts
        expanded_map = {e["idx"]: e for e in expanded}
        u_list = []
        for idx in combined_idxs:
            if idx in expanded_map:
                u = expanded_map[idx]
                source_type = "expanded"
            else:
                cu = self.all_chunks[idx]
                source_type = "seed"
                if isinstance(cu, dict):
                    text_val = cu.get("text", "")
                    url_val  = cu.get("url", "")
                    fn_val   = cu.get("file_name", "")
                elif isinstance(cu, str) and "||" in cu:
                    fn_val, text_val = cu.split("||", 1)
                    url_val = ""
                else:
                    text_val, url_val, fn_val = str(cu), "", ""
                u = {"text": text_val, "url": url_val, "file_name": fn_val}
            u_list.append(u)

        # 7) Rerank combined by cosine
        texts = [u["text"] for u in u_list]
        embs2 = self.st_model.encode(texts)
        sims2 = cosine_similarity([q_emb], embs2).flatten()
        current_app.logger.info(f"🔄 Re-ranked {len(u_list)} combined chunks.")

        if len(sims2) == 0:
            current_app.logger.warning("⚠️ Rerank produced no candidates; returning fallback.")
            return {"answer": "I don’t have enough information.", "link": None, "file": None}

        # 8) Pick top-5 candidates
        top5 = np.argsort(sims2)[-5:][::-1]
        current_app.logger.info(f"🏅 Final top-5 positions: {top5}")

        # 9) Build context from top-5
        units = []
        sources = []   
        for rank, pos in enumerate(top5, start=1):
            u = u_list[pos]
            src = u["url"] or u["file_name"] or "No Link Available"
            if u["url"]:
                sources.append(u["url"])
            snippet = u["text"].replace("\n", " ")[:100]
            current_app.logger.info(f"   [{rank}] src={src!r}, snippet={snippet!r}")
            units.append(f"**Source:** {src}\n{u['text']}")

        context = "\n---\n".join(units)
        current_app.logger.info(f"📚 Built context with {len(units)} units, {len(context)} chars.")

        # 10) Prompt the LLM
        prompt = (
        "You are a helpful assistant which generates Response to the given Question below. Respond the user’s Question **directly** and **only** using the context below. "
        "Do **not** invent any information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "**Instructions:**\n"
        "1. If the context contains enough relevant details—even indirectly—use it to construct your response. \n"
        "2. If the context contains no relevant information, respond exactly:\n"
        "   I don’t have enough information. Please dont give source. "
        
    )

        current_app.logger.debug(f"🤖 Prompt to LLM:\n{prompt}")

        out = self.llm.model.invoke(prompt).content.strip()
        current_app.logger.info("✅ Received response from LLM.")

        # 11) Extract link & file
        link_match = re.search(r"https?://\S+", out)
        link = link_match.group(0) if link_match else None
        if link:
            current_app.logger.info(f"🔗 LLM answer included link: {link}")
        file_used = next((u["file_name"] for u in u_list if u.get("file_name")), None)
        if file_used:
            current_app.logger.info(f"📄 LLM answer referenced file: {file_used}")

        clean = re.sub(r"https?://\S+", "", out).strip()
        return {"answer": clean, "link": link, "file": file_used, "sources": sources}


       
    def _format_prompt(self, query, product_links, retrieved_texts):
        max_input_length = 6000
        trimmed_texts = retrieved_texts[:max_input_length]
#  f"Available Links:\n" + "\n".join(product_links) + "\n\n"
        return (
    f"""You are a helpful assistant. Answer the user’s question **directly** and **in Markdown**, without prefacing “the context shows…” or echoing back the question. 
— If you do have enough information in the provided context, just give the answer (with steps or bullets as needed).  
— If you don’t have exact information, respon"d exactly:  
    I don’t have enough information. else, respond related if you find any...
Use:
- Numbered lists for steps
- Bold for headings
- Line breaks between paragraphs"""
    f"context:\n{retrieved_texts}\n\n"
   
    f"User Question:\n{query}\n\n"
    f"Instructions:\n"
    "- Read the context carefully.\n"
    "- If the context includes a link relevant to your response, include **only one link** on a new line at the end.\n"
    "- Do **NOT** include fake links like 'your own generated'or  'No Link Available' or placeholder text.\n"
    "- If no link is available, **just skip it** — only return the answer.\n"
    "- If there's no relevant info in the context, return exactly: 'I don't have enough information.' **Do not return any links or sources in this case.**\n"  # Modified this line
    "- Stick strictly to the context provided. Don't make things up.\n"
    "- When providing step-by-step instructions, format them using numbered Markdown list syntax.\n"
    "- Format:\n"
    "  [Answer]\n"
    "  [Link / source — only if real, available, and useful]\n"
)
    def load_graph_from_db(self, graph_id: int):
        storage = GraphPostgresStorage(dsn=os.getenv("DSN"))
        G_nx = storage.load_graph(graph_id)

        # Store graph info for query
        self.graph_id = graph_id
        self.G_nx = G_nx
        self.chunk_node_ids = list(G_nx.nodes())
        self.chunks = [G_nx.nodes[n].get("text", "") for n in self.chunk_node_ids]

        # --- NEW: Build all_chunks with url from the node attributes ---
        self.all_chunks = [
            {
                "text":   G_nx.nodes[n].get("text", ""),
                "file_name": G_nx.nodes[n].get("file_name"," "),
                "url":    G_nx.nodes[n].get("url", "No Link Available")
            }
            for n in self.chunk_node_ids
        ]

        self.all_embeddings = self.st_model.encode(
            [chunk["text"] for chunk in self.all_chunks if chunk["text"]]
        )



        # inside GraphFiles in graphfiles.py

    def load_graph_from_redis(self):
        """Load the entire semantic_graph from RedisGraph into memory."""
        store = GraphRedisStorage()
        graph = store.graph

        # 1) Pull all nodes
        q_nodes = "MATCH (n:Chunk) RETURN n.idx, n.text, n.url, n.file_name, n.cluster"
        result  = graph.query(q_nodes)
        G_nx    = nx.Graph()

        for idx, text, url, fn, cluster in result.result_set:
            idx = int(idx)
            # text is stored literally
            G_nx.add_node(
                idx,
                text    = text,
                url     = url or "",
                file_name = fn or "",
                cluster = int(cluster)
            )

        # 2) Pull all SIMILAR edges
        q_edges = "MATCH (a:Chunk)-[r:SIMILAR]->(b:Chunk) RETURN a.idx, b.idx, r.weight"
        result  = graph.query(q_edges)
        for u, v, w in result.result_set:
            G_nx.add_edge(int(u), int(v), weight=float(w))

        # 3) Store for querying
        self.G_nx           = G_nx
        self.chunk_node_ids = list(G_nx.nodes())
        # rebuild all_chunks exactly as you did in build
        self.all_chunks     = [
            {
              "text":      G_nx.nodes[n]["text"],
              "url":       G_nx.nodes[n]["url"],
              "file_name": G_nx.nodes[n]["file_name"]
            }
            for n in self.chunk_node_ids
        ]
        # 4) Recompute embeddings
        texts = [c["text"] for c in self.all_chunks]
        self.all_embeddings = self.st_model.encode(texts)

        current_app.logger.info(f"🔄 Loaded graph from Redis: {len(self.all_chunks)} nodes, {G_nx.number_of_edges()} edges")
