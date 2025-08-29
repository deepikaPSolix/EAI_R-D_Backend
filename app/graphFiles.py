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
import time
from collections import Counter
from urllib.parse import urlparse
from pyvis.network import Network
from app.crew_cluster_labeler import generate_unique_label, batch_generate_cluster_labels
from app.chroma_db import ChromaDB  # Added import for ChromaDB
import uuid  # For generating unique IDs
import spacy
from nltk.corpus import stopwords

dns_host = os.getenv("DNS_HOST")
dns_dbname = os.getenv("DNS_DBNAME")
dns_user = os.getenv("DNS_USER")
dns_password = os.getenv("DNS_PASSWORD")
dns_port = os.getenv("DNS_PORT")
dns = f"host={dns_host} dbname={dns_dbname} user={dns_user} password={dns_password} port={dns_port}"
nlp = spacy.load("en_core_web_sm")
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    STOPWORDS = set()

def normalize_text(text):
    """
    Lowercase, remove stopwords, lemmatize.
    """
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in STOPWORDS]
    return " ".join(tokens)

def deduplicate_chunks(chunks, prioritize_graph=True):
    """
    Deduplicate chunks by their text content only (not by ID).
    Optionally prioritize graph-retrieved chunks over semantic ones.
    
    Args:
        chunks: List of chunk dictionaries
        prioritize_graph: Whether to prioritize graph-retrieved chunks
        
    Returns:
        List of deduplicated chunks
    """
    seen_texts = set()
    unique_chunks = []
    
    # Sort chunks by retrieval type if prioritizing graph chunks
    if prioritize_graph:
        chunks = sorted(chunks, key=lambda x: 0 if isinstance(x, dict) and x.get("retrieval_type") == "graph" else 1)
        current_app.logger.info(f"🔄 Sorting chunks to prioritize graph-based retrieval")
    
    duplicates_by_text = 0
    
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("data")
        else:
            text = str(chunk)
        if text and text in seen_texts:
            duplicates_by_text += 1
            continue
        if text:
            seen_texts.add(text)
        unique_chunks.append(chunk)
    
    current_app.logger.info(f"🔍 Deduplication: removed {duplicates_by_text} duplicates by text content")
    current_app.logger.info(f"📊 Deduplication results: {len(unique_chunks)} unique chunks from {len(chunks)} total")
    
    return unique_chunks
class GraphFiles():
    def __init__(self):
        # Use global/shared SentenceTransformer instance for performance
        # Lazy, per-process model init to avoid CUDA in forked children
        import torch
        from sentence_transformers import SentenceTransformer

        if not hasattr(self, "_st_model") or self._st_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # choose your model; keep it small unless you really need bigger
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.st_model = self._st_model

        self.llm = LLMModel.from_together()
        self.all_chunks=None
        self.all_embeddings = None        
        self.session_id = str(uuid.uuid4())[:8]  # Generate a unique session ID        # Initialize ChromaDB client for graph chunks
        self.chroma_db = ChromaDB()
        # Clean up old session collections to prevent unlimited growth
        deleted_count = self.chroma_db.cleanup_session_collections(max_collections=15, preserve_newest=10)
        if deleted_count > 0:
            current_app.logger.info(f"🧹 Cleaned up {deleted_count} old ChromaDB collections")
            
        # Clean up old Redis graphs to prevent unlimited growth
        redis_store = GraphRedisStorage()
        deleted_graphs = redis_store.cleanup_old_graphs(max_graphs=15)
        if deleted_graphs > 0:
            current_app.logger.info(f"🧹 Cleaned up {deleted_graphs} old Redis graphs")
            
        # Create a unique collection name for this session
        self.collection_name = f"graph_chunks_{self.session_id}"
        current_app.logger.info(f"🆕 Creating new session-specific ChromaDB collection: {self.collection_name}")
        # Create a new collection for this session (no need to delete anything)
        self.graph_collection = self.chroma_db.chroma_client.get_or_create_collection(name=self.collection_name)
        current_app.logger.info(f"[GRAPHFILES] Session ID: {self.session_id}")
    
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
            filename = os.path.splitext(os.path.basename(source))[0]
            current_app.logger.info(f"[GRAPHFILES] split_source_and_text: source='{source}', filename='{filename}'")
            cleaned_lines = []
            for line in text.splitlines():
                line_lower = line.strip().lower()
                # Strict: do not skip lines based on fuzzy filename matching
                # Only skip if the exact filename (with all spaces) is present
                if filename in line_lower:
                    continue
                cleaned_lines.append(line.strip())
            text = "\n".join(cleaned_lines)

        return source, text.strip()


    def build_similarity_graph(self,
                               chunks,
                               threshold: float = 0.85,
                               max_chunks_for_graph: int = 1500):
        current_app.logger.info(f"CHUNKS received: {len(chunks)}")
        
        # Deduplicate chunks by text before building the graph
        original_chunk_count = len(chunks)
        chunks = deduplicate_chunks(chunks, prioritize_graph=False)
        deduped_chunk_count = len(chunks)
        current_app.logger.info(f"🧹 Deduplicated input chunks for graph: {original_chunk_count} -> {deduped_chunk_count} unique chunks")
        
        # Normalize text before embedding
        for chunk in chunks:
            if isinstance(chunk, dict):
                chunk["text"] = normalize_text(chunk.get("text", ""))
            else:
                chunk = normalize_text(str(chunk))

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

        # After clustering, log cluster membership counts        from collections import Counter
        counts = Counter(partition.membership)
        
        for idx, com in enumerate(partition.membership):
            G_nx.nodes[top_idx[idx]]['cluster'] = int(com)
        current_app.logger.info("✅ Leiden partition done")        
        pg = GraphPostgresStorage(dns=dns)
        self.graph_id = pg.save_graph(G_nx)
        current_app.logger.info(f"Saved graph {self.graph_id} to Postgres")        # Use session-specific RedisGraph storage with optimized settings
        redis_store = GraphRedisStorage(session_id=self.session_id, timeout=120)  # Increased timeout for large graphs
        
        # Store graph with optimized batching
        start_time = time.time()
        current_app.logger.info(f"Storing graph with {G_nx.number_of_nodes()} nodes and {G_nx.number_of_edges()} edges to RedisGraph")
        result = redis_store.store_graph(G_nx)
        
        if result:
            # Run optimization after successful storage
            redis_store.optimize_graph()
            # Preload cache for faster subsequent queries
            redis_store.preload_cache()
            # Clean up old graphs to prevent Redis from growing too large
            deleted = redis_store.cleanup_old_graphs(max_graphs=20)
            if deleted > 0:
                current_app.logger.info(f"Cleaned up {deleted} old RedisGraph instances")
        
        elapsed = time.time() - start_time
        current_app.logger.info(f"Saved graph to RedisGraph with session ID {self.session_id} in {elapsed:.2f} seconds")

        # Store chunks in ChromaDB
        self.store_chunks_in_chroma()
        current_app.logger.info("Saved chunks to ChromaDB")

        # After Leiden clustering
        # Merge clusters with highly similar centroids
        cluster_ids = set(partition.membership)
        cluster_centroids = {}
        for cid in cluster_ids:
            indices = [i for i, c in enumerate(partition.membership) if c == cid]
            if indices:
                cluster_embs = emb_trim[indices]
                cluster_centroids[cid] = np.mean(cluster_embs, axis=0)
        merged = set()
        for cid1 in cluster_ids:
            for cid2 in cluster_ids:
                if cid1 >= cid2 or cid1 in merged or cid2 in merged:
                    continue
                sim = cosine_similarity([cluster_centroids[cid1]], [cluster_centroids[cid2]])[0][0]
                if sim > 0.80:
                    # Merge cid2 into cid1
                    for i, c in enumerate(partition.membership):
                        if c == cid2:
                            partition.membership[i] = cid1
                    merged.add(cid2)
        current_app.logger.info(f"✅ Merged clusters with highly similar centroids: {merged}")

        self.G_nx           = G_nx
        self.chunk_node_ids = list(G_nx.nodes())
        self.chunks         = [G_nx.nodes[n]["text"] for n in self.chunk_node_ids]
        self.embeddings     = self.st_model.encode(self.chunks)

        return self.graph_id, G_nx      
    def load_graph_from_redis(self):
        """
        Load graph data from RedisGraph when needed. This is a fallback method
        when in-memory graph is not available.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            current_app.logger.info(f"🔄 Attempting to load graph from session-specific RedisGraph {self.session_id}")
            redis_store = GraphRedisStorage(session_id=self.session_id, timeout=60)  # Increased timeout for large graphs
            
            # Create a new in-memory graph
            G_nx = nx.Graph()
            
            # Load nodes in batches for better memory management
            node_count = 0
            for node_batch in redis_store.get_nodes_in_batches(batch_size=1000):
                if not node_batch:
                    if node_count == 0:
                        current_app.logger.warning("⚠️ No nodes found in RedisGraph")
                        return False
                    else:
                        break
                
                # Add nodes to the graph
                for node in node_batch:
                    idx = int(node['idx'])
                    text = node['text']
                    source = node['source']
                    cluster = int(node['cluster']) if node['cluster'] is not None else -1
                    G_nx.add_node(idx, text=text, source=source, cluster=cluster)
                    node_count += 1
                
                current_app.logger.debug(f"Loaded {node_count} nodes from RedisGraph")
            
            if node_count == 0:
                current_app.logger.warning("⚠️ No nodes found in RedisGraph")
                return False
            
            # Load edges in batches for better memory management
            edge_count = 0
            for edge_batch in redis_store.get_edges_in_batches(batch_size=5000):
                if not edge_batch:
                    break
                
                # Add edges to the graph
                for source_idx, target_idx, weight in edge_batch:
                    # Skip invalid edges
                    if not G_nx.has_node(source_idx) or not G_nx.has_node(target_idx):
                        current_app.logger.warning(f"⚠️ Skipping edge between missing nodes: {source_idx}->{target_idx}")
                        continue
                        
                    G_nx.add_edge(source_idx, target_idx, weight=weight)
                    edge_count += 1
                
                current_app.logger.debug(f"Loaded {edge_count} edges from RedisGraph")
            
            # Update the instance variables
            self.G_nx = G_nx
            self.chunk_node_ids = list(G_nx.nodes())
            self.all_chunks = [
                {"text": G_nx.nodes[n].get("text", ""), "source": G_nx.nodes[n].get("source", "No Source Available")}
                for n in self.chunk_node_ids
            ]
            self._ensure_all_chunks_are_dicts()
            
            # Load or generate embeddings
            try:
                self.all_embeddings = self.st_model.encode([c["text"] for c in self.all_chunks])
                current_app.logger.info(f"✅ Successfully generated embeddings for {len(self.all_chunks)} chunks")
            except Exception as e:
                current_app.logger.error(f"❌ Failed to generate embeddings: {e}")
                return False
                
            current_app.logger.info(f"✅ Successfully loaded graph from RedisGraph with {G_nx.number_of_nodes()} nodes and {G_nx.number_of_edges()} edges")
            return True
            
        except Exception as e:
            current_app.logger.error(f"❌ Failed to load graph from RedisGraph: {e}", exc_info=True)
            return False
            self.G_nx = G_nx
            self.chunk_node_ids = list(G_nx.nodes())
            self.all_chunks = [
                {"text": G_nx.nodes[n].get("text", ""), "source": G_nx.nodes[n].get("source", "No Source Available")}
                for n in self.chunk_node_ids
            ]
            self._ensure_all_chunks_are_dicts()
            
            # Load or generate embeddings
            try:
                self.all_embeddings = self.st_model.encode([c["text"] for c in self.all_chunks])
                current_app.logger.info(f"✅ Successfully generated embeddings for {len(self.all_chunks)} chunks")
            except Exception as e:
                current_app.logger.error(f"❌ Failed to generate embeddings: {e}")
                return False
                
            current_app.logger.info(f"✅ Successfully loaded graph from RedisGraph with {G_nx.number_of_nodes()} nodes and {G_nx.number_of_edges()} edges")
            return True
            
        except Exception as e:
            current_app.logger.error(f"❌ Failed to load graph from RedisGraph: {e}", exc_info=True)
            return False    
    def _redisgraph_expand(self, seed_ids, max_depth=2, limit=20):
        if seed_ids is None or len(seed_ids) == 0:
            return []

        try:
            # Use session-specific RedisGraph storage with increased timeout
            store = GraphRedisStorage(session_id=self.session_id, timeout=30)
            graph = store.graph
            
            # Convert seed IDs to strings for consistent handling
            seed_str_ids = [str(i) for i in seed_ids]
            
            # Limit the number of seed IDs to process to avoid very large queries
            if len(seed_str_ids) > 10:
                current_app.logger.warning(f"⚠️ Too many seed IDs ({len(seed_str_ids)}), limiting to 10")
                seed_str_ids = seed_str_ids[:10]
            
            # Query using original_id property for non-numeric IDs
            numeric_seeds = []
            string_seeds = []
            
            for seed_id in seed_str_ids:
                try:
                    numeric_seeds.append(int(seed_id))
                except (ValueError, TypeError):
                    string_seeds.append(seed_id)
            
            # Optimize query based on seed types
            if not numeric_seeds and not string_seeds:
                return []
                
            # Use parameters for query to prevent injection and improve caching
            params = {}
            
            # Build efficient query with parameters
            if numeric_seeds and string_seeds:
                # Mixed case: both numeric and string IDs
                params['numeric_ids'] = numeric_seeds
                params['string_ids'] = string_seeds
                
                q = f"""
                MATCH (c:Chunk)
                WHERE c.idx IN $numeric_ids OR c.original_id IN $string_ids
                MATCH (c)-[:SIMILAR*1..{max_depth}]->(x:Chunk)
                RETURN DISTINCT x.idx, x.text, x.source, x.original_id
                LIMIT {limit}
                """
            elif numeric_seeds:
                # Only numeric IDs
                params['numeric_ids'] = numeric_seeds
                
                q = f"""
                MATCH (c:Chunk)
                WHERE c.idx IN $numeric_ids
                MATCH (c)-[:SIMILAR*1..{max_depth}]->(x:Chunk)
                RETURN DISTINCT x.idx, x.text, x.source, x.original_id
                LIMIT {limit}
                """
            else:
                # Only string IDs
                params['string_ids'] = string_seeds
                
                q = f"""
                MATCH (c:Chunk)
                WHERE c.original_id IN $string_ids
                MATCH (c)-[:SIMILAR*1..{max_depth}]->(x:Chunk)
                RETURN DISTINCT x.idx, x.text, x.source, x.original_id
                LIMIT {limit}
                """
              # Execute the query with parameters
            start_time = time.time()
            result = store.query_with_timeout(q, params=params, timeout=30)  # Use timeout-protected query
            query_time = time.time() - start_time
            
            if result is None:
                current_app.logger.warning("Graph expansion query timed out, returning empty results")
                return []
                
            current_app.logger.debug(f"Graph expansion query completed in {query_time:.2f} seconds")
            
            valid_results = []
            
            max_idx = len(self.all_chunks) - 1 if self.all_chunks else -1
            
            for r in result.result_set:
                idx = int(r[0])
                original_id = r[3] if len(r) > 3 else str(idx)
                
                try:
                    idx_to_use = int(original_id)
                except ValueError:
                    idx_to_use = original_id
                
                if isinstance(idx_to_use, int) and (max_idx == -1 or idx_to_use <= max_idx):
                    valid_results.append({"idx": idx_to_use, "text": r[1], "source": r[2]})
                elif isinstance(idx_to_use, str):
                    valid_results.append({"idx": idx_to_use, "text": r[1], "source": r[2]})
                else:
                    current_app.logger.warning(f"⚠️ Graph returned out-of-range index: {idx_to_use} (max valid: {max_idx})")
            
            current_app.logger.info(f"📊 Graph expansion: {len(valid_results)}/{len(result.result_set)} valid results")
            return valid_results
            
        except Exception as e:
            current_app.logger.error(f"❌ Graph expansion failed: {str(e)}", exc_info=True)
            return []

    def get_main_node_label_from_url(self, url):
        netloc = urlparse(url).netloc
        if (netloc.startswith("www.")):
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

        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

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

        cluster_texts = []
        cluster_ids = []
        for cluster_id, nodes in clusters.items():
            if cluster_id == -1:
                continue
            cluster_text = "\n".join([get_clean_text(G_nx, node) for node in nodes])
            cluster_texts.append(cluster_text)
            cluster_ids.append(cluster_id)
        labels_dict = batch_generate_cluster_labels(cluster_texts, max_workers=5)
        all_labels = {}
        for idx, cluster_id in enumerate(cluster_ids):
            label = labels_dict.get(idx, f"Cluster {cluster_id}")
            existing_labels.add(label.lower())
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
            
            redis_store = GraphRedisStorage(session_id=self.session_id)
            redis_store.store_graph(combined_G)
            current_app.logger.info(f"Stored combined clusters in session-specific RedisGraph '{redis_store.graph_name}'")
            
            self.store_chunks_in_chroma()
            current_app.logger.info(f"Stored combined clusters in session-specific ChromaDB '{self.collection_name}'")

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

    def store_chunks_in_chroma(self):
        """
        Store all chunks in ChromaDB for efficient vector retrieval.
        Uses a session-specific collection to ensure isolation between different runs.
        """
        if not self.all_chunks:
            current_app.logger.warning("No chunks to store in ChromaDB")
            return False
        
        try:
            ids = [str(uuid.uuid4()) for _ in range(len(self.all_chunks))]
            texts = [chunk["text"] for chunk in self.all_chunks]
            metadatas = []
            
            for i, chunk in enumerate(self.all_chunks):
                cluster_id = -1
                if hasattr(self, "G_nx") and self.G_nx.has_node(i):
                    cluster_id = self.G_nx.nodes[i].get("cluster", -1)
                
                source = chunk.get("source", "No Source Available")
                if source is None:
                    source = "No Source Available"
                    
                graph_id = getattr(self, "graph_id", None)
                if graph_id is None:
                    graph_id = -1
                    
                file_names = chunk.get("file_names", [])
                if file_names is None:
                    file_names = []
                file_names_str = ",".join(file_names) if file_names else ""
                
                metadata = {
                    "source": source,
                    "graph_id": graph_id,
                    "node_id": i,
                    "cluster_id": cluster_id,
                    "chunk_id": str(uuid.uuid4()),
                    "file_names": file_names_str
                }
                
                for key, value in list(metadata.items()):
                    if value is None:
                        metadata[key] = ""
                
                metadatas.append(metadata)
            
            self.graph_collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            self.chroma_id_map = {id_str: idx for idx, id_str in enumerate(ids)}
            
            current_app.logger.info(f"✅ Successfully stored {len(ids)} chunks in session-specific ChromaDB collection '{self.collection_name}'")
            return True
        except Exception as e:
            current_app.logger.error(f"❌ Failed to store chunks in ChromaDB: {e}", exc_info=True)
            return False
    def query_chroma_chunks(self, query_text, n_results=8):
        """
        Query the ChromaDB collection for chunks relevant to the query.
        
        Args:
            query_text (str): The query text
            n_results (int): Maximum number of results to return
            
        Returns:
            List of chunk dictionaries with text and source
        """
        if not hasattr(self, "graph_collection"):
            current_app.logger.warning("❌ ChromaDB collection not initialized")
            return []
            
        try:              
            current_app.logger.info(f"🔍 Querying current session's ChromaDB collection '{self.collection_name}' with '{query_text[:50]}...' for {n_results} results")
            
            available_chunks = len(self.all_chunks) if self.all_chunks else 100
            actual_n_results = min(n_results, available_chunks)
            
            results = self.graph_collection.query(
                query_texts=[query_text],
                n_results=actual_n_results
            )
            
            chunks = []
            if results and len(results["ids"]) > 0:
                num_results = len(results["ids"][0])
                current_app.logger.info(f"✅ Current session's ChromaDB collection returned {num_results} results")
                
                for i in range(num_results):
                    distance = results["distances"][0][i] if "distances" in results else None
                    source = results["metadatas"][0][i].get("source", "No Source Available")
                    cluster_id = results["metadatas"][0][i].get("cluster_id", -1)
                    
                    chunk = {
                        "text": results["documents"][0][i],
                        "source": source,
                        "graph_node_idx": results["metadatas"][0][i].get("node_id"),
                        "chunk_id": results["metadatas"][0][i].get("chunk_id"),
                        "cluster_id": cluster_id,                        
                        "graph_id": results["metadatas"][0][i].get("graph_id"),
                        "distance": distance
                    }
                    chunks.append(chunk)
                    
                    current_app.logger.debug(f"📄 Result {i+1}: Source={source}, Cluster={cluster_id}, Distance={f'{distance:.4f}' if distance is not None else 'N/A'}")
                
                sources = Counter([c.get("source", "Unknown") for c in chunks])
                clusters = Counter([c.get("cluster_id", -1) for c in chunks])
                
                current_app.logger.info(f"📊 Retrieved chunks from {len(sources)} unique sources across {len(clusters)} clusters")
                    
            else:
                current_app.logger.warning("⚠️ ChromaDB query returned no results")
                
            return chunks
        except Exception as e:
            current_app.logger.error(f"❌ ChromaDB query failed: {e}", exc_info=True)
            return []
    def query_graph_link_response(self, query: str) -> dict:
        """
        Enhanced hybrid RAG method that combines ChromaDB vector search with graph traversal
        to find the most relevant context for answering the query.

        Args:
        query: The user's question

        Returns:
        Dictionary with answer, source file, and other metadata
        """
        current_app.logger.info(f"📝 [START] Processing query: {query!r}")

        if not self.all_chunks:
            current_app.logger.warning("❌ No chunks available for retrieval")
            return {"answer": "I don't have enough information.", "source": None}

        current_app.logger.info(f"📊 Total chunks in memory: {len(self.all_chunks)}")

        all_relevant_chunks = []
        graph_chunks = []
        semantic_chunks = []
        source_files = set()
        
        current_app.logger.info("🔍 [STEP 1] Retrieving semantically similar chunks from ChromaDB")
        max_chroma_results = min(len(self.all_chunks) if self.all_chunks else 50, 100)
        chroma_chunks = self.query_chroma_chunks(query, n_results=max_chroma_results)
        if chroma_chunks:
            current_app.logger.info(f"✅ Found {len(chroma_chunks)} relevant chunks from ChromaDB")
            for chunk in chroma_chunks:
                chunk["retrieval_type"] = "semantic"
            semantic_chunks.extend(chroma_chunks)
            for chunk in chroma_chunks:
                if chunk.get('source'):
                    source_files.add(chunk.get('source'))
        else:
            current_app.logger.warning("⚠️ No chunks found in ChromaDB")
        current_app.logger.info("🔍 [STEP 2] Performing graph-based retrieval")
        self._ensure_all_chunks_are_dicts()

        current_app.logger.info("📊 Computing query embedding")
        q_emb = self.st_model.encode([query])[0]

        sims = cosine_similarity([q_emb], self.all_embeddings).flatten()
        total_chunks = len(self.all_chunks)
        if total_chunks <= 20:
            k = max(3, total_chunks // 2)
        elif total_chunks <= 100:
            k = max(10, total_chunks // 4)
        else:
            k = max(20, total_chunks // 10)
        seeds = sims.argsort()[-k:][::-1].tolist()
        current_app.logger.info(f"✅ Found {len(seeds)} initial seed chunks based on embedding similarity")

        current_app.logger.info(f"🔍 Expanding graph with depth=2 from {len(seeds)} seed nodes")
        expansion_limit = max(30, min(total_chunks, total_chunks // 3))
        expanded = self._redisgraph_expand(seeds, max_depth=2, limit=expansion_limit)
        current_app.logger.info(f"✅ Graph expansion found {len(expanded)} additional connected chunks")
        combined_idxs = list(seeds)

        for e in expanded:
            if e["idx"] not in combined_idxs:
                combined_idxs.append(e["idx"])
        
        current_app.logger.info(f"📊 Total unique graph nodes after expansion: {len(combined_idxs)}")
        for idx in combined_idxs:
            if idx < 0 or idx >= len(self.all_chunks):
                current_app.logger.warning(f"⚠️ Graph returned invalid node index: {idx} (max index: {len(self.all_chunks)-1})")
                continue
                
            cu = self.all_chunks[idx]
            chunk = {
                "text": cu.get("text", ""),
                "source": cu.get("source", "No Source Available"),
                "graph_node_idx": idx,
                "retrieval_type": "graph"
            }
            graph_chunks.append(chunk)
            if chunk.get('source'):
                source_files.add(chunk.get('source'))
        current_app.logger.info("🔍 [STEP 3] Merging and deduplicating chunks")
        current_app.logger.info(f"📊 Before deduplication: {len(graph_chunks)} graph chunks, {len(semantic_chunks)} semantic chunks")
        
        all_chunks_combined = graph_chunks + semantic_chunks

        all_relevant_chunks = deduplicate_chunks(all_chunks_combined, prioritize_graph=True)

        current_app.logger.info(f"📊 After deduplication: {len(all_relevant_chunks)} unique chunks")
        current_app.logger.info(f"📊 Breakdown: {len([c for c in all_relevant_chunks if c.get('retrieval_type') == 'graph'])} graph chunks, {len([c for c in all_relevant_chunks if c.get('retrieval_type') == 'semantic'])} semantic chunks")

        current_app.logger.info("🔍 [STEP 4] Re-ranking chunks based on relevance to query")
        texts = [c["text"] for c in all_relevant_chunks]
        if texts:
            current_app.logger.info("📊 Computing similarity scores for re-ranking")
            q_emb = self.st_model.encode([query])[0]
            chunk_embs = self.st_model.encode(texts)
            sims = cosine_similarity([q_emb], chunk_embs).flatten()

            available_chunks = len(all_relevant_chunks)
            
            query_words = len(query.split())
            is_complex_query = query_words > 10 or any(word in query.lower() for word in ['analyze', 'compare', 'detailed', 'comprehensive', 'all', 'every', 'list'])
            
            if is_complex_query:
                max_context_chunks = min(available_chunks, max(20, available_chunks // 2))
            else:
                max_context_chunks = min(available_chunks, max(10, available_chunks // 3))
                
            top_k = max_context_chunks
            top_indices = np.argsort(sims)[::-1][:top_k]
            current_app.logger.info(f"📊 Selected top {len(top_indices)} chunks for context (query complexity: {'high' if is_complex_query else 'normal'})")

            current_app.logger.info("📝 Building context for LLM prompt")
            context_units = []
            
            for i in top_indices:
                chunk = all_relevant_chunks[i]
                src = chunk.get("source", "Unknown Source")
                retrieval_type = chunk.get("retrieval_type", "unknown")
                similarity_score = sims[i]
                
                current_app.logger.info(f"📄 Including chunk from {src} (type: {retrieval_type}, similarity: {similarity_score:.4f})")
                
                chunk_text = chunk['text']
                
                if any(pattern in query.lower() for pattern in ['id', 'number', 'amount', 'date', 'order']):
                    import re
                    filename_base = os.path.splitext(os.path.basename(src))[0]
                    
                    numbers_in_filename = re.findall(r'\d+', filename_base)
                    if numbers_in_filename:
                        chunk_text = f"[DOCUMENT: {filename_base} - Contains: {', '.join(numbers_in_filename)}]\n{chunk_text}"
                
                import re
                url_pattern = r"^https?://"
                if re.match(url_pattern, src):
                    context_units.append(f"**Source Document:** {src}\n**Content:** {chunk_text}")
                else:
                    context_units.append(f"**Source Document:** {os.path.basename(src)}\n**Content:** {chunk_text}")
                
                if src:
                    source_files.add(src)

            context = "\n---\n".join(context_units)
            current_app.logger.info(f"📊 Final context built with {len(context_units)} chunks")
        else: 
            current_app.logger.warning("⚠️ No text content found in chunks")
            context = ""
            top_indices = []
        top_sources = []
        seen_sources = set()
        source_reasons = {}
        for i in top_indices:
            chunk = all_relevant_chunks[i]
            src = chunk.get("source", "")
            if src and src not in seen_sources:
                top_sources.append(src)
                seen_sources.add(src)
                reason = {
                    "picked_text": chunk["text"],
                    "similarity_score": float(sims[i]),
                    "retrieval_type": chunk.get("retrieval_type", "unknown"),
                    "query": query
                }
                source_reasons[src] = reason
                current_app.logger.info(f"🔎 Source picked: {src} | Reason: {reason}")

        unique_sources = len(top_sources)

        current_app.logger.info("🔍 [STEP 5] Building LLM prompt with context")
        prompt = (
                "You are an expert document analyst. Your task is to thoroughly examine ALL provided context and extract EVERY relevant piece of information that answers the user's question.\n\n"
                f" Context from {unique_sources} document sources:\n{context}\n\n"
                f" User Query: {query}\n\n"
                "**CRITICAL INSTRUCTIONS - READ CAREFULLY:**\n"
                " COMPREHENSIVE ANALYSIS REQUIRED:\n"
                "- Examine EVERY SINGLE document excerpt in the context above\n"
                "- Do NOT stop after finding just a few items - search through ALL content\n"
                "- Look for patterns, numbers, IDs, names, dates, or ANY data points relevant to the query\n"
                "- If the query asks for a list or collection, find ALL instances across ALL documents\n"
                "- Process each document section systematically and thoroughly\n\n"
                " PROCESSING APPROACH:\n"
                "- Scan through each source document methodically\n"
                "- Cross-reference information between documents\n"
                "- Consolidate findings from multiple sources\n"
                "- Present information in a clear, organized manner\n"
                "- Include quantitative details when available (counts, amounts, percentages, etc.)\n\n"
                " QUALITY STANDARDS:\n"
                "- Be exhaustive in your search - don't miss any relevant data\n"
                "- Maintain accuracy - only include information explicitly stated in the context\n"
                "- If information spans multiple documents, synthesize it comprehensively\n"
                "- For numerical data, include specific values, not approximations\n"
                "- If you cannot find sufficient information, respond only with the context information with the related answer.\n\n"
                "🎯 OUTPUT REQUIREMENTS:\n"
                "- Provide complete, thorough responses\n"
                "- Structure your answer logically\n"
                "- Include all relevant findings, not just highlights\n"
                "- Be comprehensive.\n\n"
                "- At the end of your answer, add a line: Sources Used: <comma-separated list of source document names you used for your answer, up to 3>"
            )
        try:
            current_app.logger.info("🔍 [STEP 6] Generating response with LLM")
            current_app.logger.info(f"📝 Sending comprehensive prompt with {len(prompt)} characters to LLM")
            raw_response = self.llm.model.invoke(prompt)
            response_text = str(raw_response.content).strip()
            current_app.logger.info(f"✅ LLM generated a response of {len(response_text)} characters")

            if "[NO_ANSWER]" in response_text:
                current_app.logger.warning("⚠️ LLM indicated insufficient information")
                return {"answer": "I don't have enough information.", "source": None}

            import re
            sources_used = []
            match = re.search(r"Sources Used:\s*(.*)", response_text)
            if match:
                sources_line = match.group(1)
                sources_used = [s.strip() for s in sources_line.split(",") if s.strip()]
                sources_used = sources_used[:3]
                response_text = re.sub(r"Sources Used:.*", "", response_text).strip()
            else:
                sources_used = top_sources[:3]

            primary_source = sources_used[0] if sources_used else None
            if len(top_indices) > 0:
                top_chunk = all_relevant_chunks[top_indices[0]]
                if top_chunk.get('source'):
                    primary_source = top_chunk.get('source')
                    current_app.logger.info(f"📄 Primary source set to: {primary_source}")

            graph_chunk_count = len([c for c in all_relevant_chunks if c.get("retrieval_type") == "graph"])
            semantic_chunk_count = len([c for c in all_relevant_chunks if c.get("retrieval_type") == "semantic"])
            current_app.logger.info(f"""
            📊 [PERFORMANCE METRICS]
            - Total chunks considered: {len(all_relevant_chunks)}
            - Graph-based chunks: {graph_chunk_count}
            - Semantic chunks: {semantic_chunk_count}
            - Source files referenced: {len(source_files)}
            - Response length: {len(response_text)} characters
            """)

            return {
                "answer": response_text,
                "sources": sources_used,
                "stats": {
                    "total_chunks": len(all_relevant_chunks),
                    "graph_chunks": graph_chunk_count,
                    "semantic_chunks": semantic_chunk_count,
                    "source_files": len(source_files),
                    "unique_sources": len(sources_used)
                }
            }

        except Exception as e:
            current_app.logger.error(f"❌ LLM call failed: {str(e)}", exc_info=True)
            return {"answer": "There was an error while generating the response.", "source": None}