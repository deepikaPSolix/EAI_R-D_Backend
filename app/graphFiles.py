
from flask import current_app, Flask
import networkx as nx
import igraph as ig
from collections import defaultdict
import leidenalg
import os
import time  # Ensure time is imported for performance tracking
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
import json
from collections import Counter
from urllib.parse import urlparse
from pyvis.network import Network
from app.crew_cluster_labeler import generate_unique_label, batch_generate_cluster_labels
from app.chroma_db import ChromaDB  # Added import for ChromaDB
import uuid  # For generating unique IDs
import spacy
from nltk.corpus import stopwords
from pathlib import Path

dns_host = os.getenv("DNS_HOST") or os.getenv("DB_HOST")
dns_dbname = os.getenv("DNS_DBNAME") or os.getenv("DB_NAME")
dns_user = os.getenv("DNS_USER") or os.getenv("DB_USER")
dns_password = os.getenv("DNS_PASSWORD") or os.getenv("DB_PASSWORD")
dns_port = os.getenv("DNS_PORT") or os.getenv("DB_PORT", "5432")
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
    if not chunks:
        return []
        
    # Use a dictionary for faster lookups with hash-based text matching
    # This is faster than using a set + separate list
    unique_dict = {}
    
    # Skip sorting if there's nothing to prioritize
    if prioritize_graph and any(isinstance(c, dict) and c.get("retrieval_type") == "graph" for c in chunks):
        # Faster sorting with direct comparison - prioritize graph chunks
        chunks = sorted(chunks, key=lambda x: 0 if isinstance(x, dict) and x.get("retrieval_type") == "graph" else 1)
    
    # Fast path: process all chunks in a single pass
    for chunk in chunks:
        # Handle dictionaries (most common case)
        if isinstance(chunk, dict):
            # Fast direct lookup, no conditional branching
            text = chunk.get("text", chunk.get("data", ""))
        else:
            # Fallback for non-dict chunks
            text = str(chunk)
            
        # Skip empty text
        if not text:
            continue
            
        # Store only the first occurrence (which will be a graph chunk if prioritized)
        if text not in unique_dict:
            unique_dict[text] = chunk
    
    # Return values from the dictionary (already unique)
    return list(unique_dict.values())

class GraphFiles():
    def __init__(self):
        # Reuse the global SentenceTransformer from app/__init__.py to save GPU memory
        from app import st_model as _global_st
        self.st_model = _global_st

        self.llm = LLMModel.from_openai()
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
                if isinstance(chunk, dict) and "text" in chunk:
                    # Preserve both source and url fields when available
                    source = chunk.get("source", chunk.get("url", "No Source Available"))
                    valid_chunks.append({
                        "text": chunk.get("text", ""),
                        "source": source
                    })
                elif isinstance(chunk, dict):
                    # Check for URL or source in any dict format
                    source = chunk.get("source", chunk.get("url", "No Source Available"))
                    valid_chunks.append({
                        "text": chunk.get("text", ""),
                        "source": source
                    })
                elif isinstance(chunk, str):
                    try:
                        parsed = ast.literal_eval(chunk)
                        if isinstance(parsed, dict):
                            # Check for URL or source in parsed dict
                            source = parsed.get("source", parsed.get("url", "No Source Available"))
                            valid_chunks.append({
                                "text": parsed.get("text", ""),
                                "source": source
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
            # Check if source is a URL - if so, don't do filename extraction
            if source.startswith(("http://", "https://")):
                # For URLs, preserve the full URL as the source and clean the text
                cleaned_lines = [line.strip() for line in text.splitlines()]
                text = "\n".join(cleaned_lines)
                # No additional processing needed for URLs - keep the URL as is
            else:
                # For files, extract the filename and remove filename references from text
                filename = os.path.splitext(os.path.basename(source))[0]
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
                            # Check for URL in both 'url' and 'source' keys
                            source = parsed.get("url", parsed.get("source", "")).strip()
                        else:
                            source, text = self.split_source_and_text(chunk)
                    except Exception:
                        source, text = self.split_source_and_text(chunk)

                elif isinstance(chunk, dict):
                    text = chunk.get("text", "").strip()
                    # Check for URL in both 'url' and 'source' keys 
                    source = chunk.get("url", chunk.get("source", "")).strip()
                    
                # Ensure URLs are preserved as-is in source
                if source and source.startswith(("http://", "https://")):
                    # For URLs, don't do any filename extraction or manipulation
                    # Simply preserve the URL as the source
                    pass
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

        # EMBED: compute normalized numpy embeddings once (cosine-ready)
        self.all_embeddings = self.st_model.encode(
            [c["text"] for c in self.all_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

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

        # (left as-is) — no behavioral change elsewhere that depends on this line
        # self.embeddings     = self.st_model.encode(self.chunks)

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
            
            # EMBED: generate normalized embeddings (cosine-ready)
            try:
                self.all_embeddings = self.st_model.encode(
                    [c["text"] for c in self.all_chunks],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                current_app.logger.info(f"✅ Successfully generated embeddings for {len(self.all_chunks)} chunks")
            except Exception as e:
                current_app.logger.error(f"❌ Failed to generate embeddings: {e}")
                return False
                
            current_app.logger.info(f"✅ Successfully loaded graph from RedisGraph with {G_nx.number_of_nodes()} nodes and {G_nx.number_of_edges()} edges")
            return True
            
        except Exception as e:
            current_app.logger.error(f"❌ Failed to load graph from RedisGraph: {e}", exc_info=True)
            return False
            # (duplicated block below left untouched to avoid changing other logic)

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

    def _redisgraph_expand(self, seed_ids, max_depth=1, limit=20):
        """
        Expand seed nodes in the graph to find connected chunks.
        
        Args:
            seed_ids: List of node IDs to use as starting points
            max_depth: Maximum traversal depth (1 or 2 recommended)
            limit: Maximum number of results to return
            
        Returns:
            List of expanded nodes with their attributes
        """
        if seed_ids is None or len(seed_ids) == 0:
            current_app.logger.warning("⚠️ No seed IDs provided for graph expansion")
            return []

        start_time = time.time()
        try:
            # Use session-specific RedisGraph storage with increased timeout
            store = GraphRedisStorage(session_id=self.session_id, timeout=30)
            graph = store.graph
            
            # Convert seed IDs to strings for consistent handling
            seed_str_ids = [str(i) for i in seed_ids]
            
            # OPTIMIZATION: Intelligently limit seeds based on depth
            # For depth=1, we can process more seeds efficiently
            # For depth=2, limit more aggressively to avoid exponential explosion
            max_seeds = 15 if max_depth == 1 else 8
            
            if len(seed_str_ids) > max_seeds:
                current_app.logger.info(f"🔍 Too many seed IDs ({len(seed_str_ids)}), limiting to {max_seeds} for depth={max_depth}")
                seed_str_ids = seed_str_ids[:max_seeds]
            
            # Split seeds into numeric and string types for optimized query
            numeric_seeds = []
            string_seeds = []
            
            for seed_id in seed_str_ids:
                try:
                    numeric_seeds.append(int(seed_id))
                except (ValueError, TypeError):
                    string_seeds.append(seed_id)
            
            if not numeric_seeds and not string_seeds:
                return []
                
            # OPTIMIZATION: Build query parameters once
            params = {}
            
            # OPTIMIZATION: Sort seeds for better cache hits
            if numeric_seeds:
                numeric_seeds.sort()
                params['numeric_ids'] = numeric_seeds
            
            if string_seeds:
                string_seeds.sort()
                params['string_ids'] = string_seeds
            
            # OPTIMIZATION: Build efficient query with parameters and better indexes
            query_parts = []
            
            # For better performance, use more focused queries based on depth
            if max_depth == 1:
                # Direct neighbors query (faster)
                if numeric_seeds and string_seeds:
                    q = """
                    MATCH (c:Chunk)
                    WHERE c.idx IN $numeric_ids OR c.original_id IN $string_ids
                    MATCH (c)-[:SIMILAR]->(x:Chunk)
                    RETURN DISTINCT x.idx, x.text, x.source, x.original_id
                    LIMIT $limit
                    """
                elif numeric_seeds:
                    q = """
                    MATCH (c:Chunk)
                    WHERE c.idx IN $numeric_ids
                    MATCH (c)-[:SIMILAR]->(x:Chunk)
                    RETURN DISTINCT x.idx, x.text, x.source, x.original_id
                    LIMIT $limit
                    """
                else:
                    q = """
                    MATCH (c:Chunk)
                    WHERE c.original_id IN $string_ids
                    MATCH (c)-[:SIMILAR]->(x:Chunk)
                    RETURN DISTINCT x.idx, x.text, x.source, x.original_id
                    LIMIT $limit
                    """
            else:
                # Multi-hop traversal (slower)
                if numeric_seeds and string_seeds:
                    q = f"""
                    MATCH (c:Chunk)
                    WHERE c.idx IN $numeric_ids OR c.original_id IN $string_ids
                    MATCH (c)-[:SIMILAR*1..{max_depth}]->(x:Chunk)
                    RETURN DISTINCT x.idx, x.text, x.source, x.original_id
                    LIMIT $limit
                    """
                elif numeric_seeds:
                    q = f"""
                    MATCH (c:Chunk)
                    WHERE c.idx IN $numeric_ids
                    MATCH (c)-[:SIMILAR*1..{max_depth}]->(x:Chunk)
                    RETURN DISTINCT x.idx, x.text, x.source, x.original_id
                    LIMIT $limit
                    """
                else:
                    q = f"""
                    MATCH (c:Chunk)
                    WHERE c.original_id IN $string_ids
                    MATCH (c)-[:SIMILAR*1..{max_depth}]->(x:Chunk)
                    RETURN DISTINCT x.idx, x.text, x.source, x.original_id
                    LIMIT $limit
                    """
            
            # Add limit as parameter for better query caching
            params['limit'] = limit
              
            # Execute the query with parameters and timeout protection
            current_app.logger.info(f"🔍 Expanding graph from {len(seed_str_ids)} seeds with depth={max_depth}, limit={limit}")
            query_start = time.time()
            result = store.query_with_timeout(q, params=params, timeout=30)
            query_time = time.time() - query_start
            
            if result is None:
                current_app.logger.warning("⚠️ Graph expansion query timed out, returning empty results")
                return []
                
            current_app.logger.info(f"⏱️ Graph expansion query completed in {query_time:.3f}s")
            
            # OPTIMIZATION: Pre-allocate for efficiency and track stats
            valid_results = []
            invalid_count = 0
            
            max_idx = len(self.all_chunks) - 1 if self.all_chunks else -1
            
            # OPTIMIZATION: Much more efficient bulk result processing
            valid_results = []
            invalid_count = 0
            result_count = len(result.result_set) if result.result_set else 0
            
            # Pre-allocate for performance when possible
            if result_count > 0:
                valid_results = []
                
                # Process all results in one fast loop without excessive logging
                for r in result.result_set:
                    # Process index with minimal type checking
                    idx = int(r[0]) if r[0] is not None else -1
                    original_id = r[3] if len(r) > 3 and r[3] is not None else str(idx)
                    
                    # Determine index to use quickly
                    if isinstance(original_id, int) or (isinstance(original_id, str) and original_id.isdigit()):
                        try:
                            idx_to_use = int(original_id)
                            # Fast validity check
                            if max_idx == -1 or 0 <= idx_to_use <= max_idx:
                                valid_results.append({
                                    "idx": idx_to_use,
                                    "text": r[1] or "",
                                    "source": r[2] or "No Source Available"
                                })
                            else:
                                invalid_count += 1
                        except (ValueError, TypeError):
                            # String that looked like a number but wasn't
                            valid_results.append({
                                "idx": original_id,
                                "text": r[1] or "",
                                "source": r[2] or "No Source Available"
                            })
                    else:
                        # Just use the string as is
                        valid_results.append({
                            "idx": original_id,
                            "text": r[1] or "",
                            "source": r[2] or "No Source Available"
                        })
            
            # Only log if there were actually invalid nodes to report
            if invalid_count > 0:
                current_app.logger.warning(f"⚠️ Skipped {invalid_count} invalid nodes from graph results")
            
            # Calculate percentage only if needed and with safety check
            if result_count > 0:
                valid_percent = len(valid_results) / result_count * 100
                current_app.logger.info(f"📊 Graph expansion: {len(valid_results)}/{result_count} valid results ({valid_percent:.1f}%)")
            
            total_time = time.time() - start_time
            current_app.logger.info(f"⏱️ Total graph expansion time: {total_time:.3f}s")
            
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

            # EMBED: normalized embeddings for combined graph
            self.all_embeddings = self.st_model.encode([
                chunk["text"] for chunk in self.all_chunks if chunk["text"]
            ], convert_to_numpy=True, normalize_embeddings=True)
            
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

            # EMBED: push precomputed (normalized) embeddings into Chroma
            embs = self.all_embeddings
            if embs is None:
                embs = self.st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

            self.graph_collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embs.tolist(),  # EMBED
            )
            
            self.chroma_id_map = {id_str: idx for idx, id_str in enumerate(ids)}
            
            current_app.logger.info(f"✅ Successfully stored {len(ids)} chunks in session-specific ChromaDB collection '{self.collection_name}'")
            return True
        except Exception as e:
            current_app.logger.error(f"❌ Failed to store chunks in ChromaDB: {e}", exc_info=True)
            return False

    # EMBED: accept q_emb and query with query_embeddings (no query-time embedding inside Chroma)
    def query_chroma_chunks(self, query_text, n_results=8, q_emb=None):
        """
        Query the ChromaDB collection for chunks relevant to the query.
        
        Args:
            query_text (str): The query text
            n_results (int): Maximum number of results to return
            q_emb (np.ndarray or list): precomputed normalized query embedding (optional)
            
        Returns:
            List of chunk dictionaries with text and source
        """
        start_time = time.time()
        if not hasattr(self, "graph_collection"):
            current_app.logger.warning("❌ ChromaDB collection not initialized")
            return []
            
        try:
            current_app.logger.info(f"🔍 Querying current session's ChromaDB collection '{self.collection_name}' with '{query_text[:50]}...' for {n_results} results")
            
            available_chunks = len(self.all_chunks) if self.all_chunks else 100
            actual_n_results = min(n_results, available_chunks)

            # OPTIMIZATION: Use provided embedding if available to avoid recomputation
            if q_emb is None:
                embed_start = time.time()
                q_emb = self.st_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
                current_app.logger.info(f"⏱️ Query embedding computed in {time.time() - embed_start:.3f}s")
            else:
                current_app.logger.info(f"✅ Using pre-computed query embedding")
            
            # use query_embeddings to avoid query-time embedding in Chroma
            query_start = time.time()
            results = self.graph_collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=actual_n_results
            )
            
            chunks = []
            if results and len(results["ids"]) > 0:
                num_results = len(results["ids"][0])
                current_app.logger.info(f"✅ Current session's ChromaDB collection returned {num_results} results")
                
                # Track metadata
                sources_counter = Counter()
                clusters_counter = Counter()
                
                for i in range(num_results):
                    distance = results["distances"][0][i] if "distances" in results else None
                    source = results["metadatas"][0][i].get("source", "No Source Available")
                    cluster_id = results["metadatas"][0][i].get("cluster_id", -1)
                    
                    # OPTIMIZATION: Track metadata during iteration to avoid additional loops
                    sources_counter[source] += 1
                    clusters_counter[cluster_id] += 1
                    
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
                    
            else:
                current_app.logger.warning("⚠️ ChromaDB query returned no results")
            
            return chunks
        except Exception as e:
            current_app.logger.error(f"❌ ChromaDB query failed: {e}", exc_info=True)
            return []

    def query_graph_link_response(self, query: str) -> dict:

        start_time = time.time()
        current_app.logger.info(f"📝 [START] Processing query: {query!r}")

        if not self.all_chunks:
            current_app.logger.warning("❌ No chunks available for retrieval")
            return {"answer": "I don't have enough information.", "source": None}

        current_app.logger.info(f"📊 Total chunks in memory: {len(self.all_chunks)}")

        # Initialize containers
        all_relevant_chunks = []
        graph_chunks = []
        semantic_chunks = []
        source_files = set()
        
        # Initialize counter variables to prevent UnboundLocalError
        graph_chunk_count = 0
        semantic_chunk_count = 0
        used_sources = []
        
        # Initialize timing variables to prevent UnboundLocalError
        step3_start = time.time()
        
        # OPTIMIZATION: Compute query embedding only once and reuse for all operations
        embed_start = time.time()
        query_embedding = self.st_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        current_app.logger.info(f"⏱️ Query embedding computed in {time.time() - embed_start:.3f}s")
        

        is_complex_query = None  # Will be determined by knee detection
        complexity_score = 0.5   # Neutral starting point
        
        def detect_knee_point(scores, min_k=5):
           
            if len(scores) <= min_k:
                current_app.logger.info(f"📊 Too few scores ({len(scores)}), using all available chunks")
                return len(scores), 1.0  # If few scores, use all and mark as complex
            
            # Use more of the curve for short lists, less for long lists
            cutoff = min(len(scores), max(30, len(scores) // 3))
            
            # Focus on the most relevant portion of the curve
            working_scores = scores[:cutoff]
            
            # Create a straight line from first to last point in our working set
            x = np.arange(len(working_scores))
            first, last = working_scores[0], working_scores[-1]
            line = first - (first - last) * (x / (len(working_scores) - 1))
            

            distances = working_scores - line
            

            window_size = max(2, min(5, len(distances) // 10))
            if window_size > 1 and len(distances) > window_size*2:
                smoothed_distances = np.convolve(distances, np.ones(window_size)/window_size, mode='valid')
                # Pad the start to maintain array size
                pad = np.zeros(window_size - 1)
                smoothed_distances = np.concatenate((pad, smoothed_distances))
            else:
                smoothed_distances = distances
            
            # Find the knee point (maximum distance)
            knee_idx = np.argmax(smoothed_distances)
            
            # Ensure we take at least min_k points 
            knee_idx = max(knee_idx, min_k)
            
            if knee_idx < 3 and len(scores) > 10 and scores[0] > 0.8:
                first_gap = scores[0] - scores[5]  
                if first_gap < 0.1:  
                    knee_idx = max(knee_idx, min(10, len(scores) // 5))
                    current_app.logger.info(f"📊 Extended knee point due to consistently high relevance")
            
            # Calculate signal strength ratio: how much stronger is the top result vs average?
            signal_strength = scores[0] / (np.mean(scores[:cutoff]) + 0.0001)
            
            # Calculate the slope at the knee point
            if knee_idx > 0 and knee_idx < len(working_scores) - 1:
                slope = abs(working_scores[knee_idx+1] - working_scores[knee_idx-1]) / 2
            else:
                slope = 0
            
            # Quantify how sharp the knee is (sharpness = complexity)
            knee_sharpness = min(1.0, slope * 10)
            
            # Calculate knee position factor (earlier = more complex query)
            position_factor = 1 - (knee_idx / len(working_scores))
            

            score_std = np.std(scores[:cutoff])
            std_factor = min(1.0, score_std * 10)
            
            # Combine multiple signals to determine overall complexity
            complexity = (0.4 * position_factor) + (0.3 * knee_sharpness) + (0.3 * std_factor)
            
            
            return knee_idx, complexity
        
        current_app.logger.info(f"📝 Will determine query complexity dynamically using knee detection")
        
        # STEP 1: Vector-based retrieval with ChromaDB
        step1_start = time.time()
        current_app.logger.info("🔍 [STEP 1] Retrieving semantically similar chunks from ChromaDB")
        
        # Dynamic retrieval count based on dataset size and distribution
        total_available = len(self.all_chunks) if self.all_chunks else 100
        # Square root scaling with reasonable bounds
        retrieval_base = max(20, min(100, int(np.sqrt(total_available) * 5)))
        max_chroma_results = min(total_available, retrieval_base)
        
        current_app.logger.info(f"📊 Retrieving up to {max_chroma_results} chunks from ChromaDB")
            
        # Pass the pre-computed embedding to avoid recalculation
        chroma_chunks = self.query_chroma_chunks(query, n_results=max_chroma_results, q_emb=query_embedding)
        
        if chroma_chunks:
            current_app.logger.info(f"✅ Found {len(chroma_chunks)} relevant chunks from ChromaDB")
            for chunk in chroma_chunks:
                chunk["retrieval_type"] = "semantic"
                # Track sources during iteration to avoid additional loops later
                if chunk.get('source'):
                    source_files.add(chunk.get('source'))
            semantic_chunks.extend(chroma_chunks)
        else:
            current_app.logger.warning("⚠️ No chunks found in ChromaDB")
        
        current_app.logger.info(f"⏱️ Step 1 (ChromaDB retrieval) completed in {time.time() - step1_start:.3f}s")
            
        # STEP 2: Graph-based retrieval
        step2_start = time.time()
        current_app.logger.info("🔍 [STEP 2] Performing graph-based retrieval")
        self._ensure_all_chunks_are_dicts()


        sim_start = time.time()
        sims = cosine_similarity([query_embedding], self.all_embeddings).flatten()
        current_app.logger.info(f"⏱️ Similarity calculation completed in {time.time() - sim_start:.3f}s")
        
        total_chunks = len(self.all_chunks)
        

        base_seed_count = max(3, min(20, int(np.sqrt(total_chunks))))
        current_app.logger.info(f"📊 Base seed count from dataset size: {base_seed_count}")
        
        # Sort scores for analysis
        sorted_scores = np.sort(sims)[::-1]  # Sort in descending order
        
        # Determine cutoff for analysis (avoid analyzing the long tail)
        analysis_cutoff = min(total_chunks, max(20, int(total_chunks * 0.2)))
        
        # Calculate score gaps and rate of decline
        if len(sorted_scores) > 5:
            # Look at the gap between top scores to understand relevance distribution
            top_score = sorted_scores[0]
            percentile_50_score = sorted_scores[min(len(sorted_scores)-1, analysis_cutoff // 2)]
            percentile_80_score = sorted_scores[min(len(sorted_scores)-1, int(analysis_cutoff * 0.8))]
            
            # Calculate drop-off rates
            top_gap = top_score - sorted_scores[min(5, len(sorted_scores)-1)]
            overall_drop = top_score - percentile_50_score
            

            seed_factor = 1.0 - min(0.7, top_gap * 3)  # Higher gap = lower factor
            
            # Calculate dynamic seed count
            dynamic_seeds = int(base_seed_count * (1.0 + seed_factor))
            
            current_app.logger.info(f"""
            📊 Seed selection analysis:
            - Top score: {top_score:.4f}
            - 50th percentile: {percentile_50_score:.4f}
            - Top-5 gap: {top_gap:.4f}
            - Overall drop: {overall_drop:.4f}
            - Seed factor: {seed_factor:.2f}
            - Dynamic seed count: {dynamic_seeds}
            """)
        else:
            # For very small datasets
            dynamic_seeds = base_seed_count
        

        knee_analysis_size = min(len(sorted_scores), max(20, int(total_chunks * 0.1)))
        analysis_scores = sorted_scores[:knee_analysis_size]
        
        if len(analysis_scores) > 5:
            # Create a straight line from first to last value in our considered range
            x = np.arange(len(analysis_scores))
            first, last = analysis_scores[0], analysis_scores[-1]
            line = first - (first - last) * (x / (len(x) - 1))
            
            # Find distances from line
            distances = analysis_scores - line
            
            # Find knee point
            knee_idx = np.argmax(distances) + 1  # +1 to include the knee point
            
            # Use knee detection to refine seed count
            knee_seeds = knee_idx + int(knee_idx * 0.5)  # Add some margin to the knee
            
            current_app.logger.info(f"📊 Knee detection for seeds found knee at index {knee_idx}")
            
            # Take the larger of dynamic calculation and knee detection
            k = max(dynamic_seeds, knee_seeds)
        else:
            k = dynamic_seeds
        
        # Apply reasonable bounds
        min_seeds = max(3, min(10, total_chunks // 10))
        max_seeds = min(total_chunks // 3, max(20, int(np.sqrt(total_chunks) * 5)))
        k = max(min_seeds, min(max_seeds, k))
        
        current_app.logger.info(f"📊 Final seed count: {k} (min: {min_seeds}, max: {max_seeds})")
        
        seeds = sims.argsort()[-k:][::-1].tolist()


        if len(seeds) > 0:
            seed_scores = [sims[i] for i in seeds]
            avg_seed_score = np.mean(seed_scores)

            depth = 1 if avg_seed_score > 0.7 else 2
            current_app.logger.info(f"📊 Dynamic graph depth: {depth} (avg seed score: {avg_seed_score:.4f})")
        else:
            depth = 1
        current_app.logger.info(f"🔍 Expanding graph with depth={depth} from {len(seeds)} seed nodes")
        
        # Dynamic expansion limit based on seeds and dataset
        seed_count = len(seeds)
        # Scale with diminishing returns as seed count increases
        expansion_per_seed = max(3, min(10, 30 // (1 + np.log10(max(1, seed_count)))))
        expansion_limit = min(total_chunks // 2, seed_count * expansion_per_seed)
        current_app.logger.info(f"📊 Dynamic expansion limit: {expansion_limit} ({expansion_per_seed:.1f} per seed)")
        
        expand_start = time.time()
        expanded = self._redisgraph_expand(seeds, max_depth=depth, limit=expansion_limit)
        current_app.logger.info(f"⏱️ Graph expansion completed in {time.time() - expand_start:.3f}s")
        current_app.logger.info(f"✅ Graph expansion found {len(expanded)} additional connected chunks")
        
        # Combine seed and expanded nodes
        combined_idxs = list(seeds)
        for e in expanded:
            if e["idx"] not in combined_idxs:
                combined_idxs.append(e["idx"])
        
        # SUPER OPTIMIZATION: Process graph nodes with minimal overhead
        chunks_len = len(self.all_chunks)
        current_app.logger.info(f"📊 Total unique graph nodes after expansion: {len(combined_idxs)}")
        
        # Filter valid indices once in a list comprehension
        valid_idxs = [idx for idx in combined_idxs if isinstance(idx, int) and 0 <= idx < chunks_len]
        invalid_count = len(combined_idxs) - len(valid_idxs)
        
        # Only log invalid nodes if we actually have any
        if invalid_count > 0:
            current_app.logger.info(f"⚠️ Filtered out {invalid_count} invalid node indices from graph")
        
        # Fast bulk processing with minimal dictionary operations and string concatenation
        graph_chunks_data = []
        total_chars = 0
        
        # Pre-allocate for even better performance
        graph_chunks = [None] * len(valid_idxs)
        
        # Process in one efficient batch
        for i, idx in enumerate(valid_idxs):
            chunk_data = self.all_chunks[idx]
            text = chunk_data.get("text", "")
            source = chunk_data.get("source", "No Source Available")
            
            # Create chunk dict directly without intermediate variables
            graph_chunks[i] = {
                "text": text,
                "source": source,
                "graph_node_idx": idx,
                "retrieval_type": "graph",
                "embedding_idx": idx  # Store index for reusing embeddings later
            }
            
            total_chars += len(text)
            
            # Track sources in the same loop
            if source:
                source_files.add(source)
                
        # Log total character count for monitoring
        current_app.logger.info(f"📊 Retrieved {len(graph_chunks)} valid graph chunks with {total_chars} total characters")
        current_app.logger.info(f"⏱️ Step 2 (Graph retrieval) completed in {time.time() - step2_start:.3f}s")
        
        # STEP 3: Merge and deduplicate chunks
        step3_start = time.time()
        current_app.logger.info("🔍 [STEP 3] Merging and deduplicating chunks")
        current_app.logger.info(f"📊 Before deduplication: {len(graph_chunks)} graph chunks ({total_chars} chars), {len(semantic_chunks)} semantic chunks")
        
        all_chunks_combined = graph_chunks + semantic_chunks
        all_relevant_chunks = deduplicate_chunks(all_chunks_combined, prioritize_graph=True)
        
        current_app.logger.info(f"📊 After deduplication: {len(all_relevant_chunks)} unique chunks")
        graph_chunk_count = len([c for c in all_relevant_chunks if c.get('retrieval_type') == 'graph'])
        semantic_chunk_count = len([c for c in all_relevant_chunks if c.get('retrieval_type') == 'semantic'])
        current_app.logger.info(f"📊 Breakdown: {graph_chunk_count} graph chunks, {semantic_chunk_count} semantic chunks")
        
        current_app.logger.info(f"⏱️ Step 3 (Deduplication) completed in {time.time() - step3_start:.3f}s")

        # STEP 4: Re-rank chunks
        step4_start = time.time()
        current_app.logger.info("🔍 [STEP 4] Re-ranking chunks based on relevance to query")
        texts = [c["text"] for c in all_relevant_chunks]
        
        if not texts:
            current_app.logger.warning("⚠️ No text content found in chunks")
            context = ""
            top_indices = []
            sims = []
        else:
            # OPTIMIZATION: Reuse pre-computed embeddings where possible
            current_app.logger.info("📊 Computing similarity scores for re-ranking")
            
            # OPTIMIZATION: Use the same embedding that was computed at the beginning
            q_emb = query_embedding
            
            # OPTIMIZATION: Avoid recomputing embeddings for graph-retrieved chunks
            chunk_embs = []
            reused_embeddings = 0
            newly_computed = 0
            
            for chunk in all_relevant_chunks:
                # If it's a graph chunk with a known embedding index, reuse from all_embeddings
                if chunk.get("retrieval_type") == "graph" and "embedding_idx" in chunk:
                    idx = chunk["embedding_idx"]
                    if 0 <= idx < len(self.all_embeddings):
                        chunk_embs.append(self.all_embeddings[idx])
                        reused_embeddings += 1
                        continue
            
            # Only compute new embeddings for chunks without pre-computed embeddings
            remaining_texts = []
            remaining_indices = []
            
            for i, chunk in enumerate(all_relevant_chunks):
                if not (chunk.get("retrieval_type") == "graph" and 
                        "embedding_idx" in chunk and 
                        0 <= chunk["embedding_idx"] < len(self.all_embeddings)):
                    remaining_texts.append(chunk["text"])
                    remaining_indices.append(i)
            
            if remaining_texts:
                current_app.logger.info(f"📊 Computing {len(remaining_texts)} new embeddings for remaining chunks")
                new_embeddings = self.st_model.encode(remaining_texts, convert_to_numpy=True, normalize_embeddings=True)
                newly_computed = len(remaining_texts)
                
                # Insert new embeddings at the correct positions
                final_chunk_embs = [None] * len(all_relevant_chunks)
                
                # First, place the reused embeddings
                for i, chunk in enumerate(all_relevant_chunks):
                    if chunk.get("retrieval_type") == "graph" and "embedding_idx" in chunk:
                        idx = chunk["embedding_idx"]
                        if 0 <= idx < len(self.all_embeddings):
                            final_chunk_embs[i] = self.all_embeddings[idx]
                
                # Then, insert the newly computed embeddings
                for i, orig_idx in enumerate(remaining_indices):
                    final_chunk_embs[orig_idx] = new_embeddings[i]
                
                chunk_embs = np.array(final_chunk_embs)
            else:
                # All embeddings were reused
                chunk_embs = np.array(chunk_embs)
            
            current_app.logger.info(f"📊 Reused {reused_embeddings} embeddings, computed {newly_computed} new ones")
            
            # Now compute similarities with the query
            sims = cosine_similarity([q_emb], chunk_embs).flatten()

            available_chunks = len(all_relevant_chunks)
            
            # Sort scores in descending order
            sorted_indices = np.argsort(sims)[::-1]
            sorted_scores = sims[sorted_indices]
            
            if len(sorted_scores) > 5:
                top_mean = np.mean(sorted_scores[:5])
                overall_mean = np.mean(sorted_scores[:min(30, len(sorted_scores))])
                score_gap = top_mean - overall_mean
                
                min_k = max(5, min(25, int(15 * (1.0 - min(0.9, score_gap * 5)))))
                current_app.logger.info(f"📊 Dynamic minimum chunks: {min_k} (score gap: {score_gap:.4f})")
            else:
                # For very small result sets
                min_k = 5
            
            # Apply knee detection algorithm
            knee_idx, complexity_score = detect_knee_point(sorted_scores, min_k=min_k)
            
            max_factor = 0.3 + (complexity_score * 0.4)  # Range from 30% to 70% based on complexity
            max_allowed = min(available_chunks, max(25, int(available_chunks * max_factor)))
            
            # Determine final chunk count using knee index with reasonable bounds
            top_k = min(max_allowed, max(min_k, knee_idx))
            
            # Use complexity score to determine if query is complex
            is_complex_query = complexity_score > 0.5
            
            # Select top chunks based on calculated k
            top_indices = sorted_indices[:top_k]
            
            current_app.logger.info(f"📊 Pure data-driven selection: {top_k} chunks (complexity score: {complexity_score:.2f})")
            current_app.logger.info(f"📊 Selected {len(top_indices)}/{available_chunks} chunks for context")

            # OPTIMIZATION: Build context with minimal overhead
            current_app.logger.info("📝 Building context for LLM prompt")
            
            # Use dictionaries for fast lookups
            context_by_source = {}
            top_sources_by_similarity = {}
            
            # Compile regex patterns once instead of repeatedly
            import re
            url_pattern = re.compile(r"^https?://")
            number_pattern = re.compile(r'\d+')
            
            # Check if we need special ID handling once instead of per chunk
            needs_id_info = any(pattern in query.lower() for pattern in ['id', 'number', 'amount', 'date', 'order'])
            
            # Process chunks in a single pass with minimal string operations
            for i in top_indices:
                chunk = all_relevant_chunks[i]
                chunk_text = chunk.get('text', '')
                src = chunk.get("source", "Unknown Source")
                sim_score = float(sims[i])
                
                # Initialize source entry if needed
                if src not in context_by_source:
                    context_by_source[src] = []
                
                # Update max similarity tracking
                if src not in top_sources_by_similarity or sim_score > top_sources_by_similarity[src]:
                    top_sources_by_similarity[src] = sim_score
                
                # Only do extra document ID processing if needed based on query
                if needs_id_info:
                    filename_base = os.path.splitext(os.path.basename(src))[0]
                    numbers_in_filename = number_pattern.findall(filename_base)
                    
                    if numbers_in_filename:
                        chunk_text = f"[DOCUMENT: {filename_base} - Contains: {', '.join(numbers_in_filename)}]\n{chunk_text}"
                
                # Format content with minimal branching - handle URLs vs files differently
                if url_pattern.match(src):
                    # For URLs, use the URL directly with domain name for better context
                    from urllib.parse import urlparse
                    parsed_url = urlparse(src)
                    domain = parsed_url.netloc
                    src_display = f"{domain}{parsed_url.path}"
                    current_app.logger.info(f"📄 Using URL source in context: {src}")
                else:
                    # For files, just use the filename
                    src_display = os.path.basename(src)
                    current_app.logger.info(f"📄 Using file source in context: {src_display}")
                
                formatted_content = f"**Source Document:** {src_display}\n**Content:** {chunk_text}"
                
                # Store context with all needed metadata
                context_by_source[src].append({
                    "text": formatted_content,
                    "similarity": sim_score,
                    "idx": i
                })
                
                # Track source files in one pass
                source_files.add(src)
            
            # Sort and flatten in one go for better performance
            context_units = []
            
            # First sort each source's chunks by similarity
            for src, chunks in context_by_source.items():
                # Sort in place to avoid creating new lists
                chunks.sort(key=lambda x: x["similarity"], reverse=True)
                # Extract text directly to final list
                for chunk in chunks:
                    context_units.append(chunk["text"])

            context = "\n---\n".join(context_units)
            
            # Get the top 3 sources that contributed to context, based on highest similarity score
            used_sources = sorted(top_sources_by_similarity.keys(), key=lambda s: top_sources_by_similarity[s], reverse=True)
            top_context_sources = used_sources[:3]
            
            # Log the sources used for context, with statistics
            current_app.logger.info(f"📊 Sources contributing to context (top {len(top_context_sources)} of {len(used_sources)}):")
            for src in top_context_sources:
                current_app.logger.info(f"📄 Source: {src} | Max similarity: {top_sources_by_similarity[src]:.4f}")
                
            current_app.logger.info(f"📊 Final context built with {len(context_units)} chunks from {len(context_by_source)} sources")
        
        current_app.logger.info(f"⏱️ Step 4 (Re-ranking) completed in {time.time() - step4_start:.3f}s")
        
        # OPTIMIZATION: Track top sources more efficiently
        top_sources = []
        seen_sources = set()
        source_reasons = {}
        
        if len(top_indices) > 0:
            # Sort sources by highest similarity score
            source_max_sim = {}
            source_best_chunk = {}
            
            for i in top_indices:
                chunk = all_relevant_chunks[i]
                src = chunk.get("source", "")
                
                if not src:
                    continue
                    
                if src not in source_max_sim or sims[i] > source_max_sim[src]:
                    source_max_sim[src] = float(sims[i])
                    source_best_chunk[src] = {
                        "picked_text": chunk["text"],
                        "similarity_score": float(sims[i]),
                        "retrieval_type": chunk.get("retrieval_type", "unknown"),
                        "query": query
                    }
            
            # Sort sources by similarity score
            sorted_sources = sorted(source_max_sim.keys(), key=lambda s: source_max_sim[s], reverse=True)
            top_sources = sorted_sources
            
            # Log reasons for top sources
            for src in sorted_sources[:3]: 
                source_reasons[src] = source_best_chunk[src]
                current_app.logger.info(f"🔎 Source picked: {src} | Similarity: {source_max_sim[src]:.4f}")

        unique_sources = len(top_sources)

        # STEP 5: Build prompt and generate response
        step5_start = time.time()
        current_app.logger.info("🔍 [STEP 5] Building LLM prompt with context")
        
        prompt = (
    "ROLE:\n"
    "You are an expert document analyst. Use ONLY the provided CONTEXT. "
    "If the query cannot be answered from CONTEXT, output exactly: [NO_ANSWER].\n\n"

    f"CONTEXT (from {unique_sources} document sources):\n"
    f"{context}\n\n"

    f"USER QUERY:\n{query}\n\n"

    "OUTPUT CONTRACT (FOLLOW ALL):\n"
    "A) Truth & Scope\n"
    "- Never invent facts. If unsupported → [NO_ANSWER]. If partially supported, state only what is supported.\n\n"

    "B) Substance & Length\n"
    "- Default to 2–4 short paragraphs (~150–300 words total) when explanation is needed.\n"
    "- After paragraphs, include a compact bullet list of 3–7 items for steps/fields/takeaways when relevant.\n\n"

    "C) Layout & Spacing (HARD RULES)\n"
    "- Paragraphs: each separated by exactly ONE blank line (i.e., a single '\\n\\n').\n"
    "- Bullets: each bullet on its own line, prefixed by '• ' (U+2022 + space). No numbering unless present in CONTEXT.\n"
    "- Bullet blocks: one blank line BEFORE the bullet block and one AFTER it.\n"
    "- Sub-bullets (only if needed): prefix with '– ' and indent with a single space after the main bullet line.\n"
    "- No trailing spaces; no double blank lines anywhere.\n"
    "- If an HTML table is emitted, put ONE blank line before it and ONE blank line after it.\n\n"

    "D) Table Decision (MANDATORY HEURISTICS)\n"
    "- Emit an HTML table fragment when comparing ≥2 items, listing many similar items with shared fields, "
    "showing metrics/specs/timelines/schedules, or when the user asks list/compare/overview/top N/differences/fields/schema.\n"
    "- If a table is emitted, ALSO include a 1–2 sentence summary paragraph adjacent to it (above or below).\n\n"

    "E) HTML TABLE RULES (apply ONLY when emitting a table)\n"
    "- Output a fragment ONLY (no <!DOCTYPE>, <html>, <head>, <body>, <style>, or code fences).\n"
    "- Structure EXACTLY:\n"
    "  <table>\n"
    "    <thead><tr><th>…</th></tr></thead>\n"
    "    <tbody><tr><td>…</td></tr></tbody>\n"
    "  </table>\n"
    "- One <th> per column; one <td> per cell; keep columns ≤ 8 when possible.\n"
    "- Mark numeric cells with class='numeric'.\n"
    "- If >25 rows, show the most relevant 10–25 and say more exist.\n\n"

    "F) Style\n"
    "- Plain language. Keep jargon only if present in CONTEXT. Prefer concrete specifics over vague phrasing.\n\n"

    "G) Final Self-Check BEFORE sending\n"
    "- If unsupported → [NO_ANSWER].\n"
    "- If a table is present, confirm the exact fragment structure and numeric class usage.\n"
    "- Confirm paragraphs/bullets/table are separated with EXACT spacing rules above. No code fences; no visible citations.\n\n"

    "H) Source Tracking (REQUIRED, hidden)\n"
    '- Append EXACTLY at the very end:\n'
    '  <div class=\"sources-used\" style=\"display:none\">[comma-separated filenames/URLs used, in importance order]</div>\n'
)

        try:
            import re
            llm_start = time.time()
            current_app.logger.info("🔍 [STEP 6] Generating response with LLM")
            current_app.logger.info(f"📝 Sending prompt with {len(prompt)} characters to LLM")
            
            raw_response = self.llm.model.invoke(prompt)
            llm_time = time.time() - llm_start
            
            response_text = str(raw_response.content).strip()
            current_app.logger.info(f"✅ LLM generated a response of {len(response_text)} characters in {llm_time:.3f}s")

            if "[NO_ANSWER]" in response_text:
                current_app.logger.warning("⚠️ LLM indicated insufficient information")
                return {"answer": "I don't have enough information.", "source": None}
                
            sources_used = []
            # Compile regex once for better performance
            sources_pattern = re.compile(r'<div class="sources-used" style="display:none">(.*?)</div>', re.IGNORECASE | re.DOTALL)
            sources_match = sources_pattern.search(response_text)
            
            if sources_match:
                # Extract the sources from the hidden div
                sources_text = sources_match.group(1).strip()
                # Split by comma and clean up each source in one comprehension
                raw_sources = [s.strip() for s in sources_text.split(',') if s.strip()]
                
                # Create a lookup dictionary for faster source matching
                filename_to_source = {os.path.basename(full_src): full_src for full_src in used_sources}
                
                # Check for URLs in used_sources for direct mapping
                import re
                url_pattern = re.compile(r"^https?://")
                
                for raw_src in raw_sources:
                    # Remove HTML tags if present in source
                    clean_src = re.sub(r'<.*?>', '', raw_src).strip()
                    
                    # Check if this source is a URL first - preserve complete URL
                    if url_pattern.match(clean_src):
                        current_app.logger.info(f"📄 Extracted URL source from LLM: {clean_src}")
                        sources_used.append(clean_src)
                        continue
                        
                    # Check if this might be a URL with the http/https prefix removed
                    if '.' in clean_src and '/' in clean_src and not ' ' in clean_src:
                        if not clean_src.startswith('http'):
                            possible_url = f"https://{clean_src}"
                            if url_pattern.match(possible_url):
                                current_app.logger.info(f"📄 Added prefix to URL source: {possible_url}")
                                sources_used.append(possible_url)
                                continue
                    
                    # For filenames, try to match to full source paths - exact matches only, no fuzzy matching
                    if raw_src in filename_to_source:
                        full_path = filename_to_source[raw_src]
                        # If the full path is a URL, keep it as is
                        if url_pattern.match(full_path):
                            current_app.logger.info(f"📄 Mapped LLM source to URL: {full_path}")
                            sources_used.append(full_path)
                        else:
                            current_app.logger.info(f"📄 Mapped LLM source to file: {raw_src}")
                            sources_used.append(full_path)
                        continue

                    # Fallback: just use the raw source as provided by LLM
                    sources_used.append(clean_src)
                
                # Remove the hidden div from the response - do this only once
                response_text = sources_pattern.sub('', response_text).strip()
                current_app.logger.info(f"🔍 Extracted {len(sources_used)} sources actually used by the LLM")
            else:
                current_app.logger.info("⚠️ No source information provided by LLM, using context sources")
                sources_used = top_context_sources
            
            sources_used = sources_used[:3]
            
            if "Sources:" in response_text or "Sources Used:" in response_text:
                response_text = re.sub(r"(?:Sources|Sources Used):\s*.*?(?:\n|$)", "", response_text, flags=re.IGNORECASE|re.DOTALL).strip()
                current_app.logger.info("🔄 Removed explicit source citations from LLM response")
                
                
            # Check for HTML tables (routes.py will handle the formatting)
            if "<table" in response_text:
                table_count = response_text.count("<table")
                current_app.logger.info(f"📊 Response contains {table_count} HTML tables")

            # Set primary source to the most relevant source that actually contributed chunks
            # Don't modify the source - keep it exactly as it was provided
            primary_source = sources_used[0] if sources_used else None
            if primary_source:
                # Don't extract just the filename - keep the full URL or path
                current_app.logger.info(f"📄 Primary source set to: {primary_source} (highest relevance in context)")

            # Collect metrics once with optimized counting
            graph_chunk_count = sum(1 for c in all_relevant_chunks if c.get("retrieval_type") == "graph")
            semantic_chunk_count = len(all_relevant_chunks) - graph_chunk_count  
            
            total_time = time.time() - start_time
            
            # Only log essential performance metrics (reduce logging overhead)
            current_app.logger.info(f"⏱️ Total processing time: {total_time:.3f}s, LLM: {llm_time:.3f}s")

            # Use the sources that were actually used by the LLM (already limited to top 3)
            final_sources = sources_used
            
                # Process sources to handle URLs and files differently
            processed_sources = []
            import re
            url_pattern = re.compile(r"^https?://")
            
            for src in final_sources:
                if not src:
                    continue
                
                # Clean any HTML that might be in the source
                clean_src = re.sub(r'<.*?>', '', str(src)).strip()
                
                # Check if this source is a URL - preserve complete URL
                if url_pattern.match(clean_src):
                    current_app.logger.info(f"📄 Using URL source directly: {clean_src}")
                    processed_sources.append(clean_src)
                # Check if this is a file ending with .html or .htm - likely a URL source
                elif clean_src.endswith(('.html', '.htm')):
                    # Check if we have a matching URL source in our context
                    found_url = None
                    for context_src in top_context_sources:
                        if clean_src in context_src and url_pattern.match(context_src):
                            found_url = context_src
                            break
                    
                    # If we found a matching URL, use it directly
                    if found_url:
                        current_app.logger.info(f"📄 Mapped HTML file to URL source: {found_url}")
                        processed_sources.append(found_url)
                    else:
                        # Try to rebuild the URL from the original source
                        base_url = "https://www.solix.com/solutions/"
                        possible_url = f"{base_url}{clean_src.replace('.html', '')}"
                        current_app.logger.info(f"📄 Constructed URL for HTML file: {possible_url}")
                        processed_sources.append(possible_url)
                # Check for URL-like structures that might be missing the http prefix
                elif ('.' in clean_src and '/' in clean_src and not ' ' in clean_src):
                    possible_url = f"https://{clean_src}" if not clean_src.startswith('http') else clean_src
                    current_app.logger.info(f"📄 Treating as URL: {possible_url}")
                    processed_sources.append(possible_url)
                else:
                    # For files, extract just the filename for proper download handling
                    filename = os.path.basename(clean_src) if os.path.sep in clean_src else clean_src
                    current_app.logger.info(f"📄 Using file source in response: {filename}")
                    processed_sources.append(filename)            # Final cleanup of the response (minimal processing)
            response_text = response_text.strip()
                
            return {
                "answer": response_text,
                "sources": processed_sources
            }

        except Exception as e:
            current_app.logger.error(f"❌ LLM call failed: {str(e)}", exc_info=True)
            total_time = time.time() - start_time
            current_app.logger.error(f"⏱️ Failed execution total time: {total_time:.3f}s")
            return {"answer": "There was an error while generating the response.", "sources": []}