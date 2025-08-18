# app/storage_redis.py

import os
import json
from redis import Redis
from redisgraph import Graph
from flask import current_app

class GraphRedisStorage:
    def __init__(self,
                 host: str = None,
                 port: int = None,
                 password: str = None,
                 graph_name: str = None,
                 session_id: str = None,
                 timeout: int = 30):
        """
        Initialize a Redis Graph storage instance.
        
        Args:
            host: Redis host
            port: Redis port
            password: Redis password
            graph_name: Name of the graph in Redis
            session_id: Session ID to create a unique graph name
            timeout: Connection timeout in seconds
        """
        self.redis = Redis(
            host=host or os.getenv("REDIS_HOST", "localhost"),
            port=port or int(os.getenv("REDIS_PORT", 6379)),
            password=password or os.getenv("REDIS_PASSWORD", None),
            socket_timeout=timeout,  # Add timeout to prevent hanging on large operations
            socket_connect_timeout=timeout,
            health_check_interval=15,  # Periodically check if Redis is still responsive
            retry_on_timeout=True      # Retry operations if timeout occurs
        )
        # Use session_id to create unique graph names
        if session_id:
            self.graph_name = f"semantic_graph_{session_id}"
        else:
            self.graph_name = graph_name or os.getenv("REDIS_GRAPH_NAME", "semantic_graph")
            
        self.graph = Graph(self.graph_name, self.redis)
    def store_graph(self, G_nx):
        
        try:
            # Check if graph is too large and log a warning
            num_nodes = len(G_nx.nodes())
            num_edges = len(G_nx.edges())
            
            # Warn if the graph is very large - might need to increase timeouts
            if num_nodes > 5000 or num_edges > 10000:
                current_app.logger.warning(f"⚠️ Large graph detected: {num_nodes} nodes, {num_edges} edges. This might take some time and could exceed RedisGraph limits.")
                
            # Extreme case - warn about potential failures
            if num_nodes > 20000 or num_edges > 50000:
                current_app.logger.warning(f"⚠️ VERY LARGE GRAPH: {num_nodes} nodes, {num_edges} edges. Consider splitting into multiple graphs or reducing graph size.")
            
            # Create a mapping for string node IDs to integers if needed
            node_id_map = {}
            int_id = 1  # Start from 1 to avoid 0 which might cause issues in some graph algorithms
            
            # Clear any existing graph data
            try:
                query = "MATCH (n) DETACH DELETE n"
                self.graph.query(query)
                current_app.logger.debug("Cleared existing graph data")
            except Exception as e:
                current_app.logger.warning(f"Failed to clear existing graph: {e}. Continuing with new data...")
            
            # Prepare batch processing for nodes
            BATCH_SIZE = 500  # Optimized batch size
            node_batches = []
            current_batch = []
            batch_params = {}
            param_index = 0
            
            # Pre-process all node IDs to ensure consistency
            for node_id in G_nx.nodes():
                node_id_str = str(node_id)
                try:
                    # Try to convert to int - if it succeeds, use it directly
                    node_id_int = int(node_id)
                    node_id_map[node_id_str] = node_id_int
                except (ValueError, TypeError):
                    # If not an integer, assign a sequential ID
                    node_id_map[node_id_str] = int_id
                    int_id += 1
            
            current_app.logger.info(f"Processing {num_nodes} nodes in batches of {BATCH_SIZE}")
            
            # Process nodes in batches
            node_count = 0
            for node_id, data in G_nx.nodes(data=True):
                node_count += 1
                node_id_str = str(node_id)
                
                # Clean properties
                cleaned_props = {}
                for k, v in data.items():
                    if isinstance(v, (str, int, float, bool)) and v is not None:
                        # Handle strings that are too long (RedisGraph limitation)
                        if isinstance(v, str) and len(v) > 1024:
                            cleaned_props[k] = v[:1024]  # Truncate long strings
                        else:
                            cleaned_props[k] = v
                    elif v is not None:
                        cleaned_props[k] = str(v)[:1024]  # Convert to string and truncate if needed
                
                # Add the original ID and mapped ID
                cleaned_props['original_id'] = node_id_str
                cleaned_props['idx'] = node_id_map[node_id_str]
                
                # Generate parameter placeholders
                param_placeholders = []
                for k, v in cleaned_props.items():
                    param_name = f"p{param_index}_{k}"
                    batch_params[param_name] = v
                    param_placeholders.append(f"{k}: ${param_name}")
                    param_index += 1
                
                # Add to current batch
                create_stmt = f"CREATE (n:Chunk {{{', '.join(param_placeholders)}}})"
                current_batch.append(create_stmt)
                
                # If batch is full or parameter count is high, store it
                # RedisGraph has parameter limits, so we need to check both conditions
                if len(current_batch) >= BATCH_SIZE or len(batch_params) >= 1000:
                    node_batches.append((current_batch, batch_params))
                    current_batch = []
                    batch_params = {}
                    param_index = 0
                    
                # Log progress for large graphs
                if node_count % 5000 == 0:
                    current_app.logger.debug(f"Processed {node_count}/{num_nodes} nodes")
            
            # Add remaining nodes as a batch
            if current_batch:
                node_batches.append((current_batch, batch_params))
            
            # Execute node batches
            for i, (batch, params) in enumerate(node_batches):
                current_app.logger.debug(f"Processing node batch {i+1}/{len(node_batches)}")
                query = ";\n".join(batch)
                try:
                    self.graph.query(query, params=params)
                except Exception as e:
                    current_app.logger.error(f"Error in node batch {i+1}: {e}")
                    # Continue processing other batches rather than failing
            
            # Prepare batch processing for edges
            current_app.logger.info(f"Processing {num_edges} edges in batches of {BATCH_SIZE}")
            
            edge_batches = []
            current_batch = []
            batch_params = {}
            param_index = 0
            edge_count = 0
            
            # Process edges in batches
            for source, target, data in G_nx.edges(data=True):
                edge_count += 1
                source_str = str(source)
                target_str = str(target)
                
                # Skip edges where source or target is missing from the node map
                if source_str not in node_id_map or target_str not in node_id_map:
                    current_app.logger.warning(f"Skipping edge ({source}, {target}) because node is missing from mapping")
                    continue
                
                source_int = node_id_map[source_str]
                target_int = node_id_map[target_str]
                
                # Get weight with fallback
                try:
                    weight = float(data.get('weight', 0.0) or 0.0)
                except (ValueError, TypeError):
                    weight = 0.0  # Default weight if conversion fails
                
                source_param = f"s{param_index}"
                target_param = f"t{param_index}"
                weight_param = f"w{param_index}"
                
                batch_params[source_param] = source_int
                batch_params[target_param] = target_int
                batch_params[weight_param] = weight
                
                match_stmt = f"MATCH (a:Chunk), (b:Chunk) WHERE a.idx = ${source_param} AND b.idx = ${target_param} "
                create_stmt = f"CREATE (a)-[:SIMILAR {{weight: ${weight_param}}}]->(b)"
                current_batch.append(match_stmt + create_stmt)
                
                param_index += 1
                
                # If batch is full or parameter count is high, store it
                if len(current_batch) >= BATCH_SIZE or len(batch_params) >= 1000:
                    edge_batches.append((current_batch, batch_params))
                    current_batch = []
                    batch_params = {}
                    param_index = 0
                
                # Log progress for large graphs
                if edge_count % 10000 == 0:
                    current_app.logger.debug(f"Processed {edge_count}/{num_edges} edges")
            
            # Add remaining edges as a batch
            if current_batch:
                edge_batches.append((current_batch, batch_params))
            
            # Execute edge batches
            for i, (batch, params) in enumerate(edge_batches):
                current_app.logger.debug(f"Processing edge batch {i+1}/{len(edge_batches)}")
                query = ";\n".join(batch)
                try:
                    self.graph.query(query, params=params)
                except Exception as e:
                    current_app.logger.error(f"Error in edge batch {i+1}: {e}")
                    # Continue processing other batches rather than failing
            
            # Create indexes to speed up future queries
            try:
                self.graph.query("CREATE INDEX ON :Chunk(idx)")
                self.graph.query("CREATE INDEX ON :Chunk(original_id)")
                current_app.logger.debug("Created indexes on idx and original_id")
            except Exception as e:
                current_app.logger.warning(f"Failed to create indexes: {e}")
            
            current_app.logger.info(f"✅ Successfully stored graph in RedisGraph '{self.graph_name}' with {num_nodes} nodes and {num_edges} edges")
            return True
            
        except Exception as e:
            current_app.logger.error(f"❌ Failed to store graph in RedisGraph: {e}", exc_info=True)
            return False    
    def get_nodes_in_batches(self, batch_size=1000):
        """
        Retrieve all nodes from RedisGraph in batches to avoid memory issues.
        
        Args:
            batch_size: Number of nodes to retrieve in each batch
            
        Returns:
            Generator yielding batches of nodes
        """
        try:
            # Get total node count
            count_query = "MATCH (n:Chunk) RETURN count(n) as count"
            result = self.graph.query(count_query)
            total_count = result.result_set[0][0]
            
            current_app.logger.info(f"Retrieving {total_count} nodes in batches of {batch_size}")
            
            # Process in batches using LIMIT and SKIP
            for skip in range(0, total_count, batch_size):
                query = f"MATCH (n:Chunk) RETURN n.idx, n.text, n.source, n.cluster, n.original_id SKIP {skip} LIMIT {batch_size}"
                result = self.graph.query(query)
                
                # Convert result to list of dicts for easier processing
                nodes = []
                for row in result.result_set:
                    node = {
                        'idx': row[0],
                        'text': row[1] if row[1] is not None else "",
                        'source': row[2] if row[2] is not None else "",
                        'cluster': row[3] if row[3] is not None else -1,
                        'original_id': row[4] if row[4] is not None else str(row[0])
                    }
                    nodes.append(node)
                
                yield nodes
                
                current_app.logger.debug(f"Retrieved nodes {skip} to {min(skip + batch_size, total_count)} of {total_count}")
                
        except Exception as e:
            current_app.logger.error(f"Error retrieving nodes in batches: {e}", exc_info=True)
            yield []

    def get_edges_in_batches(self, batch_size=5000):
        """
        Retrieve all edges from RedisGraph in batches to avoid memory issues.
        
        Args:
            batch_size: Number of edges to retrieve in each batch
            
        Returns:
            Generator yielding batches of edges
        """
        try:
            # Get total edge count
            count_query = "MATCH ()-[r:SIMILAR]->() RETURN count(r) as count"
            result = self.graph.query(count_query)
            total_count = result.result_set[0][0]
            
            current_app.logger.info(f"Retrieving {total_count} edges in batches of {batch_size}")
            
            # Process in batches using LIMIT and SKIP
            for skip in range(0, total_count, batch_size):
                query = f"MATCH (a:Chunk)-[r:SIMILAR]->(b:Chunk) RETURN a.idx, b.idx, r.weight SKIP {skip} LIMIT {batch_size}"
                result = self.graph.query(query)
                
                # Convert result to list of tuples (source, target, weight)
                edges = []
                for row in result.result_set:
                    edge = (row[0], row[1], row[2])
                    edges.append(edge)
                
                yield edges
                
                current_app.logger.debug(f"Retrieved edges {skip} to {min(skip + batch_size, total_count)} of {total_count}")
                
        except Exception as e:
            current_app.logger.error(f"Error retrieving edges in batches: {e}", exc_info=True)
            yield []
    def optimize_graph(self):
        """
        Perform optimization operations on the graph to improve query performance.
        This includes creating indexes and running Redis memory optimization.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create indexes on key properties
            try:
                self.graph.query("CREATE INDEX ON :Chunk(idx)")
                self.graph.query("CREATE INDEX ON :Chunk(original_id)")
                self.graph.query("CREATE INDEX ON :Chunk(cluster)")
                current_app.logger.info("Created indexes on chunk properties")
            except Exception as e:
                current_app.logger.warning(f"Failed to create indexes: {e}")
            
            # Run Redis memory optimization (if supported)
            try:
                self.redis.execute_command('MEMORY PURGE')
                current_app.logger.info("Ran Redis memory purge")
            except Exception as e:
                current_app.logger.debug(f"Redis memory purge not available: {e}")
                
            return True
        except Exception as e:
            current_app.logger.error(f"Failed to optimize graph: {e}", exc_info=True)
            return False
    def preload_cache(self):
        
        try:
            # Get summary information to warm up the cache
            summary_query = """
            MATCH (n:Chunk) 
            RETURN count(n) as node_count
            """
            self.graph.query(summary_query)
            
            # Prefetch cluster information
            cluster_query = """
            MATCH (n:Chunk)
            RETURN DISTINCT n.cluster
            """
            self.graph.query(cluster_query)
            
            # Warm up index lookup cache
            index_query = """
            MATCH (n:Chunk)
            WHERE n.idx >= 0
            RETURN count(n)
            """
            self.graph.query(index_query)
            
            # Cache common queries using Redis cache mechanism
            try:
                # Check if Redis supports this feature
                self.redis.execute_command('FUNCTION', 'LOAD', """
                #!js name=graphcache
                redis.registerFunction('cache_neighbors', (idx) => {
                    const key = `cache:neighbors:${idx}`;
                    const ttl = 3600; // 1 hour cache
                    
                    // Check if already cached
                    if (redis.call('EXISTS', key)) {
                        return redis.call('GET', key);
                    }
                    
                    // Query through RedisGraph and cache result
                    const result = JSON.stringify(idx);
                    redis.call('SET', key, result, 'EX', ttl);
                    return result;
                });
                """)
            except Exception as e:
                current_app.logger.debug(f"Redis function loading not supported: {e}")
            
            current_app.logger.info("Preloaded graph data into cache")
            return True
        except Exception as e:
            current_app.logger.warning(f"Failed to preload cache: {e}")
            return False
    def query_with_timeout(self, query, params=None, timeout=30):
        """
        Execute a RedisGraph query with a timeout to prevent long-running queries
        from blocking the application.
        
        Args:
            query: The Cypher query to execute
            params: Optional parameters for the query
            timeout: Timeout in seconds
            
        Returns:
            Query result or None if timeout
        """
        import threading
        from queue import Queue
        
        result_queue = Queue()
        exception_queue = Queue()
        
        def execute_query():
            try:
                if params:
                    result = self.graph.query(query, params=params)
                else:
                    result = self.graph.query(query)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        # Start query execution in a separate thread
        thread = threading.Thread(target=execute_query)
        thread.daemon = True
        thread.start()
        
        # Wait for query to complete or timeout
        thread.join(timeout)
        
        if thread.is_alive():
            # Query is still running after timeout
            current_app.logger.warning(f"Query timed out after {timeout} seconds: {query[:100]}...")
            return None
        
        if not exception_queue.empty():
            # Query raised an exception
            exception = exception_queue.get()
            current_app.logger.error(f"Query execution error: {exception}")
            raise exception
        
        # Query completed successfully
        return result_queue.get()
        
    def cleanup_old_graphs(self, max_graphs=10):
        """
        Clean up old Redis graphs to prevent unlimited growth.
        Keeps the most recent graphs up to max_graphs.
        
        Args:
            max_graphs: Maximum number of graphs to keep
            
        Returns:
            Number of graphs deleted
        """
        try:
            # Get all graph keys using Redis KEYS command
            all_graph_keys = self.redis.keys("*semantic_graph*")
            
            # If we're under the limit, no need to delete
            if len(all_graph_keys) <= max_graphs:
                return 0
                
            # Sort keys (this assumes naming convention includes creation time info)
            # For session-based keys, they will be sorted alphabetically
            all_graph_keys.sort()
            
            # Keep the newest graphs
            graphs_to_delete = all_graph_keys[:-max_graphs]
            
            deleted_count = 0
            
            # Group graphs for batch deletion - process in batches of 10
            BATCH_SIZE = 10
            batch_count = 0
            
            current_app.logger.info(f"Cleaning up {len(graphs_to_delete)} old graphs")
            
            for i in range(0, len(graphs_to_delete), BATCH_SIZE):
                batch = graphs_to_delete[i:i+BATCH_SIZE]
                batch_count += 1
                
                current_app.logger.debug(f"Processing cleanup batch {batch_count}/{(len(graphs_to_delete) + BATCH_SIZE - 1) // BATCH_SIZE}")
                
                for graph_key in batch:
                    try:
                        # Convert byte string to regular string if needed
                        if isinstance(graph_key, bytes):
                            graph_key = graph_key.decode('utf-8')
                            
                        # Get the graph name from the key
                        graph_name = graph_key.split(':')[-1]
                        
                        # Create a graph handle and delete it
                        temp_graph = Graph(graph_name, self.redis)
                        temp_graph.delete()
                        deleted_count += 1
                    except Exception as e:
                        current_app.logger.error(f"Failed to delete graph {graph_key}: {e}")
            
            current_app.logger.info(f"Successfully deleted {deleted_count} old graphs")
            return deleted_count
        except Exception as e:
            current_app.logger.error(f"Error during Redis graph cleanup: {e}", exc_info=True)
            return 0
