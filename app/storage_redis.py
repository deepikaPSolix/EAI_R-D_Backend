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
                 session_id: str = None):
        self.redis = Redis(
            host    = host    or os.getenv("REDIS_HOST", "localhost"),
            port    = port    or int(os.getenv("REDIS_PORT", 6379)),
            password= password or os.getenv("REDIS_PASSWORD", None)
        )
        # Use session_id to create unique graph names
        if session_id:
            self.graph_name = f"semantic_graph_{session_id}"
        else:
            self.graph_name = graph_name or os.getenv("REDIS_GRAPH_NAME", "semantic_graph")
            
        self.graph = Graph(self.graph_name, self.redis)

    def store_graph(self, G_nx):
        """
        Store a NetworkX graph in RedisGraph.
        
        Args:
            G_nx: NetworkX graph object
        """
        try:
            # Create a mapping for string node IDs to integers if needed
            # This handles node IDs like "graph_id_node_idx" that can't be directly stored in RedisGraph
            node_id_map = {}
            int_id = 0
            
            # Clear any existing graph data
            query = "MATCH (n) DETACH DELETE n"
            self.graph.query(query)
            
            # Add nodes first
            for node_id, data in G_nx.nodes(data=True):
                # Convert the node ID to string for consistent handling
                node_id_str = str(node_id)
                
                # Check if the node ID is non-numeric
                try:
                    # Try to parse as integer directly
                    int(node_id)
                    # If successful, use the original ID
                    node_id_map[node_id_str] = node_id
                except (ValueError, TypeError):
                    # If not a valid integer, assign a new integer ID
                    node_id_map[node_id_str] = int_id
                    int_id += 1
                
                # Convert node properties to appropriate types
                cleaned_props = {}
                for k, v in data.items():
                    # Ensure property values are valid for RedisGraph
                    if isinstance(v, (str, int, float, bool)) and v is not None:
                        cleaned_props[k] = v
                    elif v is not None:
                        cleaned_props[k] = str(v)
                
                # Add the original ID as a property for reference
                cleaned_props['original_id'] = node_id_str
                
                # Add node properties
                props_str = ", ".join([f"{k}: ${k}" for k in cleaned_props.keys()])
                params = cleaned_props.copy()
                
                # Use the mapped integer ID for the node
                params['idx'] = node_id_map[node_id_str]
                
                # Create node in RedisGraph
                query = f"CREATE (n:Chunk {{idx: $idx, {props_str}}})"
                self.graph.query(query, params=params)
            
            # Add edges
            for source, target, data in G_nx.edges(data=True):
                source_str = str(source)
                target_str = str(target)
                
                # Get the mapped integer IDs
                source_int = node_id_map[source_str]
                target_int = node_id_map[target_str]
                
                # Create edges with the integer IDs
                weight = data.get('weight', 0.0)
                if weight is not None:
                    query = """
                    MATCH (a:Chunk), (b:Chunk)
                    WHERE a.idx = $source AND b.idx = $target
                    CREATE (a)-[:SIMILAR {weight: $weight}]->(b)
                    """
                    self.graph.query(query, params={
                        'source': source_int,
                        'target': target_int,
                        'weight': float(weight)
                    })
            
            current_app.logger.info(f"✅ Successfully stored graph in RedisGraph '{self.graph_name}' with {len(G_nx.nodes())} nodes and {len(G_nx.edges())} edges")
            return True
            
        except Exception as e:
            current_app.logger.error(f"❌ Failed to store graph in RedisGraph: {e}", exc_info=True)
            return False

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
            for graph_key in graphs_to_delete:
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
                    print(f"Failed to delete graph {graph_key}: {e}")
            
            return deleted_count
        except Exception as e:
            print(f"Error during Redis graph cleanup: {e}")
            return 0
