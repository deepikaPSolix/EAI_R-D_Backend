# app/storage_redis.py

import os
import json
from redis import Redis
from redisgraph import Graph

class GraphRedisStorage:
    def __init__(self,
                 host: str = None,
                 port: int = None,
                 password: str = None,
                 graph_name: str = None):
        self.redis = Redis(
            host    = host    or os.getenv("REDIS_HOST", "localhost"),
            port    = port    or int(os.getenv("REDIS_PORT", 6379)),
            password= password or os.getenv("REDIS_PASSWORD", None)
        )
        self.graph_name = graph_name or os.getenv("REDIS_GRAPH_NAME", "semantic_graph")
        self.graph = Graph(self.graph_name, self.redis)

    def store_graph(self, G_nx):
        # 1) Delete any existing graph
        try:
            self.graph.delete()
        except:
            pass

        # 2) Recreate the Graph handle
        self.graph = Graph(self.graph_name, self.redis)

        # 3) Insert nodes one-by-one
        for idx, data in G_nx.nodes(data=True):
            # Ensure text is a string
            raw_text = data.get("text", "")
            if isinstance(raw_text, dict):
                text_val = raw_text.get("text", "")
            else:
                text_val = str(raw_text)

            props = {
                "idx":       idx,
                "text":      text_val,
                'source': data.get('source',''),
                "cluster":   int(data.get("cluster", -1))
            }

            # Build a Cypher map literal safely
            cypher_props = "{" + ", ".join(
                f"{k}: {json.dumps(v)}" for k, v in props.items()
            ) + "}"
            query = f"CREATE (n:Chunk {cypher_props})"
            self.graph.query(query)

        # 4) Insert edges one-by-one
        for u, v, d in G_nx.edges(data=True):
            weight = float(d.get("weight", 0.0))
            query = (
                f"MATCH (a:Chunk {{idx: {u}}}), (b:Chunk {{idx: {v}}}) "
                f"CREATE (a)-[:SIMILAR {{weight: {weight}}}]->(b)"
            )
            self.graph.query(query)
