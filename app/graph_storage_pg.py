import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import networkx as nx

class GraphPostgresStorage:
    def __init__(self, dsn):
        self.conn = psycopg2.connect(dsn)
        self._ensure_tables_exist()

    def _ensure_tables_exist(self):
        with self.conn:
            with self.conn.cursor() as cur:
                cur.execute("SET TIME ZONE 'America/Chicago';")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS graphs (
                        id SERIAL PRIMARY KEY,
                        source_label TEXT,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS graph_nodes (
                        id SERIAL PRIMARY KEY,
                        graph_id INTEGER REFERENCES graphs(id) ON DELETE CASCADE,
                        cluster_id INTEGER,
                        node_idx INTEGER,
                        file_name TEXT,
                        url TEXT,
                        text TEXT
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS graph_edges (
                        id SERIAL PRIMARY KEY,
                        graph_id INTEGER REFERENCES graphs(id) ON DELETE CASCADE,
                        source_idx INTEGER,
                        target_idx INTEGER,
                        weight FLOAT
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS cluster_labels (
                        id SERIAL PRIMARY KEY,
                        graph_id INTEGER REFERENCES graphs(id) ON DELETE CASCADE,
                        cluster_id INTEGER,
                        label TEXT,
                        UNIQUE (graph_id, cluster_id)
                    );
                """)


    def save_graph(self, G_nx) -> int:
        from datetime import datetime
        import pytz

        central_now = datetime.now(pytz.timezone('America/Chicago'))

        def clean(s):
            return s.replace("\x00", "") if isinstance(s, str) else s

        with self.conn:
            with self.conn.cursor() as cur:
                # Insert graph metadata
                cur.execute("""
                    INSERT INTO graphs (source_label, created_at)
                    VALUES (%s, %s)
                    RETURNING id
                """, ("", central_now))
                graph_id = cur.fetchone()[0]

                node_data = []
                for idx, data in G_nx.nodes(data=True):
                    raw_text = data.get("text", "")
                    cluster_id = int(data.get("cluster", -1))
                    file_name = clean(data.get("file_name"))

                    # 🔄 Fallback: Extract file_name if embedded in text
                    if not file_name and isinstance(raw_text, dict):
                        embedded_text = clean(raw_text.get("text", ""))
                        if "||" in embedded_text:
                            file_name_candidate, _ = embedded_text.split("||", 1)
                            file_name = clean(file_name_candidate.strip())
                    elif not file_name and isinstance(raw_text, str):
                        if "||" in raw_text:
                            file_name_candidate, _ = raw_text.split("||", 1)
                            file_name = clean(file_name_candidate.strip())

                    # Clean and assign text and url
                    if isinstance(raw_text, dict):
                        text = clean(raw_text.get("text", ""))
                        url = clean(raw_text.get("url", None))
                    else:
                        text = clean(str(raw_text))
                        url = clean(data.get("url"))

                    node_data.append((
                        int(graph_id),
                        cluster_id,
                        int(idx),
                        file_name,
                        url,
                        text
                    ))

                # 🚀 Insert all nodes
                execute_values(cur, """
                    INSERT INTO graph_nodes (graph_id, cluster_id, node_idx, file_name, url, text)
                    VALUES %s
                """, node_data)

                # Insert all edges
                edge_data = [
    (
        int(graph_id),
        int(u),         # <-- cast here
        int(v),         # <-- and here
        float(data.get('weight', 1.0))
    )
    for u, v, data in G_nx.edges(data=True)]
                execute_values(cur, """
                    INSERT INTO graph_edges (graph_id, source_idx, target_idx, weight)
                    VALUES %s
                """, edge_data)

        return graph_id


    def store_graph_metadata(self, graph_id: int, source_label: str):
        with self.conn:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE graphs SET source_label = %s WHERE id = %s
                """, (source_label, graph_id))

    def list_graphs(self, limit=25):
        with self.conn:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT id, source_label, created_at
                    FROM graphs
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
                rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "label": row[1] or f"Graph {row[0]}",
                "created_at": row[2].isoformat() if row[2] else None
            }
            for row in rows
        ]

    def load_graph(self, graph_id: int):
        G = nx.Graph()

        with self.conn:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT node_idx, text, cluster_id, file_name, url
                    FROM graph_nodes
                    WHERE graph_id = %s
                """, (graph_id,))
                for row in cur.fetchall():
                    node_id, text, cluster,file_name, url= row
                    G.add_node(node_id, text=text, cluster=cluster,file_name=file_name,url=url)

                cur.execute("""
                    SELECT source_idx, target_idx, weight
                    FROM graph_edges
                    WHERE graph_id = %s
                """, (graph_id,))
                for row in cur.fetchall():
                    u, v, weight = row
                    G.add_edge(u, v, weight=weight)

        return G