from neo4j import GraphDatabase

class Neo4jAdapter:
    def __init__(self, uri="neo4j://localhost:7689", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_1hop_neighbors(self, entity_id, limit=50):
        """
        Lấy 1-hop lân cận của một entity_id.
        """
        # Lưu ý: r.type lấy ra ID của relation (P-ID) đã được lữu trữ thay vì Tên lable Neo4j
        query = """
        MATCH (h:Entity {id: $entity_id})-[r]->(t:Entity)
        RETURN h.id AS head, r.type AS relation, t.id AS tail
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, entity_id=entity_id, limit=limit)
                return [{"head": record["head"], "relation": record["relation"], "tail": record["tail"]} for record in result]
        except Exception as e:
            print(f"Lỗi truy vấn Neo4j: {e}")
            return []
            
    def get_multi_hop_neighbors(self, entity_id, hops=2, limit=50):
        """
        Mở rộng: Lấy multi-hop (ví dụ 2 hop) từ một điểm.
        """
        query = f"""
        MATCH p=(h:Entity {{id: $entity_id}})-[*1..{hops}]->(t:Entity)
        RETURN [rel in relationships(p) | type(rel)] as path_relations, 
               [node in nodes(p) | node.id] as path_nodes
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, entity_id=entity_id, limit=limit)
                paths = []
                for record in result:
                    paths.append({
                        "nodes": record["path_nodes"],
                        "relations": record["path_relations"]
                    })
                return paths
        except Exception as e:
            print(f"Lỗi truy vấn Neo4j (Multi-hop): {e}")
            return []

    def execute_cypher(self, query, **params):
        """
        Thực thi câu lệnh Cypher truyền vào động (tương thích Graph Isomorphism).
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, **params)
                return [record for record in result]
        except Exception as e:
            print(f"Lỗi Native Cypher: {e}")
            return []
