from neo4j import GraphDatabase

# JSON 数据
data = {
"entities": {},
"relationships": {}
      
}

uri = "neo4j://localhost:7687"
user = "neo4j"
password = "your_password"

class Neo4jDatabase:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def create_entity(self, entity):
        with self.driver.session() as session:
            session.write_transaction(self._create_entity, entity)
    
    @staticmethod
    def _create_entity(tx, entity):
        query = (
            "MERGE (e:Entity {name: $name, type: $type, description: $description})"
        )
        tx.run(query, name=entity["entity_name"], type=entity["entity_type"], description=entity["entity_description"])
    
    def create_relationship(self, relationship):
        with self.driver.session() as session:
            session.write_transaction(self._create_relationship, relationship)
    
    @staticmethod
    def _create_relationship(tx, relationship):
        query = (
            "MATCH (a:Entity {name: $source_entity}), (b:Entity {name: $target_entity}) "
            "MERGE (a)-[r:RELATED_TO {description: $description, strength: $strength}]->(b)"
        )
        tx.run(query, source_entity=relationship["source_entity"], target_entity=relationship["target_entity"],
               description=relationship["relationship_description"], strength=relationship["relationship_strength"])

# 初始化 Neo4j 数据库连接
neo4j_db = Neo4jDatabase(uri, user, password)

# 创建实体
for entity in data["entities"]:
    neo4j_db.create_entity(entity)

# 创建关系
for relationship in data["relationships"]:
    neo4j_db.create_relationship(relationship)

# 关闭数据库连接
neo4j_db.close()