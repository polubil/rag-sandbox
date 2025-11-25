from qdrant_client import QdrantClient
from qdrant_client.models import *
from uuid import uuid4

class Store:
    
    def __init__(self, batch: int = 100):
        self.client = QdrantClient(host="localhost", port=6333)
        self.batch = batch
        self.distance_enum = {
            "dot": Distance.DOT,
            "cosine": Distance.COSINE
        }
        
    def create_collection(self, collection_name, dim=312, distance="dot"):
        if not self.client.collection_exists(collection_name):
            vector_params = VectorParams(
                size=dim, 
                distance=self.distance_enum.get(
                    distance, 
                    Distance.COSINE
                )
            )
            r = self.client.create_collection(collection_name, vector_params)
            if r:
                print(f"collection {collection_name} create")
            else: 
                print(f"Could not create collection {collection_name}")
        else:
            print(f"Collection {collection_name} already exists")
            
    def push_points(self, collection_name: str, data):
        ops = len(data)
        points = []
        for i in range(ops):
            points.append(
                PointStruct(
                    id=uuid4(),
                    vector=data[i]["new_embeddings"],
                    payload={
                        "title": data[i]["title"],
                        "chunk_idx": data[i]["chunk_idx"],
                        "text": data[i]["text"]
                    }
                )
            )
            if (i+1)%self.batch == 0 or i == ops-1:
                r = self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                points = []
                print(f"Upserted {i+1} of {ops} points.")
                print(r)
                
    def find_sim(self, vecs: list[list[float]], collection_name):
        search_params = SearchParams(
            hnsw_ef=128
        )
        r = self.client.query_points(
            collection_name=collection_name,
            query=vecs if len(vecs) > 1 else vecs[0],
            search_params=search_params
        ).points
        return r
            
    
            
    
    