import numpy as np
import faiss

# vectors = np.random.random((n, dimension)).astype('float32')
# dimension = 128    # dimensions of each vector  
# k = 3  # number of nearest neighbours
# encoder = ... # pre-trained encoder

class VectorDB:
    def __init__(self, dimension, k):
        self.dimension = dimension
        self.k = k
        self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(128))  # L2 distance is used for similarity search

    def upsert(self, new, tag):
        new_vector = encoder.encode(new)
        self.index.add_with_ids(new_vector, tag)

    def fetch(self, query): 
        query_vector = encoder.encode(query)
        distances, indices = self.index.search(query_vector, k=self.k)
        closest_tag = metadata[indices.flatten()]
        return distances, indices, closest_tag


# # TEST
# Generate random vectors and associated metadata for demonstration
data = np.random.random((10000, 128)).astype('float32')
metadata = np.array([f"metadata_{i}" for i in range(10000)], dtype='str')

# Build an index with metadata
index = faiss.IndexIDMap2(faiss.IndexFlatL2(128))  # L2 distance is used for similarity search
index.add_with_ids(data, np.arange(len(metadata)))

# Query for the nearest neighbors
query_vector = np.random.random((1, 128)).astype('float32')
D, I = index.search(query_vector, k=5)

# Retrieve metadata for the nearest neighbors
nearest_neighbors_metadata = metadata[I.flatten()]

print("Distances:", D)
print("Indices:", I)
print("Nearest Neighbors Metadata:", nearest_neighbors_metadata)
