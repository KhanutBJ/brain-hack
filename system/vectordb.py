import faiss

# vectors = np.random.random((n, dimension)).astype('float32')
dimension = 128    # dimensions of each vector  
k = 3  # number of nearest neighbours
encoder = ... # pre-trained encoder

class VectorDB:
    def __init__(self, dimension, k):
        self.dimension = dimension
        self.k = k
        self.index = faiss.IndexFlatL2(dimension)

    def upsert(self, new):
        new_vector = encoder.encode(new)
        self.index.add(new_vector)

    def fetch(self, query):
        query_vector = encoder.encode(query)
        distances, indices = self.index.search(query_vector, self.k)
        return distances, indices
