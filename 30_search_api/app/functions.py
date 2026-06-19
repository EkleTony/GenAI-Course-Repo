import numpy as np


def return_search_result_indexes(query, df, model, dist, top_k=5):
    """ 
    Convert user query to embedding and return indexes of closest results...!
    """

    # Convert query text into embedding
    query_embedding = model.encoder([query])

    # Load stored embeddign from dataframes
    data = df.collect()
    embeddings = np.array(data["embedding"].to_list())

    # Compute distance between query and stored embeddings
    distances = dist.pairwise(query_embedding, embeddings)[0]

    # Return indexes of top-k closest results
    return np.argsort(distances)[:top_k]
