def verify_face(test_embedding, threshold=0.5):
    embeddings_dict = load_embeddings()
    min_dist = float('inf')
    identity = None

    for name, embeddings in embeddings_dict.items():
        for stored_embedding in embeddings:
            dist = np.linalg.norm(test_embedding - stored_embedding)
            if dist < min_dist:
                min_dist = dist
                identity = name

    if min_dist < threshold:
        return f"Verified: {identity}"
    else:
        return "No match found"
