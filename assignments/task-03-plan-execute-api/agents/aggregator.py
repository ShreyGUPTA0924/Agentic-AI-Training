import logging
import numpy as np
from sentence_transformers import SentenceTransformer, util
from state import PlanExecuteState

logger = logging.getLogger(__name__)

def aggregator_node(state: PlanExecuteState) -> dict:
    """Aggregator node that deduplicates retrieved document chunks using semantic similarity."""
    logger.info("Aggregator Agent invoked.")
    
    retrieved_docs = state.get("retrieved_docs", [])
    original_query = state.get("original_query", "")

    if not retrieved_docs:
        logger.warning("No retrieved documents to aggregate.")
        return {"aggregated_context": []}

    logger.info(f"Total retrieved chunks from parallel workers: {len(retrieved_docs)}")

    # 1. Exact string deduplication first
    unique_texts = []
    seen = set()
    for doc in retrieved_docs:
        cleaned = doc.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_texts.append(cleaned)

    logger.info(f"Unique chunks after exact string matching: {len(unique_texts)}")

    if len(unique_texts) <= 1:
        return {"aggregated_context": unique_texts}

    # 2. Semantic similarity deduplication (similarity threshold >= 0.95)
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode(unique_texts, convert_to_tensor=True)
        
        # Calculate cosine similarity matrix
        sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        
        keep_indices = []
        for i in range(len(unique_texts)):
            duplicate = False
            for j in keep_indices:
                if sim_matrix[i][j] >= 0.95:
                    logger.info(f"Deduplicating semantically similar chunk {i} (similarity {sim_matrix[i][j]:.4f} with chunk {j})")
                    duplicate = True
                    break
            if not duplicate:
                keep_indices.append(i)
                
        aggregated_context = [unique_texts[i] for i in keep_indices]
        logger.info(f"Final aggregated chunks after semantic deduplication: {len(aggregated_context)}")
        
        return {"aggregated_context": aggregated_context}
    except Exception as e:
        logger.error(f"Error during semantic deduplication in Aggregator: {e}", exc_info=True)
        # Fallback to exact-match unique texts
        return {"aggregated_context": unique_texts}
