from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

def find_most_similar_laws(question, answer, articles_df, top_k=3):
    query = question + " " + answer
    query_emb = model.encode(query, convert_to_tensor=True)
    law_embs = model.encode(articles_df["조문내용"].tolist(), convert_to_tensor=True)
    hits = util.semantic_search(query_emb, law_embs, top_k=top_k)[0]
    results = [articles_df.iloc[hit['corpus_id']] for hit in hits]
    return results

def find_most_similar_cases(question, answer, case_df, top_k=3):
    query = question + " " + answer
    query_emb = model.encode(query, convert_to_tensor=True)
    case_embs = model.encode(case_df["summary"].tolist(), convert_to_tensor=True)
    hits = util.semantic_search(query_emb, case_embs, top_k=top_k)[0]
    results = [case_df.iloc[hit['corpus_id']] for hit in hits]
    return results
