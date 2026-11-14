
import requests
import numpy as np
import faiss

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "phi3:latest"

# Curated evidence passages
DOCUMENTS = [
    {
        "id": "CDC_SYMPTOMS",
        "topic": "symptoms",
        "section": "Common symptoms",
        "text": (
            "Common symptoms of pneumonia include chest pain when "
            "breathing or coughing, altered mental status or confusion, "
            "cough, fatigue, fever or chills, nausea, vomiting or diarrhea, "
            "and shortness of breath."
        ),
        "url": "https://www.cdc.gov/pneumonia/about/"
    },
    {
        "id": "IDSA_DIAGNOSIS",
        "topic": "diagnosis",
        "section": "Adult CAP diagnostic criteria",
        "text": (
            "Adult community-acquired pneumonia diagnostic criteria: "
            "newly recognized pulmonary infiltrate or infiltrates on chest "
            "imaging AND at least one respiratory symptom AND at least one "
            "other sign or finding. Respiratory symptoms include new or "
            "increased cough, new or increased sputum production, dyspnea, "
            "or pleuritic chest pain. Other findings include abnormal lung "
            "sounds, fever of at least 100.4 F, leukocytosis or unexplained "
            "bandemia, or hypoxia below 90%. If clinical suspicion is high "
            "despite a negative chest radiograph, consider chest CT."
        ),
        "url": "https://www.idsociety.org/globalassets/idsa/practice-guidelines/community-acquired-pneumonia-in-adults/cap-clinical-pathway-final-online.pdf"
    },
    {
        "id": "IDSA_SEVERITY",
        "topic": "severity",
        "section": "Severe CAP criteria",
        "text": (
            "Severe community-acquired pneumonia is defined by one major "
            "criterion OR three or more minor criteria. Major criteria are "
            "septic shock requiring vasopressors and respiratory failure "
            "requiring mechanical ventilation. Minor criteria are "
            "respiratory rate at least 30 breaths per minute, PaO2/FiO2 "
            "ratio at most 250, multilobar infiltrates, confusion or "
            "disorientation, uremia with blood urea nitrogen at least "
            "20 mg/dL, leukopenia below 4000 cells per microliter, "
            "thrombocytopenia below 100000 cells per microliter, "
            "hypothermia below 36 C, and hypotension requiring aggressive "
            "fluid resuscitation."
        ),
        "url": "https://www.idsociety.org/globalassets/idsa/practice-guidelines/community-acquired-pneumonia-in-adults/cap-clinical-pathway-final-online.pdf"
    },
    {
        "id": "CDC_CAUSES",
        "topic": "causes",
        "section": "Causes",
        "text": (
            "Pneumonia is an infection of the lungs. It can be caused by "
            "bacteria, viruses, fungi, and parasites. Bacteria and viruses "
            "are common causes, while fungi and parasites are less common."
        ),
        "url": "https://www.cdc.gov/pneumonia/about/"
    }
]

# Known topic routing prevents confusing diagnosis and severity.
def detect_topic(query):
    q = query.lower()

    # Check specific topics before generic keywords.
    if any(x in q for x in [
        "severe", "severity", "major criteria",
        "minor criteria", "vasopressor",
        "mechanical ventilation"
    ]):
        return "severity"

    if any(x in q for x in [
        "diagnos", "infiltrate", "chest imaging",
        "cap diagnostic"
    ]):
        return "diagnosis"

    if any(x in q for x in [
        "symptom", "signs", "cough", "fever",
        "breathing", "chest pain", "fatigue"
    ]):
        return "symptoms"

    if any(x in q for x in [
        "cause", "caused", "organism",
        "bacteria", "virus", "fungi", "parasite"
    ]):
        return "causes"

    # Generic "criteria" alone is insufficient to identify a topic.
    return None




def get_embedding(text, is_query=False):
    prefix = "search_query: " if is_query else "search_document: "

    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": prefix + text
        },
        timeout=120
    )
    response.raise_for_status()

    vector = np.array(
        response.json()["embedding"],
        dtype=np.float32
    ).reshape(1, -1)

    faiss.normalize_L2(vector)
    return vector


def build_index():
    print("Building semantic evidence index...")

    vectors = [
        get_embedding(doc["text"])
        for doc in DOCUMENTS
    ]

    matrix = np.vstack(vectors)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    print(f"Indexed {len(DOCUMENTS)} evidence passages.")
    return index



def retrieve(query, index, top_k=2):
    topic = detect_topic(query)

    # Reject unrelated or ambiguous questions.
    if topic is None:
        return []

    candidates = [
        i for i, doc in enumerate(DOCUMENTS)
        if doc["topic"] == topic
    ]

    if not candidates:
        return []

    query_vector = get_embedding(query, is_query=True)

    candidate_vectors = np.vstack([
        get_embedding(DOCUMENTS[i]["text"])
        for i in candidates
    ])

    similarities = (
        candidate_vectors @ query_vector.T
    ).flatten()

    ranked = np.argsort(similarities)[::-1]

    results = []

    for position in ranked[:top_k]:
        score = float(similarities[position])

        results.append({
            **DOCUMENTS[candidates[position]],
            "score": score
        })

    # Reject weak matches.
    if not results or results[0]["score"] < 0.65:
        return []

    return results



def generate_answer(query, evidence):
    context = "\n\n".join(
        f"Source: {item['id']}\n"
        f"Section: {item['section']}\n"
        f"Evidence: {item['text']}\n"
        f"Reference: {item['url']}"
        for item in evidence
    )

    prompt = f"""
You are a medical evidence assistant.

Answer the user's question using ONLY the evidence below.

Rules:
1. Do not add facts that are absent from the evidence.
2. Do not invent medical recommendations or treatments.
3. If the evidence does not answer the question, say so.
4. Preserve important conditions, numbers and AND/OR relationships.
5. Cite the source ID used in your answer.
6. Do not present yourself as a doctor.
7. Do not diagnose an individual patient.

Evidence:
{context}

Question:
{query}

Give a concise, clear answer with a source citation.
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 250
            }
        },
        timeout=300
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def answer_query(query, index):
    print("\n" + "=" * 65)
    print("QUESTION:", query)

    evidence = retrieve(query, index)

    if not evidence:
        print("ABSTAIN: No relevant evidence found.")
        return

    best_score = evidence[0]["score"]

    # Conservative initial threshold; validate with test queries.
    if best_score < 0.35:
        print(
            f"ABSTAIN: Retrieval confidence too low "
            f"(similarity={best_score:.4f})."
        )
        return

    print("\nRETRIEVED EVIDENCE:")

    for item in evidence:
        print(
            f"\n{item['id']} | {item['section']} "
            f"| Similarity: {item['score']:.4f}"
        )
        print("Reference:", item["url"])

    print("\nGENERATED ANSWER:")
    print(generate_answer(query, evidence))


if __name__ == "__main__":
    index = build_index()

    test_queries = [
        "What are the common symptoms of pneumonia?",
        "What findings are included in adult CAP diagnostic criteria?",
        "Explain severe community acquired pneumonia criteria.",
        "What causes pneumonia?",
        "What is the treatment for a broken leg?"
    ]

    for question in test_queries:
        answer_query(question, index)
