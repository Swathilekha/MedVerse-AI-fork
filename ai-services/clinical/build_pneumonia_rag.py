import os
import json
import re
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = "ai-services/clinical/rag"
os.makedirs(BASE, exist_ok=True)

# Source-grounded chunks, separated by clinical topic.
documents = [
    {
        "id": "CDC_SYMPTOMS",
        "topic": "symptoms",
        "title": "CDC - About Pneumonia",
        "section": "Common symptoms",
        "keywords": "symptoms cough fever chills chest pain fatigue confusion "
                    "shortness of breath nausea vomiting diarrhea",
        "text": (
            "Common symptoms of pneumonia include chest pain when breathing "
            "or coughing, altered mental status or confusion, cough, fatigue, "
            "fever or chills, nausea, vomiting or diarrhea, and shortness "
            "of breath."
        ),
        "source": "https://www.cdc.gov/pneumonia/about/"
    },
    {
        "id": "IDSA_DIAGNOSIS",
        "topic": "diagnosis",
        "title": "IDSA - CAP Clinical Pathway",
        "section": "Adult CAP diagnostic criteria",
        "keywords": "diagnosis diagnostic criteria pulmonary infiltrate chest "
                    "imaging respiratory symptoms cough sputum dyspnea "
                    "pleuritic chest pain abnormal lung sounds fever "
                    "leukocytosis hypoxia",
        "text": (
            "Adult community-acquired pneumonia diagnostic criteria: "
            "newly recognized pulmonary infiltrate or infiltrates on chest "
            "imaging AND at least one respiratory symptom AND at least one "
            "other sign or finding. Respiratory symptoms include new or "
            "increased cough, new or increased sputum production, dyspnea, "
            "or pleuritic chest pain. Other findings include abnormal lung "
            "sounds such as rhonchi or rales, fever of at least 100.4 F, "
            "leukocytosis or unexplained bandemia, or hypoxia below 90%. "
            "If clinical suspicion is high despite a negative chest "
            "radiograph, consider chest CT."
        ),
        "source": (
            "https://www.idsociety.org/globalassets/idsa/practice-guidelines/"
            "community-acquired-pneumonia-in-adults/"
            "cap-clinical-pathway-final-online.pdf"
        )
    },
    {
        "id": "IDSA_SEVERITY",
        "topic": "severity",
        "title": "IDSA - CAP Clinical Pathway",
        "section": "Severe CAP criteria",
        "keywords": "severity severe pneumonia major minor criteria septic "
                    "shock vasopressors mechanical ventilation respiratory "
                    "failure respiratory rate oxygen pao2 fio2 multilobar "
                    "infiltrates confusion uremia leukopenia thrombocytopenia "
                    "hypothermia hypotension",
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
        "source": (
            "https://www.idsociety.org/globalassets/idsa/practice-guidelines/"
            "community-acquired-pneumonia-in-adults/"
            "cap-clinical-pathway-final-online.pdf"
        )
    },
    {
        "id": "CDC_CAUSES",
        "topic": "causes",
        "title": "CDC - About Pneumonia",
        "section": "Causes",
        "keywords": "causes bacteria viruses fungi parasites infection germs",
        "text": (
            "Pneumonia is an infection of the lungs. It can be caused by "
            "bacteria, viruses, fungi, and parasites. Bacteria and viruses "
            "are common causes, while fungi and parasites are less common."
        ),
        "source": "https://www.cdc.gov/pneumonia/about/"
    }
]

# Build TF-IDF over both passage text and topic keywords.
corpus = [
    doc["title"] + " " + doc["section"] + " " +
    doc["keywords"] + " " + doc["text"]
    for doc in documents
]

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

document_vectors = vectorizer.fit_transform(corpus)

def detect_topic(query):
    q = query.lower()

    if any(x in q for x in ["severe", "severity", "major criteria",
                            "minor criteria", "vasopressor"]):
        return "severity"

    if any(x in q for x in ["diagnos", "diagnostic", "criteria",
                            "infiltrate", "chest imaging"]):
        return "diagnosis"

    if any(x in q for x in ["symptom", "signs", "cough", "fever"]):
        return "symptoms"

    if any(x in q for x in ["cause", "caused", "organism", "bacteria",
                            "virus", "fungi"]):
        return "causes"

    return None

def retrieve(query, top_k=3, threshold=0.08):
    topic = detect_topic(query)

    q = query
    if topic:
        q += " " + topic + " " + topic

    query_vector = vectorizer.transform([q])
    scores = cosine_similarity(query_vector, document_vectors)[0]

    # Restrict candidates to the detected topic.
    candidates = [
        i for i, doc in enumerate(documents)
        if topic is None or doc["topic"] == topic
    ]

    ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked[:top_k]:
        if scores[idx] < threshold:
            continue

        doc = documents[idx]
        results.append({
            "id": doc["id"],
            "title": doc["title"],
            "section": doc["section"],
            "similarity": round(float(scores[idx]), 4),
            "passage": doc["text"],
            "source": doc["source"]
        })

    return results

# Save knowledge base and retrieval artifacts.
with open(os.path.join(BASE, "pneumonia_knowledge_base.json"),
          "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

joblib.dump(vectorizer, os.path.join(BASE, "rag_vectorizer.joblib"))
joblib.dump(document_vectors, os.path.join(BASE, "rag_document_vectors.joblib"))

if __name__ == "__main__":
    tests = [
        "What are the common symptoms of pneumonia?",
        "What findings are included in adult CAP diagnostic criteria?",
        "Explain severe community acquired pneumonia criteria.",
        "What causes pneumonia?",
        "What is the treatment for a broken leg?"
    ]

    for query in tests:
        print("\nQUERY:", query)
        results = retrieve(query)

        if not results:
            print("ABSTAIN: No sufficiently relevant evidence found.")
            continue

        for result in results:
            print("\nSource ID:", result["id"])
            print("Section:", result["section"])
            print("Similarity:", result["similarity"])
            print("Evidence:", result["passage"])
            print("Reference:", result["source"])
