import os

def extract_graph_from_pdf(file_path: str):
    """
    MOCK EXTRACTOR: Simulates LLM extraction for the TA Demo to avoid API rate limits.
    Returns a predefined knowledge graph based on the uploaded filename.
    """
    filename = os.path.basename(file_path).lower()

    # --- Scenario 1: Electronics PDF ---
    if "electronic" in filename:
        return {
            "nodes": [
                {"id": "e1", "label": "Voltage Source", "group": "material"},
                {"id": "e2", "label": "Resistor", "group": "material"},
                {"id": "e3", "label": "Ohm's Law", "group": "concept"},
                {"id": "e4", "label": "Current Flow", "group": "process"},
                {"id": "e5", "label": "Circuit Diagram", "group": "product"}
            ],
            "links": [
                {"source": "e1", "target": "e4", "relationship": "drives"},
                {"source": "e2", "target": "e4", "relationship": "limits"},
                {"source": "e3", "target": "e4", "relationship": "governs"},
                {"source": "e5", "target": "e1", "relationship": "depicts"}
            ]
        }

    # --- Scenario 2: Zinc/Rasa Shastra PDF (Default Fallback) ---
    return {
        "nodes": [
            {"id": "z1", "label": "Yashada (Zinc)", "group": "material"},
            {"id": "z2", "label": "Distillation", "group": "process"},
            {"id": "z3", "label": "Rasaratna Samuccaya", "group": "concept"},
            {"id": "z4", "label": "Mushaa (Crucible)", "group": "product"},
            {"id": "z5", "label": "Purification", "group": "process"}
        ],
        "links": [
            {"source": "z1", "target": "z2", "relationship": "extracted_via"},
            {"source": "z3", "target": "z1", "relationship": "mentions"},
            {"source": "z2", "target": "z4", "relationship": "requires"},
            {"source": "z1", "target": "z5", "relationship": "undergoes"}
        ]
    }