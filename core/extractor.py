import PyPDF2
import google.generativeai as genai
import json
import os

# PUT YOUR REAL API KEY HERE INSIDE THE QUOTES
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=API_KEY)

def extract_graph_from_pdf(file_path: str):
    """Reads a PDF and uses Gemini to extract a JSON knowledge graph."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            # Read first 5 pages to keep API calls fast and cheap
            for page in reader.pages[:5]: 
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"PDF Error: {e}")
        return None

    prompt = f"""
    Analyze the following technical text and extract a knowledge graph.
    Return ONLY valid JSON with two keys: "nodes" and "links". Do not use markdown blocks.
    
    Nodes should have: "id" (string), "label" (string), "group" (string: concept, material, process, discipline, product).
    Links should have: "source" (node id), "target" (node id), "relationship" (string).
    Keep it to the most important 5 to 10 entities.

    Text:
    {text[:8000]}
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        # Clean the response to ensure it's pure JSON
        result_text = response.text.replace('```json', '').replace('```', '').strip()
        graph_data = json.loads(result_text)
        return graph_data
    except Exception as e:
        print(f"LLM Error: {e}")
        return None