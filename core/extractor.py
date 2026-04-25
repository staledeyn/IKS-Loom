import PyPDF2, json, os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
)
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

def extract_graph_from_pdf(file_path: str):
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages[:10]:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"PDF Error: {e}")
        return None

    prompt = f"""Analyze the following technical text and extract a knowledge graph.
Return ONLY valid JSON with two keys: "nodes" and "links". No markdown.

Nodes: "id" (string), "label" (string), "group" (concept|material|process|discipline|product).
Links: "source" (node id), "target" (node id), "relationship" (string).
Extract 10-20 most important entities.

Text:
{text[:12000]}"""

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"LLM Error: {e}")
        return None