import fitz, json, os, base64
from io import BytesIO
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
)
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

def pdf_pages_to_base64(file_path: str, max_pages: int = 6):
    doc = fitz.open(file_path)
    images = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(dpi=150)
        buf = BytesIO(pix.tobytes("png"))
        images.append(base64.b64encode(buf.getvalue()).decode())
    doc.close()
    return images

def extract_graph_from_pdf(file_path: str):
    try:
        images = pdf_pages_to_base64(file_path)
    except Exception as e:
        print(f"PDF Error: {e}")
        return None

    if not images:
        print("No pages extracted")
        return None

    content = [
        {
            "type": "text",
            "text": """Analyze these manuscript pages and extract a knowledge graph.
Return ONLY valid JSON with two keys: "nodes" and "links". No markdown.

Nodes: "id" (short slug, no spaces), "label" (display name), "group" (concept|material|process|discipline|product).
Links: "source" (node id), "target" (node id), "relationship" (short verb phrase).
Extract 10-20 most important entities and their relationships."""
        }
    ]

    for img_b64 in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "low"}
        })

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content.strip())
        if "nodes" not in result:
            for val in result.values():
                if isinstance(val, dict) and "nodes" in val:
                    result = val
                    break
        print(f"DEBUG extracted {len(result.get('nodes',[]))} nodes, {len(result.get('links',[]))} links")
        return result
    except Exception as e:
        print(f"LLM Error: {e}")
        return None