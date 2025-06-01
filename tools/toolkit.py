import math
import requests

def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
import requests
from xml.etree import ElementTree as ET

def pubmed_search(query: str) -> str:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = base_url + "esearch.fcgi"
    fetch_url = base_url + "efetch.fcgi"

    # Step 1: Search for PMIDs
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "xml",
        "retmax": "3"
    }
    response = requests.get(search_url, params=params)
    if response.status_code != 200:
        return "PubMed search failed."

    root = ET.fromstring(response.text)
    pmids = [id_tag.text for id_tag in root.findall(".//Id")]

    if not pmids:
        return "No articles found."

    # Optional: Step 2: Build URLs
    article_links = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" for pmid in pmids]
    result_text = "Top PubMed articles:\n" + "\n".join(article_links)
    return result_text
