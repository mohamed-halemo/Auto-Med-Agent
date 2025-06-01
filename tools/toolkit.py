import math
import requests
from xml.etree import ElementTree as ET

import requests
from xml.etree import ElementTree as ET

def pubmed_search(query: str) -> str:
    # Base URL for NCBI E-utilities API
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = base_url + "esearch.fcgi"
    fetch_url = base_url + "efetch.fcgi"

    # Step 1: Search for article PMIDs
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "xml",
        "retmax": "3"
    }

    search_response = requests.get(search_url, params=search_params)
    if search_response.status_code != 200:
        return " PubMed search failed."

    search_root = ET.fromstring(search_response.text)
    pmids = [id_tag.text for id_tag in search_root.findall(".//Id")]

    if not pmids:
        return "🔍 No articles found."

    # Step 2: Fetch article details using efetch
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }

    fetch_response = requests.get(fetch_url, params=fetch_params)
    if fetch_response.status_code != 200:
        return " Failed to fetch article details."

    fetch_root = ET.fromstring(fetch_response.text)
    articles = fetch_root.findall(".//PubmedArticle")

    results = [" **Top PubMed Articles:**\n"]
    for i, article in enumerate(articles):
        pmid = article.findtext(".//PMID")
        title = article.findtext(".//ArticleTitle")
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
        results.append(f"{i+1}. [{title}]({link})")

    return "\n".join(results)


def clinical_trial_search(condition: str):
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.cond": condition,
        "pageSize": 3,
        "format": "json"
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"API error: HTTP {response.status_code}"

    data = response.json()
    studies = data.get("studies", [])
    if not studies:
        return "No studies found."

    results = []
    for study in studies:
        protocol = study.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})

        nct_id = identification.get("nctId", "N/A")
        title = identification.get("briefTitle", "N/A")
        overall_status = status.get("overallStatus", "Unknown")

        # Compose formatted result with direct link to ClinicalTrials.gov page
        results.append(f"Title: {title}\nStatus: {overall_status}\nNCT ID: {nct_id}\nhttps://clinicaltrials.gov/ct2/show/{nct_id}\n")

    return "\n".join(results)
