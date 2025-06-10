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
        abstract = article.findtext(".//Abstract/AbstractText", "No abstract available")
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
        
        # Create a summary of the abstract (first 2-3 sentences)
        abstract_sentences = abstract.split('. ')
        summary = '. '.join(abstract_sentences[:2]) + '.'
        
        # Format the output with title, summary, and link
        results.append(f"{i+1}. {title}\n")
        results.append(f"   Summary: {summary}\n")
        results.append(f"   Link: {link}\n")

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

    results = ["**Clinical Trials:**\n"]
    for i, study in enumerate(studies):
        protocol = study.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})

        nct_id = identification.get("nctId", "N/A")
        title = identification.get("briefTitle", "N/A")
        overall_status = status.get("overallStatus", "Unknown")

        # Format each study with clear sections
        results.append(f"{i+1}. {title}")
        results.append(f"\n   Status: {overall_status}")
        results.append(f"\n   NCT ID: {nct_id}")
        results.append(f"\n   Link: https://clinicaltrials.gov/ct2/show/{nct_id}")

    return "\n".join(results)
