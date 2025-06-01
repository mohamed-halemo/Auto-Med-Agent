import math
import requests
from xml.etree import ElementTree as ET

# Function to search PubMed for articles related to a query string
def pubmed_search(query: str) -> str:
    # Base URL for NCBI E-utilities API
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = base_url + "esearch.fcgi"  # For finding article IDs (PMIDs)
    fetch_url = base_url + "efetch.fcgi"    # (Optional) For fetching full article data

    # Step 1: Search for article PMIDs using the query string
    params = {
        "db": "pubmed",       # Search the PubMed database
        "term": query,        # The search term entered by the user
        "retmode": "xml",     # Return results in XML format
        "retmax": "3"         # Return only the top 3 results
    }
    response = requests.get(search_url, params=params)
    
    # Check for request failure
    if response.status_code != 200:
        return "PubMed search failed."

    # Parse the XML response to extract PMIDs
    root = ET.fromstring(response.text)
    pmids = [id_tag.text for id_tag in root.findall(".//Id")]

    # If no PMIDs found, return appropriate message
    if not pmids:
        return "No articles found."

    # Step 2: Create direct PubMed URLs for the found articles
    article_links = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" for pmid in pmids]
    result_text = "Top PubMed articles:\n" + "\n".join(article_links)
    
    # Return formatted string with top article links
    return result_text


import requests


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
