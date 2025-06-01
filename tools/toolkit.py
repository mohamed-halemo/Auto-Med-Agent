import math
import requests

# Simple calculator function to evaluate basic math expressions
def calculator(expression: str) -> str:
    try:
        # Evaluate the expression using Python's built-in eval (unsafe for untrusted input)
        result = eval(expression)
        return str(result)
    except Exception as e:
        # Return error message if evaluation fails
        return f"Error: {e}"

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
