from Bio import Entrez
import time
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET

Entrez.email = " "

SEARCH_KEYWORDS = '("feed efficiency" OR "growth performance" OR "pig")'
#SEARCH_KEYWORDS = '(("feed efficiency" OR "growth performance") AND "pig")'
with open("predicted_bacteria.txt", "r", encoding="utf-8") as f:
    microbe_list = [line.strip() for line in f if line.strip()]

results = []

for name in tqdm(microbe_list):
    query = f'"{name}" AND {SEARCH_KEYWORDS}'
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=3)
        record = Entrez.read(handle)
        count = int(record["Count"])
        id_list = record["IdList"]

        titles = []
        if id_list:
            fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="xml")
            fetch_data = fetch_handle.read()
            root = ET.fromstring(fetch_data)
            for article in root.findall(".//ArticleTitle"):
                titles.append(article.text.strip())
        title_str = " | ".join(titles)

        results.append((name, count, title_str))
        time.sleep(0.34) 
    except Exception as e:
        print(f"[Error] {name}: {e}")
        results.append((name, -1, "ERROR"))
df = pd.DataFrame(results, columns=["Microbe", "PubMed_Count", "Sample_Titles"])
df.to_csv("microbe_pubmed_results_all.csv", index=False)


