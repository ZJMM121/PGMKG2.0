import requests
from bs4 import BeautifulSoup
import csv
from time import sleep
from tqdm import tqdm

def search_kegg_microbes(genus_name):
    url = f"https://www.genome.jp/dbget-bin/www_bfind_sub?mode=bfind&max_hit=1000&serv=kegg&dbkey=genome&keywords={genus_name}"
    res = requests.get(url)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'html.parser')
    
    results = []
    divs = soup.find_all('div', style="width:600px")
    for div in divs:
        a_tag = div.find('a')
        kegg_id = a_tag.text.strip() if a_tag else None
        name_div = div.find('div', style="margin-left:2em")
        sub_taxa_name = name_div.text.strip() if name_div else ""
        if kegg_id:
            results.append((kegg_id, sub_taxa_name))
    return results

def get_keywords_comment(kegg_id):
    url = f"https://www.genome.jp/dbget-bin/www_bget?gn:{kegg_id}"
    res = requests.get(url)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'html.parser')
    rows = soup.find_all('tr')
    keywords = ""
    comment = ""
    for row in rows:
        th = row.find('th')
        td = row.find('td')
        if not th or not td:
            continue
        label = th.text.strip()
        value = td.text.strip()
        if label == "Keywords":
            keywords = value
        elif label == "Comment":
            comment = value
    return keywords, comment

def main():
    input_file = 'predicted_bacteria.txt'
    output_file = 'kegg_microbe_info.csv'

    with open(input_file, 'r') as f:
        genus_list = [line.strip() for line in f if line.strip()]

    all_results = []

    for genus in tqdm(genus_list, desc="Processing genus"):
        try:
            sub_taxa = search_kegg_microbes(genus)
            for kegg_id, org_name in sub_taxa:
                sleep(0.5)  
                try:
                    keywords, comment = get_keywords_comment(kegg_id)
                    all_results.append({
                        "bacteria_name": genus,
                        "sub_taxa_id": kegg_id,
                        "sub_taxa_name": org_name,
                        "keywords": keywords,
                        "comment": comment
                    })
                except Exception as e:
                    print(f"[ERROR] : {e}")
        except Exception as e:
            print(f"[ERROR]  {genus} : {e}")


    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["bacteria_name", "sub_taxa_id", "sub_taxa_name", "keywords", "comment"])
        writer.writeheader()
        writer.writerows(all_results)

if __name__ == "__main__":
    main()
