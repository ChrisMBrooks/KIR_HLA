import time, requests, re
import numpy as np
import pandas as pd

def format_hla_allele_name(allele_tuple:list):
    short_code_str = str(int(allele_tuple[1]))

    if len(short_code_str) <= 3:
        allele_tuple[1] = "0{}".format(short_code_str)
    else:
        allele_tuple[1] = short_code_str

    allele_name = "{}*{}:{}".format(
        allele_tuple[0], allele_tuple[1][:2], allele_tuple[1][2:]
    )

    return allele_name

def get_hla_allele_details(allele_name:str):
    url_template = "https://www.ebi.ac.uk/cgi-bin/ipd/api/allele?limit=10&project=HLA&query=startsWith(name,\"{}\")"
    url = url_template.format(allele_name)

    request = requests.get(url)
    
    output = {}
    
    if request.status_code == 200 and len(request.json()['data']) > 0 and \
        request.json()['data'][0]['accession'] and request.json()['data'][0]['accession'] and \
        request.json()['data'][0]['name']:

        output["ebi_id"] = request.json()['data'][0]['accession']
        output["ebi_name"] = request.json()['data'][0]['name']
    
    return output

def get_hla_prot_sequence(ebi_id:str):

    url_template = "https://www.ebi.ac.uk/Tools/dbfetch/dbfetch?db=imgthlapro;id={}&format=fasta&style=raw"
    url = url_template.format(ebi_id)

    ebi_hla_prot_request = requests.get(url)

    raw_strand = ""
    if ebi_hla_prot_request.status_code == 200:
        raw_fasta_prot_txt = ebi_hla_prot_request.text

        regex_defintion = r"(>)(\w+:)(.+\n)([ACDEFGHIKLMNPQRSTVWY\n]+)"
        uniprot_parser = re.compile(regex_defintion)
        matches = uniprot_parser.findall(raw_fasta_prot_txt)

        if len(matches) > 0:
            raw_strand=matches[0][-1].replace('\n','').strip()
    
    return raw_strand

def retrieve_hla_meta_data_from_ipd_db(alleles):
    records = []
    failed_records = []
    for allele_tuple in alleles:
        allele_name = format_hla_allele_name(allele_tuple=allele_tuple)
        ebi_hla_details = get_hla_allele_details(allele_name=allele_name)
        if "ebi_id" in ebi_hla_details:
            ebi_hla_details["prot_seq"] = get_hla_prot_sequence(ebi_hla_details["ebi_id"])

        if "ebi_id" in ebi_hla_details and ebi_hla_details["prot_seq"] != "":
                
            ebi_hla_details["hla_gene"] = allele_tuple[0]
            ebi_hla_details["short_code"] = allele_tuple[1]
            record = [ebi_hla_details["ebi_id"], ebi_hla_details["ebi_name"], 
            ebi_hla_details["hla_gene"], ebi_hla_details["short_code"], 
            ebi_hla_details["prot_seq"]
                ]
            records.append(record)
        else:
            print("Failed to find record for {}".format(allele_tuple))
            record = [allele_tuple[0], allele_tuple[1], allele_name]
            failed_records.append(record)
        #Sleep so as to not overwhelm the IP DB
        time.sleep(.1)
    return records, failed_records

def compute_positional_motif_posession(allele_df:pd.DataFrame, 
    motif_definitions:dict, offset = 24
):
    motif_statuses = []

    for index, item in allele_df['protein_sequence'].items():
        motif_status = {}
        for key in motif_definitions:
            criterea = motif_definitions[key]
            motif_status[key] = True
            for critereon in criterea:
                amino_acid = item[critereon[0]-1+offset]
                condition = amino_acid in critereon[1]
                if not condition:
                    motif_status[key] = False
                    break
                else: 
                    pass
        motif_statuses.append(motif_status)
    
    return motif_statuses

def compute_motif_posession(df_hla_allele_tbl:pd.DataFrame):  
    ligand_matching_criteria = ['hla_c_c1', 'hla_c_c2', 'hla_b_46_c1', 'hla_b_73_c1', 'hla_b_bw4']

    motifs_records = []
    for index, row in df_hla_allele_tbl.iterrows():
        critereon_results = {key:False for key in ligand_matching_criteria}
        hla_loci = row['hla_gene'].lower()

        if hla_loci == 'c':
            critereon_results['hla_c_c1'] = row['c1']
            critereon_results['hla_c_c2'] = row['c2']
        elif hla_loci == 'b':
            critereon_results['hla_b_46_c1'] = row['c1']
            critereon_results['hla_b_73_c1'] = row['c1']
            critereon_results['hla_b_bw4'] = row['bw4']

        results_record = [critereon_results[key] for key in ligand_matching_criteria]
        motifs_records.append(results_record)

    columns = ['hla_gene', 'short_code']
    #columns = ['hla_gene', 'short_code', 'c1', 'c2', 'bw4', 'bw6']
    columns.extend(ligand_matching_criteria)
    df_hla_allele_tbl[ligand_matching_criteria] = motifs_records
    return df_hla_allele_tbl[columns].copy()

def find_ligand_motifs(alleles:list):

    alleles = [[x[0], x[1:]] for x in alleles]

    records, failed_records = retrieve_hla_meta_data_from_ipd_db(alleles=alleles)

    columns = ["ebi_id", "ebi_name", "hla_gene", "short_code", "protein_sequence"]
    df_hla_allele_tbl = pd.DataFrame(records, columns=columns)

    motif_definitions = {"c1":[(77,["S"]), (78,["L"]), (79,["R"]), (80,["N"])],
                        "c2":[(77,["N"]), (78,["L"]), (79,["R"]), (80,["K"])], 
                        "bw4":[(80,["T","I"])], 
                        "bw6":[(80,["N"])]
    }

    offset = 24

    # Compute Motif Posession
    motif_statuses = compute_positional_motif_posession(
        df_hla_allele_tbl, motif_definitions, offset
    )

    motif_statuses = [[item['c1'], item['c2'], item['bw4'], item['bw6']] 
        for item in motif_statuses]

    # Format and Export Results
    df_hla_allele_tbl[['c1', 'c2', 'bw4', 'bw6']] = motif_statuses

    df_hla_allele_tbl = compute_motif_posession(df_hla_allele_tbl)

    return df_hla_allele_tbl

alleles = ['B3501', 'B4402', 'B3503']
motif_possession_df = find_ligand_motifs(alleles)

print(motif_possession_df)
