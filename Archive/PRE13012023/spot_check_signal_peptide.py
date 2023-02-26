import re, requests, time
import numpy as np
import pandas as pd

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm

config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)

# B*08:02 
url = "https://www.ebi.ac.uk/cgi-bin/ipd/api/allele/HLA00152"

codon_dict = {"UUU":"F", "CUU":"L", "AUU":"I", "GUU":"V", "UUC":"F", "CUC":"L",
     "AUC":"I", "GUC":"V", "UUA":"L", "CUA":"L", "AUA":"I", "GUA":"V", 
     "UUG":"L", "CUG":"L", "AUG":"M", "GUG":"V", "UCU":"S", "CCU":"P", 
     "ACU":"T", "GCU":"A", "UCC":"S", "CCC":"P", "ACC":"T", "GCC":"A", 
     "UCA":"S", "CCA":"P", "ACA":"T", "GCA":"A", "UCG":"S", "CCG":"P", 
     "ACG":"T", "GCG":"A", "UAU":"Y", "CAU":"H", "AAU":"N", "GAU":"D", 
     "UAC":"Y", "CAC":"H", "AAC":"N", "GAC":"D", "UAA":"Stop", "CAA":"Q", 
     "AAA":"K", "GAA":"E", "UAG":"Stop", "CAG":"Q", "AAG":"K", "GAG":"E", 
     "UGU":"C", "CGU":"R", "AGU":"S", "GGU":"G", "UGC":"C", "CGC":"R", 
     "AGC":"S", "GGC":"G", "UGA":"Stop", "CGA":"R", "AGA":"R", "GGA":"G", 
     "UGG":"W", "CGG":"R", "AGG":"R", "GGG":"G"
}

def protein(rna_segment):
    return ''.join([codon_dict[rna_segment[i-3:i]] for i in range(3, len(rna_segment), 3)])

r = requests.get(url)

exon2 = None
alpha_sgmt = ""
for feature in r.json()['feature']['genomic']:
    if feature['type'] == 'exon' and feature['number'] == '1':
        exon = feature

        start = int(exon['start']) - 1
        end = start + int(exon['length'])

        sequence = r.json()['sequence']['genomic']
        segment = sequence[start:end]

        rna_segment = segment.replace('T', 'U')

        signal_peptide = protein(rna_segment)

        print('exon:', feature['number'], start, end, len(rna_segment), len(signal_peptide), signal_peptide)

latter_half = r.json()['sequence']['protein'].replace(signal_peptide, "")
print(latter_half[80-1])