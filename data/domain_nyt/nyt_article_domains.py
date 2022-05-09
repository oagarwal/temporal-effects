import glob
from bs4 import BeautifulSoup
import sys
import numpy as np
import gzip
import json
import os
from random import shuffle
from collections import Counter


BASE_PATH = "/project/cis/nlp/data/corpora/nytimes/data/"


def list_all_files(parentdir):
        return glob.glob(parentdir+'/01/*/*.xml') + glob.glob(parentdir+'/02/*/*.xml') + glob.glob(parentdir+'/03/*/*.xml') + glob.glob(parentdir+'/04/*/*.xml') + glob.glob(parentdir+'/05/*/*.xml') + glob.glob(parentdir+'/06/*/*.xml') + glob.glob(parentdir+'/07/*/*.xml') + glob.glob(parentdir+'/08/*/*.xml')


def parse_file(f):
    soup = BeautifulSoup(open(f), 'html.parser')
    text = ''
    if soup.body.find('block',{'class':'lead_paragraph'}):
        for para in soup.body.find('block',{'class':'lead_paragraph'}).find_all('p'):
            text = text + para.text + ' '
    text = text.strip()

    domain = ''
    if soup.head.find('meta',{'name':'dsk'}):
        domain = soup.head.find('meta',{'name':'dsk'})['content']
    return text, domain


def annotate_files(files, year):
    domains = {'Cultural', 'Culture', 'Weekend', 'Arts', 'Business', 'Financial', 'Editorial', 'Foreign', 'Metropolitan', 'National', 'Sports'}
    domain_mapping = {'Cultural':'Arts', 'Culture':'Arts', 'Weekend':'Arts', 'Financial':'Business'}
    domain_sentences = {'Arts':[], 'Business':[], 'Editorial':[], 'Foreign':[], 'Metropolitan':[] , 'National':[], 'Sports':[], 'Others':[]}

    print(len(files), ' files')
    if len(files) < 40000:
        return

    for i, f in enumerate(files):
        date = f[len(BASE_PATH):-4]

        if i % 2500 == 0:
            print(date)
            print([(d, len(domain_sentences[d])) for d in domain_sentences])

        text, dsk = parse_file(f)
        if not dsk or not text:
            continue

        # Map domain
        domain = 'Others'
        for d in domains:
            if d in dsk:
                domain = d
                break
        if domain in domain_mapping:
            domain = domain_mapping[domain]
        domain_sentences[domain].append({'text':text, 'domain': domain, 'date':date})

    sents_to_write = []
    for domain in domain_sentences:
        sents_to_write += domain_sentences[domain]
    sents_to_write = sents_to_write[:40000]
    shuffle(sents_to_write)
    assert len(sents_to_write) == 40000

    domain_cnt = []
    with gzip.open(year+".jsonl.gz", 'wt') as f:
        for sent in sents_to_write:
            domain_cnt.append(sent['domain'])
            f.write(json.dumps(sent)+'\n')
    print(Counter(domain_cnt))
    print('------------------------------')
    print()


year = sys.argv[1]
print(year)
all_files = list_all_files(os.path.join(BASE_DIR, year))
annotate_files(all_files, year)

