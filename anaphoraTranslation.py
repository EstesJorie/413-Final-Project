import spacy
import coreferee
import csv
import xml.etree.ElementTree as ET
import pandas as pd
import logging
import os
import gc
import multiprocessing as mp
from itertools import islice
from tqdm import tqdm

# Set up logging
logFile = "logFile.log"
if not os.path.exists(logFile):
    print(f"Log file '{logFile}' does not exist, creating a new one.\n")

if not os.path.exists('OUTPUT'):
    os.makedirs('OUTPUT')
    print("Created OUTPUT directory")
    logging.info("Created OUTPUT directory.")

logging.basicConfig(filename= logFile,
                    filemode = 'a', #append log file, DO NOT SET to 'w' 
                    format = '%(asctime)s - %(levelname)s - %(message)s',
                    level = logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
print(f"Logging initialized. Check {logFile} for details.\n")

V1filePath = "TMX/en-frv1.tmx"

if not os.path.exists(V1filePath):
    logging.error(f"File '{V1filePath}' does not exist.")
    raise FileNotFoundError(f"File '{V1filePath}' does not exist.")

def parseTMX(filePath):
    tree = ET.parse(filePath) #parse TMX file
    root = tree.getroot() #get root element
    en_texts, fr_texts = [], [] #lists for Eng and French texts
    
    total_units = sum(1 for _ in root.iter('tu')) #total num units for bar
    
    for tu in tqdm(root.iter('tu'), total=total_units, desc="Parsing TMX file"):
        langs, texts = [], [] #lists for languages and texts
        for tuv in tu.findall('tuv'):
            lang = tuv.attrib.get('{http://www.w3.org/XML/1998/namespace}lang') or tuv.attrib.get('lang')
            seg = tuv.find('seg')
            if lang is None:
                logging.warning("Language attribute not found in tuv element.")
                continue
            if lang and seg is not None and seg.text:
                langs.append(lang.lower())
                texts.append(seg.text.strip())
                logging.info(f"Language: {lang}, Text: {seg.text.strip()}")

        if 'en' in langs and 'fr' in langs:
            if langs.count('en') > 1 or langs.count('fr') > 1:
                logging.warning("Multiple entries for the same language found in a single translation unit.")
                continue
            if langs.count('en') == 0 or langs.count('fr') == 0:
                logging.warning("Missing English or French text in a translation unit.")
                continue
            if langs.count('en') == 1 and langs.count('fr') == 1:
                logging.info("Both English and French texts found in a translation unit.")
            try:
                enText = texts[langs.index('en')]
                frText = texts[langs.index('fr')]
                en_texts.append(enText)
                fr_texts.append(frText)
            except (IndexError, ValueError):
                logging.error("Error retrieving English or French text from the lists.")
                continue

    print(f"\nParsed {len(en_texts)} parallel sentences")
    logging.info(f"Parsed {len(en_texts)} parallel sentences")
    return en_texts, fr_texts

V1enSentences, V1frSentences = parseTMX(V1filePath)
print(f"Complete!")

def batchProcess(texts, nlp, batchSize=15):
    """Processing texts in batches"""
   # Initialize lists to store data
    data = {
        'Original_Sentence': [],
        'Tokens': [],
        'Named_Entities': [],
        'Coreference': []
    }

    with nlp.select_pipes(enable=['sentencizer', 'parser', 'coreferee', 'ner']):
        for docBatch in tqdm(list(nlp.pipe(texts, batch_size=batchSize)), desc="Processing batches"):
            if not docBatch.has_annotation("SENT_START"): #MAKE SURE sentence boundaries are set! 
                nlp.get_pipe('sentencizer')(docBatch)
                
            for sent in docBatch.sents:
                data['Original_Sentence'].append(sent.text)
                data['Tokens'].append(", ".join(token.text for token in sent))
                data['Named_Entities'].append(", ".join(f"{ent.text} ({ent.label_})" for ent in sent.ents))
                
                # Process coreference chains
                coref_chains = []
                if hasattr(docBatch._, 'coref_chains') and docBatch._.coref_chains is not None:
                    for chain in docBatch._.coref_chains:
                        try:
                            mentions = [docBatch[mention[0]:mention[1]+1].text for mention in chain]
                            coref_chains.append(f"({', '.join(mentions)})")
                        except IndexError:
                            continue
                data['Coreference'].append("; ".join(coref_chains))
                
            if len(data['Original_Sentence']) % 10000 == 0:
                gc.collect()
                
        return pd.DataFrame(data)

def setPipeline(size):
    try:
        nlp = spacy.load(f"en_core_web_{size}")
        
        # Add sentencizer before coreferee
        if 'sentencizer' not in nlp.pipe_names:
            nlp.add_pipe('sentencizer', before='parser')
            logging.info(f"Added sentencizer to {size} model pipeline")
        
        # Add coreferee after parser
        if 'coreferee' not in nlp.pipe_names:
            nlp.add_pipe('coreferee', after='parser')
            logging.info(f"Added Coreferee to {size} model pipeline")
            
        return nlp
    except Exception as e:
        logging.error(f"Error setting up {size} model pipeline: {str(e)}")
        return None

spacyModels = {}
for size in ["sm", "md", "lg"]:
    nlp = setPipeline(size)
    if nlp is not None:
        spacyModels[size] = nlp
        print(f"Loaded model: en_core_web_{size}")

for modelName, nlp in spacyModels.items():
    print(f"\n{'='*50}")
    print(f"Processing with {modelName.upper()} model")
    print(f"{'='*50}")
    
    outputFile = f"OUTPUT/Output_{modelName}.csv"
    
    # Add description with model size
    desc = f"SpaCy {modelName.upper()} model"
    df = batchProcess(V1enSentences, nlp, batchSize=15)
    
    df['Model'] = modelName
    df.to_csv(outputFile, index=False)
    
    print(f"\nStatistics for {modelName.upper()} model:")
    print(f"Total sentences processed: {len(df)}")
    print(f"Sentences with coreferences: {len(df[df['Coreference'].str.len() > 0])}")
    print(f"Sentences with named entities: {len(df[df['Named_Entities'].str.len() > 0])}")
    print(f"Output saved to: {outputFile}")
    print(f"{'='*50}\n")
    
    logging.info(f"Finished processing with {modelName} model. Output: {outputFile}")
    
print("\nAll models processed successfully!")
