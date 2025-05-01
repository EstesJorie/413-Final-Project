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
from coreferee.data_model import Mention


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

tmxFiles = {
    "V1" : "TMX/en-frv1.tmx",
    "V2016" : "TMX/en-frv2016.tmx"
}

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

def setPipeline(size):
    try:
        # Load SpaCy model
        nlp = spacy.load(f"en_core_web_{size}")
        logging.info(f"Loaded {size} model")
        
        if 'sentencizer' not in nlp.pipe_names:
            nlp.add_pipe('sentencizer', first=True)
            logging.info(f"Added sentencizer to {size} model")
        
        if 'coreferee' in nlp.pipe_names:
            nlp.remove_pipe('coreferee')
        
        nlp.add_pipe('coreferee')
        logging.info(f"Added Coreferee to {size} model")
        
        # Run a test document to validate coreferee is working
        test_text = "John went to the store. He bought milk."
        doc = nlp(test_text)
        
        if not hasattr(doc._, 'coref_chains'):
            logging.error(f"Coreferee not properly initialized in {size} model")
            return None

        for chain in doc._.coref_chains:
            try:
                mentions = []
                for mention in chain:
                    if isinstance(mention, Mention):
                        start_idx = max(0, min(mention.root_index, len(doc) - 1))
                        end_idx = max(0, min(mention.root_index + 1, len(doc)))
                        if start_idx < end_idx:
                            span = doc[start_idx:end_idx]
                            mentions.append(f"{span.text} ({start_idx}:{end_idx})")
                    else:
                        logging.warning(f"Unexpected mention type: {type(mention)} - {mention}")
                if mentions:
                    logging.info(f"Coref chain: {', '.join(mentions)}")
            except Exception as e:
                logging.error(f"Error processing coreference chain: {str(e)}")
                continue

        return nlp

    except Exception as e:
        logging.error(f"Error initializing {size} model: {str(e)}")
        return None

spacyModels = {}
for size in ["sm", "md", "lg"]:
    nlp = setPipeline(size)
    if nlp is not None:
        spacyModels[size] = nlp
        print(f"Loaded model: en_core_web_{size}")
        logging.info(f"Loaded model: en_core_web_{size}")

def batchProcess(texts, nlp, batchSize=30):
    """Processing texts in batches with coreferee"""
    data = {
        'Original_Sentence': [],
        'Tokens': [],
        'Named_Entities': [],
        'Coreference': []
    }

    with nlp.select_pipes(enable=['sentencizer', 'parser', 'coreferee', 'ner']):
        for docBatch in tqdm(list(nlp.pipe(texts, batch_size=batchSize, n_process=mp.cpu_count())), desc="Processing batches"):
            doc_coref_chains = []

            if hasattr(docBatch._, 'coref_chains') and docBatch._.coref_chains is not None:
                for chain in docBatch._.coref_chains:
                    try:
                        mentions = []
                        for mention in chain:
                            if isinstance(mention, Mention):
                                start_idx = max(0, min(mention.root_index, len(docBatch) - 1))
                                end_idx = max(0, min(mention.root_index + 1, len(docBatch)))
                                if start_idx < end_idx:
                                    span = docBatch[start_idx:end_idx]
                                    mentions.append(f"{span.text} ({start_idx}:{end_idx})")
                            else:
                                logging.warning(f"Unexpected mention type: {type(mention)} - {mention}")
                        if mentions:
                            doc_coref_chains.append(f"({', '.join(mentions)})")
                    except Exception as e:
                        logging.warning(f"Error processing coref chain: {str(e)}")
                        continue

            for sent in docBatch.sents:
                data['Original_Sentence'].append(sent.text)
                data['Tokens'].append(", ".join(token.text for token in sent))
                data['Named_Entities'].append(", ".join(f"{ent.text} ({ent.label_})" for ent in sent.ents))
                data['Coreference'].append("; ".join(doc_coref_chains))

            if len(data['Original_Sentence']) % 10000 == 0:
                gc.collect()

    return pd.DataFrame(data)
for fileLabel, filePath in tmxFiles.items():
    if not os.path.exists(filePath):
        logging.error(f"File '{filePath}' does not exist.")
        continue

    enSentences, frSentences = parseTMX(tmxFiles)
    print(f"Files Parsed!")
    logging.info("TMX files successfully parsed")

    for modelName, nlp in spacyModels.items():
            print(f"\n{'='*50}")
            print(f"Processing {fileLabel} with {modelName.upper()} model")
            print(f"{'='*50}")
            
            outputFile = f"OUTPUT/OUTPUT_{fileLabel}_{modelName}.csv"
            
            df = batchProcess(enSentences, nlp, batchSize=35)
            df['Model'] = modelName
            df['TMX_File'] = fileLabel
            df.to_csv(outputFile, index=False)

            print(f"\nStatistics for {fileLabel} - {modelName.upper()} model:")
            print(f"Total sentences processed: {len(df)}")
            print(f"Sentences with coreferences: {len(df[df['Coreference'].str.len() > 0])}")
            print(f"Sentences with named entities: {len(df[df['Named_Entities'].str.len() > 0])}")
            print(f"{'='*50}\n")
            
            logging.info(f"Finished processing {fileLabel} with {modelName} model. Output: {outputFile}")
    
print("\nAll models processed successfully!")
