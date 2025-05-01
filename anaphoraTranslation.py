import spacy
import coreferee
import csv
import xml.etree.ElementTree as ET
import pandas as pd
import logging
import os
import gc
import colorlog
import multiprocessing as mp
from multiprocessing import freeze_support
from itertools import islice
from tqdm import tqdm
from coreferee.data_model import Mention


def setupLogging():
    """Initialize logging configuration with colors"""
    logFile = "logFile.log"
    
    # Remove existing handlers to avoid duplicates
    logger = colorlog.getLogger('anaphora')  # Give logger a name
    logger.handlers = []  # Clear existing handlers
    
    # Create handlers
    console_handler = colorlog.StreamHandler()
    file_handler = logging.FileHandler(logFile)
    
    # Create formatters
    console_format = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s%(reset)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    file_format = colorlog.ColoredFormatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set formatters
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(colorlog.INFO)
    
    logger.info("Logging initialized")  
    return logFile, logger

logFile, logger = setupLogging()

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
                logger.warning("Language attribute not found in tuv element.")
                continue
            if lang and seg is not None and seg.text:
                langs.append(lang.lower())
                texts.append(seg.text.strip())
                logger.info(f"Language: {lang}, Text: {seg.text.strip()}")

        if 'en' in langs and 'fr' in langs:
            if langs.count('en') > 1 or langs.count('fr') > 1:
                logger.warning("Multiple entries for the same language found in a single translation unit.")
                continue
            if langs.count('en') == 0 or langs.count('fr') == 0:
                logger.warning("Missing English or French text in a translation unit.")
                continue
            if langs.count('en') == 1 and langs.count('fr') == 1:
                logger.info("Both English and French texts found in a translation unit.")
            try:
                enText = texts[langs.index('en')]
                frText = texts[langs.index('fr')]
                en_texts.append(enText)
                fr_texts.append(frText)
            except (IndexError, ValueError):
                logger.error("Error retrieving English or French text from the lists.")
                continue

    print(f"\nParsed {len(en_texts)} parallel sentences")
    logger.info(f"Parsed {len(en_texts)} parallel sentences")
    return en_texts, fr_texts

def setPipeline(size):
    try:
        # Load SpaCy model
        nlp = spacy.load(f"en_core_web_{size}")
        logger.info(f"Initializing {size} model pipeline")
        
        if 'sentencizer' not in nlp.pipe_names:
            nlp.add_pipe('sentencizer', first=True)
            logger.info(f"Added sentencizer to {size} model")
        
        if 'coreferee' in nlp.pipe_names:
            nlp.remove_pipe('coreferee')
        
        nlp.add_pipe('coreferee')
        logger.info(f"Added Coreferee to {size} model")
        
        # Run a test document to validate coreferee is working
        test_text = "John went to the store. He bought milk."
        doc = nlp(test_text)
        
        if not hasattr(doc._, 'coref_chains'):
            logger.error(f"Coreferee not properly initialized in {size} model")
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
                        logger.warning(f"Unexpected mention type: {type(mention)} - {mention}")
                if mentions:
                    logger.info(f"Coref chain: {', '.join(mentions)}")
            except Exception as e:
                logger.error(f"Error processing coreference chain: {str(e)}")
                continue

        return nlp

    except Exception as e:
        logger.error(f"Error initializing {size} model: {str(e)}")
        return None

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
                                logger.warning(f"Unexpected mention type: {type(mention)} - {mention}")
                        if mentions:
                            doc_coref_chains.append(f"({', '.join(mentions)})")
                    except Exception as e:
                        logger.warning(f"Error processing coref chain: {str(e)}")
                        continue

            for sent in docBatch.sents:
                data['Original_Sentence'].append(sent.text)
                data['Tokens'].append(", ".join(token.text for token in sent))
                data['Named_Entities'].append(", ".join(f"{ent.text} ({ent.label_})" for ent in sent.ents))
                data['Coreference'].append("; ".join(doc_coref_chains))

            if len(data['Original_Sentence']) % 10000 == 0:
                gc.collect()

    return pd.DataFrame(data)

def main():
    logFile = setupLogging()
    logger.info("[ALERT] Starting new processing run!")

    spacyModels = {}
    for size in ["sm", "md", "lg"]:
        nlp = setPipeline(size)
        if nlp is not None:
            spacyModels[size] = nlp
            print(f"Loaded model: en_core_web_{size}")
            logger.info(f"Loaded model: en_core_web_{size}")

    for fileLabel, filePath in tmxFiles.items():
        if not os.path.exists(filePath):
            logger.error(f"File '{filePath}' does not exist.")
            continue

        enSentences, frSentences = parseTMX(filePath)
        print(f"{fileLabel} successfully parsed!")
        logger.info(f"TMX files successfully parsed for {fileLabel}")

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
                
                logger.info(f"Finished processing {fileLabel} with {modelName} model. Output: {outputFile}")
        
    print("\nAll models processed successfully!")

if __name__ == "__main__":
    freeze_support()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        logger.warning("Process interrupted by user")
    except Exception as e:
        print(f"\nError in main process: {str(e)}")
        logger.error(f"Error in main process: {str(e)}")
    finally:
        logger.info("Process completed")
