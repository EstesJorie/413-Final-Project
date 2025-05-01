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
    
    logger = colorlog.getLogger('anaphora')
    logger.handlers = []
    
    console_handler = colorlog.StreamHandler()
    file_handler = logging.FileHandler(logFile)
    
    # Set console to WARNING level (will only show warnings and errors)
    console_handler.setLevel(colorlog.WARNING)
    # Keep file handler at INFO level for complete logs
    file_handler.setLevel(colorlog.INFO)
    
    # Create formatters
    console_format = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s: %(message)s%(reset)s",  # Simplified console format
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    file_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(colorlog.DEBUG)  # Allow all levels to be processed
    
    return logFile, logger

logFile, logger = setupLogging()

tmxFiles = {
    "V1" : "TMX/en-frv1.tmx",
    "V2016" : "TMX/en-frv2016.tmx"
}

def parseTMX(filePath):
    """Parse TMX file with memory-efficient iterative parsing"""
    logger.info(f"Starting to parse: {filePath}")
    print(f"Starting to prase: {filePath}")
    
    context = ET.iterparse(filePath, events=("start", "end"))
    context = iter(context)
    
    event, root = next(context)
    
    en_texts, fr_texts = [], []
    current_texts = {}
    
    logger.info("Counting translation units...")
    total_units = sum(1 for _, elem in ET.iterparse(filePath) if elem.tag == 'tu')
    logger.info(f"Found {total_units} translation units")
    
    with tqdm(total=total_units, desc="Parsing TMX file") as pbar:
        for event, elem in context:
            if event == "end" and elem.tag == "tu":
                current_texts = {}
                
                for tuv in elem.findall('tuv'):
                    lang = tuv.attrib.get('{http://www.w3.org/XML/1998/namespace}lang') 
                    lang = lang.lower() if lang else None
                    
                    if not lang:
                        lang = tuv.attrib.get('lang')
                        lang = lang.lower() if lang else None
                        
                    if lang and (lang == 'en' or lang == 'fr'):
                        seg = tuv.find('seg')
                        if seg is not None and seg.text:
                            current_texts[lang] = seg.text.strip()
                
                if 'en' in current_texts and 'fr' in current_texts:
                    en_texts.append(current_texts['en'])
                    fr_texts.append(current_texts['fr'])
                
                # Clear element to free memory
                elem.clear()
                pbar.update(1)
                
                # Garbage collect periodically
                if len(en_texts) % 10000 == 0:
                    root.clear()
                    gc.collect()
    
    logger.info(f"Finished parsing {len(en_texts)} parallel sentences")
    return en_texts, fr_texts

def setPipeline(size, lang="en"):
    try:
        if lang == "en":
            nlp = spacy.load(f"en_core_web_{size}")
        elif lang == "fr":
            nlp = spacy.load(f"fr_core_news_{size}")
        else:
            raise ValueError("Language not supported. Only 'en' and 'fr' are supported.")
        
        logger.info(f"Initializing {lang} {size} model pipeline")
        
        if 'sentencizer' not in nlp.pipe_names:
            nlp.add_pipe('sentencizer', first=True)
            logger.info(f"Added sentencizer to {lang} {size} model")
        
        if 'coreferee' in nlp.pipe_names:
            nlp.remove_pipe('coreferee')
        
        nlp.add_pipe('coreferee')
        logger.info(f"Added Coreferee to {lang} {size} model")
        
        test_text = "John went to the store. He bought milk." if lang == "en" else "Jean est allé au magasin. Il a acheté du lait."
        doc = nlp(test_text)
        
        if not hasattr(doc._, 'coref_chains'):
            logger.error(f"Coreferee not properly initialized in {lang} {size} model")
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

def batchProcess(texts, nlp, lang="en", batchSize=30, sentence_window=4):
    """Processing texts in batches with coreferee, combining sentences into blocks for both languages"""
    data = {
        'Original_Sentence': [],
        'Tokens': [],
        'Named_Entities': [],
        'Coreference': []
    }

    with nlp.select_pipes(enable=['sentencizer', 'parser', 'coreferee', 'ner']):
        for batch_start in tqdm(range(0, len(texts), batchSize), desc=f"Processing {lang} batches"):
            # Create a larger block by combining sentences
            combined_text = " ".join(texts[batch_start: batch_start + batchSize])
            
            # Process the combined block with spaCy
            doc = nlp(combined_text)

            # After processing the combined block, reassign coreference chains to individual sentences
            coref_chains = doc._.coref_chains if hasattr(doc._, 'coref_chains') else []
            sent_idx = 0  # Index for keeping track of sentence positions
            
            for sent in doc.sents:
                tokens = ", ".join(token.text for token in sent)
                named_entities = ", ".join(f"{ent.text} ({ent.label_})" for ent in sent.ents)
                
                # Initialize coreference string
                coreference = []
                
                # Check if the sentence contains a coreference chain
                for chain in coref_chains:
                    for mention in chain:
                        # If the sentence contains this mention, we add the chain to the coreference
                        if mention.start >= sent.start and mention.end <= sent.end:
                            coreference.append(f"{mention.text} ({mention.start}:{mention.end})")
                
                # Add the data for this sentence to the DataFrame
                data['Original_Sentence'].append(sent.text)
                data['Tokens'].append(tokens)
                data['Named_Entities'].append(named_entities)
                data['Coreference'].append("; ".join(coreference))

            # Periodic garbage collection for memory efficiency
            if len(data['Original_Sentence']) % 10000 == 0:
                gc.collect()

    return pd.DataFrame(data)

def main():
    logger.warning("[ALERT] Starting new processing run!")

    for fileLabel, filePath in tmxFiles.items():
        if not os.path.exists(filePath):
            logger.error(f"TMX file missing: {filePath}")
            continue
        logger.info(f"Found TMX file: {filePath}")
        file_size = os.path.getsize(filePath) / (1024 * 1024)  # Size in MB
        logger.info(f"File size: {file_size:.2f} MB")

    spacyModels = {}
    for size in ["sm", "md", "lg"]:
        # Load English models
        nlp_en = setPipeline(size, lang="en")
        if nlp_en is not None:
            spacyModels[f"en_{size}"] = nlp_en
            print(f"Loaded model: en_core_web_{size}")
            logger.info(f"Loaded model: en_core_web_{size}")

        # Load French models
        nlp_fr = setPipeline(size, lang="fr")
        if nlp_fr is not None:
            spacyModels[f"fr_{size}"] = nlp_fr
            print(f"Loaded model: fr_core_news_{size}")
            logger.info(f"Loaded model: fr_core_news_{size}")

    for fileLabel, filePath in tmxFiles.items():
        if not os.path.exists(filePath):
            logger.error(f"File '{filePath}' does not exist.")
            continue

        enSentences, frSentences = parseTMX(filePath)
        print(f"{fileLabel} successfully parsed!")
        logger.info(f"TMX files successfully parsed for {fileLabel}")

        for modelName, nlp in spacyModels.items():
                print(f"\n{'='*50}")
                print(f"Processing OPUS {fileLabel} with spaCy {modelName.upper()} model")
                print(f"{'='*50}")
                
                outputFile = f"OUTPUT/OUTPUT_{fileLabel}_{modelName}.csv"
                
                if modelName.startswith("en"):
                    df = batchProcess(enSentences, nlp, lang="en", batchSize=35, sentence_window=4)
                elif modelName.startswith("fr"):
                    df = batchProcess(frSentences, nlp, lang="fr", batchSize=35, sentence_window=4)
                
                df['Model'] = modelName
                df['TMX_File'] = fileLabel
                df.to_csv(outputFile, index=False)

                print(f"\nStatistics for {fileLabel} - {modelName.upper()} model:")
                print(f"Total sentences processed: {len(df)}")
                print(f"Sentences with coreferences: {len(df[df['Coreference'].str.len() > 0])}")
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
