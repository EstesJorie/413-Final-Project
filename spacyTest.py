import spacy

en_sm = spacy.load("en_core_web_sm")
print(f"en_core_web_sm version: {en_sm.meta['version']}")

en_md = spacy.load("en_core_web_md")
print(f"en_core_web_md version: {en_md.meta['version']}")

en_lg = spacy.load("en_core_web_lg")
print(f"en_core_web_lg version: {en_lg.meta['version']}")

fr_sm = spacy.load("fr_core_news_sm")
print(f"fr_core_news_sm version: {fr_sm.meta['version']}")

fr_md = spacy.load("fr_core_news_md")
print(f"fr_core_news_md version: {fr_md.meta['version']}")

fr_lg = spacy.load("fr_core_news_lg")
print(f"fr_core_news_lg version: {fr_lg.meta['version']}")