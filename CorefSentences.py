import pandas as pd
import re
import spacy

filePaths = {
    "sm": "OUTPUT/OUTPUT_V1_sm.csv",
    "md": "OUTPUT/OUTPUT_V1_md.csv",
    "lg": "OUTPUT/OUTPUT_V1_lg.csv"
}

def sampleCorefSentences(file_paths, sample_size=100, random_seed=42):
    """
    Args:
        file_paths (dict): Dictionary mapping labels (e.g., 'sm', 'md', 'lg') to file paths.
        sample_size (int): Number of samples to extract per file.
        random_seed (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary mapping each label to its sampled DataFrame.
    """
    pronouns = ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'their', 'its']
    pattern = r'\b(' + '|'.join(pronouns) + r')\b'
    
    sampledData = {}

    for label, path in file_paths.items():
        print(f"\nProcessing {label.upper()} from {path}")
        df = pd.read_csv(path)

        # Filter for coreference chains
        coref_df = df[df['Coreference'].notnull() & (df['Coreference'].str.strip() != '')]
        print(f"  → Coreference sentences: {len(coref_df)}")

        # Filter for sentences with target pronouns
        pronoun_df = coref_df[coref_df['Original_Sentence'].str.contains(pattern, flags=re.IGNORECASE, regex=True)]
        print(f"  → With pronouns: {len(pronoun_df)}")

        # Sample the subset
        sample_df = pronoun_df.sample(n=min(sample_size, len(pronoun_df)), random_state=random_seed)
        sampledData[label] = sample_df

    return sampledData

def resolveCoreference(row, nlp):
    sentence = row['Original_Sentence']
    chains = row['Coreference']
    
    if pd.isna(chains) or not chains.strip():
        return sentence

    doc = nlp(sentence)
    token_list = [token.text for token in doc]
    replacements = []

    # Parse coref chains like: (Mary (0:1), She (3:4)); ...
    chain_pattern = re.findall(r'\((.*?)\)', chains)
    
    for chain in chain_pattern:
        mentions = re.findall(r'(.+?) \((\d+):(\d+)\)', chain)
        if not mentions or len(mentions) < 2:
            continue  # Need at least two mentions to resolve

        # Use the first mention as the canonical antecedent
        canonical_text = mentions[0][0]

        for text, start, end in mentions[1:]:
            start, end = int(start), int(end)
            replacements.append((start, end, canonical_text))

    # Sort replacements in reverse so indices don't shift
    replacements.sort(reverse=True, key=lambda x: x[0])

    for start, end, replacement in replacements:
        token_list[start:end] = [replacement]

    return " ".join(token_list)


samples = sampleCorefSentences(filePaths)

for label, df in samples.items():
    print(f"Resolving coreference in {label.upper()} sample...")
    
    df['Resolved_Sentence'] = df.apply(lambda row: resolveCoreference(row, nlp), axis=1)
    
    output_path = f"evaluationSample_{label}_resolved.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved resolved sentences to {output_path}")