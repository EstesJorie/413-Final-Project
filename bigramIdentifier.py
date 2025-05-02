import os
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef, log_loss, precision_recall_curve
from sklearn.preprocessing import LabelEncoder, label_binarize
from dask.distributed import Client, LocalCluster
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import sys
from datetime import datetime
from io import StringIO
from multiprocessing import freeze_support

plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 16

def setupFonts():
    # Reload the font cache if needed
    font_manager.findfont('Arial')  # This will trigger font cache rebuild if needed
    font_list = [f.name for f in font_manager.fontManager.ttflist]
    if not any(font in font_list for font in ['Arial', 'Helvetica', 'DejaVu Sans']):
        print("Warning: Preferred fonts not found, using system defaults")

def loadCombineDatasets():
    datasets = {
        'wili': {
            'train': {'x': "wili-2018/x_train.txt", 'y': "wili-2018/y_train.txt"},
            'test': {'x': "wili-2018/x_test.txt", 'y': "wili-2018/y_test.txt"}
        },
        'europarl': {
            'train': {'x': "europarl-v7/x_train.txt", 'y': "europarl-v7/y_train.txt"},
            'test': {'x': "europarl-v7/x_test.txt", 'y': "europarl-v7/y_test.txt"}
        },
        'tatoeba': {
            'train': {'x': "tatoeba-sentences/x_train.txt", 'y' : "tatoeba-sentences/y_train.txt"},
            'test': {'x': "tatoeba-sentences/x_test.txt", 'y': "tatoeba-sentences/y_test.txt"}
        },
        'oscar' : {
            'train': {'x': "oscar-data/x_train.txt", 'y' : "oscar-data/y_train.txt"},
            'test': {'x': "oscar-data/x_test.txt", 'y': "oscar-data/y_test.txt"}
        }   
    }
    
    x_train_combined = []
    y_train_combined = []
    x_test_combined = []
    y_test_combined = []
    
    # Load and combine all datasets
    for dataset_name, paths in datasets.items():
        print(f"\nLoading {dataset_name} dataset...")
        
        # Load training data
        if os.path.exists(paths['train']['x']) and os.path.exists(paths['train']['y']):
            with open(paths['train']['x'], encoding="utf-8") as f:
                x_train = [line.strip() for line in f]
            with open(paths['train']['y'], encoding="utf-8") as f:
                y_train = [line.strip() for line in f]
            x_train_combined.extend(x_train)
            y_train_combined.extend(y_train)
            print(f"Added {len(x_train)} training samples from {dataset_name}")
            
        # Load test data
        if os.path.exists(paths['test']['x']) and os.path.exists(paths['test']['y']):
            with open(paths['test']['x'], encoding="utf-8") as f:
                x_test = [line.strip() for line in f]
            with open(paths['test']['y'], encoding="utf-8") as f:
                y_test = [line.strip() for line in f]
            x_test_combined.extend(x_test)
            y_test_combined.extend(y_test)
            print(f"Added {len(x_test)} test samples from {dataset_name}")
    
    return x_train_combined, y_train_combined, x_test_combined, y_test_combined

def printMetrics(y_test_decoded, y_pred_decoded, y_pred_prob, languages):
    # Get unique classes from both true and predicted labels
    unique_classes = np.unique(np.concatenate([y_test_decoded, y_pred_decoded]))
    
    # Classification report with explicit labels
    print("Classification Report:")
    print("=" * 50)
    print(classification_report(y_test_decoded, y_pred_decoded, 
                              labels=unique_classes,
                              target_names=unique_classes))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print("=" * 50)
    cm = confusion_matrix(y_test_decoded, y_pred_decoded, 
                         labels=unique_classes)
    print(cm)

    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(y_test_decoded, y_pred_decoded)
    print("\nMatthews Correlation Coefficient (MCC):")
    print("=" * 50)
    print(f"MCC: {mcc:.4f}")
    
    # Log-Loss with explicit labels
    try:
        loss_value = log_loss(y_test_decoded, y_pred_prob,
                            labels=unique_classes)
        print("\nLog-Loss:")
        print("=" * 50)
        print(f"Log-Loss: {loss_value:.4f}")
    except ValueError as e:
        print("\nWarning: Could not calculate log loss:", str(e))

    # ROC AUC with explicit labels
    print("\nROC AUC Scores:")
    print("=" * 50)
    for i, label in enumerate(unique_classes):
        try:
            auc = roc_auc_score(y_test_decoded == label, 
                              y_pred_prob[:, i])
            print(f"Class {label}: AUC = {auc:.4f}")
        except ValueError as e:
            print(f"Class {label}: AUC = nan")

def orderResults(y_test_decoded, y_pred_decoded, label_encoder, languages, metric='f1-score'):
    # Generate the classification report
    metrics = classification_report(y_test_decoded, y_pred_decoded, 
                                   target_names=languages,
                                   output_dict=True)
    
    # Convert the classification report into a pandas DataFrame for easier sorting
    metrics_df = pd.DataFrame(metrics).T  # Transpose for easier reading

    # Sort based on the specified metric (default: 'f1-score')
    sorted_metrics_df = metrics_df.sort_values(by=metric, ascending=False)

    # Print the sorted metrics
    print(sorted_metrics_df)

    sorted_metrics_df.to_csv(f'sorted_metrics_{metric}.csv', index=True)
    return sorted_metrics_df

def plotResults(y_test_encoded, y_pred, label_encoder, languages, output_dir, chunk_size=20):
    # Generate confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    # Get the decoded labels for the report
    y_test_decoded = label_encoder.inverse_transform(y_test_encoded)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    # Generate Classification Report with target names
    metrics = classification_report(y_test_decoded, y_pred_decoded, 
                                 target_names=languages,
                                 output_dict=True)
    
    # Split the languages into smaller chunks for pagination
    num_chunks = int(np.ceil(len(languages) / chunk_size))
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(languages))
        chunk_languages = languages[start_idx:end_idx]
        
        cm_chunk = cm[start_idx:end_idx, start_idx:end_idx]
        
        fig = plt.figure(figsize=(20, 10))
        
        # Confusion Matrix
        ax1 = plt.subplot(1, 2, 1)
        sns.heatmap(cm_chunk, annot=True, fmt='d', cmap='Blues',
                    xticklabels=chunk_languages,
                    yticklabels=chunk_languages,
                    square=True,  # Make cells square
                    cbar_kws={'shrink': .8})
        
        plt.title(f'Confusion Matrix (Chunk {i+1})', pad=20)
        plt.xlabel('Predicted', labelpad=10)
        plt.ylabel('True', labelpad=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Metrics Plot
        ax2 = plt.subplot(1, 2, 2)
        metrics_df = pd.DataFrame({
            'Precision': [metrics[lang]['precision'] for lang in chunk_languages],
            'Recall': [metrics[lang]['recall'] for lang in chunk_languages],
            'F1-Score': [metrics[lang]['f1-score'] for lang in chunk_languages]
        }, index=chunk_languages)

        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn',
                    square=True,
                    cbar_kws={'shrink': .8})
        plt.title(f'Performance Metrics (Chunk {i+1})', pad=20)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'results_chunk_{i+1}.png'), 
                    bbox_inches='tight', 
                    pad_inches=0.5)
        plt.close(fig)

def plotTopClasses(y_test_decoded, y_pred_decoded, label_encoder, languages, metric='f1-score'):
    # Generate the classification report
    metrics = classification_report(y_test_decoded, y_pred_decoded, 
                                   target_names=languages,
                                   output_dict=True)
    
    # Convert the classification report into a pandas DataFrame for easier sorting
    metrics_df = pd.DataFrame(metrics).T  # Transpose for easier reading
    
    # Sort based on the chosen metric (e.g., f1-score)
    sorted_metrics_df = metrics_df.sort_values(by=metric, ascending=False)
    
    # Select the top 25 classes (languages)
    top_25_classes = sorted_metrics_df.head(25)
    
    plt.figure(figsize=(15, 8))
    ax = top_25_classes[metric].plot(kind='bar', color='skyblue',
                                   width=0.8)
    
    plt.title(f"Top 25 Languages by {metric.capitalize()}", 
             pad=20, fontsize=16, fontweight='bold')
    plt.xlabel('Language', labelpad=10, fontsize=12)
    plt.ylabel(f'{metric.capitalize()}', labelpad=10, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    for i, v in enumerate(top_25_classes[metric]):
        ax.text(i, v, f'{v:.3f}', 
                ha='center', va='bottom')
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('top_25_classes_by_' + metric + '.png',
                bbox_inches='tight',
                pad_inches=0.5)
    plt.close()

def plotClassDistribution(y_train, y_test, label_encoder, output_dir):
    # Convert to pandas Series and get value counts
    train_class_counts = pd.Series(y_train).value_counts()
    test_class_counts = pd.Series(y_test).value_counts()
    
    # Get all unique classes in sorted order
    all_classes = sorted(set(y_train) | set(y_test))
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), dpi=300)
    
    bar_params = {
        'alpha': 0.8,
        'width': 0.8,
        'edgecolor': 'black',
        'linewidth': 0.5
    }
    
    # Plot training data
    train_bars = axes[0].bar(range(len(all_classes)), 
                            [train_class_counts.get(c, 0) for c in all_classes],
                            color='skyblue', **bar_params)
    axes[0].set_title('Class Distribution in Training Data', 
                     pad=20, fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Language', labelpad=10, fontsize=12)
    axes[0].set_ylabel('Number of Samples', labelpad=10, fontsize=12)
    axes[0].set_xticks(range(len(all_classes)))
    axes[0].set_xticklabels(all_classes, rotation=45, ha='right')
    
    # Add value labels on training bars
    for bar in train_bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=8)
    
    # Plot test data
    test_bars = axes[1].bar(range(len(all_classes)), 
                           [test_class_counts.get(c, 0) for c in all_classes],
                           color='lightcoral', **bar_params)
    axes[1].set_title('Class Distribution in Test Data', 
                     pad=20, fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Language', labelpad=10, fontsize=12)
    axes[1].set_ylabel('Number of Samples', labelpad=10, fontsize=12)
    axes[1].set_xticks(range(len(all_classes)))
    axes[1].set_xticklabels(all_classes, rotation=45, ha='right')

def plotMisclassifiedSamples(x_test, y_test_decoded, y_pred_decoded, output_dir, num_samples=10):
    # Find misclassified samples
    misclassified_idx = [i for i in range(len(y_test_decoded)) 
                        if y_test_decoded[i] != y_pred_decoded[i]]
    
    confusion_counts = {}
    for idx in misclassified_idx:
        pair = (y_test_decoded[idx], y_pred_decoded[idx])
        confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
    
    sorted_confusions = sorted(confusion_counts.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
    
    selected_idx = []
    seen_pairs = set()
    for idx in misclassified_idx:
        pair = (y_test_decoded[idx], y_pred_decoded[idx])
        if pair not in seen_pairs and len(selected_idx) < num_samples:
            selected_idx.append(idx)
            seen_pairs.add(pair)
    
    samples = []
    for idx in selected_idx:
        samples.append({
            'text': x_test[idx],
            'true': y_test_decoded[idx],
            'pred': y_pred_decoded[idx],
            'confusion_count': confusion_counts[(y_test_decoded[idx], y_pred_decoded[idx])]
        })
    
    fig, axes = plt.subplots(len(samples), 1, 
                            figsize=(15, 3 * len(samples)),
                            dpi=300)
    if len(samples) == 1:
        axes = [axes]

    fontFamily = setupFonts()
    
    for i, (sample, ax) in enumerate(zip(samples, axes)):
        # Create formatted text
        text = (f"True Label: {sample['true']}\n"
                f"Predicted: {sample['pred']}\n"
                f"Times Confused: {sample['confusion_count']}\n"
                f"Sample Text:\n{sample['text']}")
        
        ax.text(0.05, 0.5, text,
                transform=ax.transAxes,
                fontsize=11,
                family=fontFamily,
                bbox=dict(facecolor='white',
                         edgecolor='gray',
                         alpha=0.9,
                         pad=10,
                         boxstyle='round'),
                wrap=True,
                verticalalignment='center')
        ax.axis('off')
    
    fig.suptitle('Most Common Misclassifications',
                 fontsize=16,
                 fontweight='bold',
                 y=0.98)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'misclassified_samples.png'),
                bbox_inches='tight',
                pad_inches=0.5,
                dpi=300)
    plt.close()
    
    report_path = os.path.join(output_dir, 'misclassification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Misclassification Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("Most Common Confusions:\n")
        for (true, pred), count in sorted_confusions[:20]:
            f.write(f"{true} → {pred}: {count} times\n")

def plotTopMisclassifiedClasses(y_test_decoded, y_pred_decoded, label_encoder, output_dir, top_n=10):
    # Generate confusion matrix
    cm = confusion_matrix(y_test_decoded, y_pred_decoded)
    
    # Calculate misclassification rates with error handling
    class_totals = np.sum(cm, axis=1)
    correct_predictions = np.diag(cm)
    
    # Avoid division by zero and handle NaN values
    misclassification_rates = np.zeros_like(class_totals, dtype=float)
    mask = class_totals > 0  # Only calculate for classes with samples
    misclassification_rates[mask] = 1 - (correct_predictions[mask] / class_totals[mask])
    
    # Get top N misclassified classes (excluding NaN)
    valid_idx = ~np.isnan(misclassification_rates)
    sorted_idx = np.argsort(misclassification_rates[valid_idx])[-top_n:][::-1]
    top_classes = label_encoder.classes_[sorted_idx]
    top_rates = misclassification_rates[sorted_idx]
    
    # Create figure with higher resolution
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_classes)), top_rates, 
                   color='tomato', alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars with error handling
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if not np.isnan(width):
            total_samples = class_totals[sorted_idx[i]]
            wrong_samples = int(np.round(total_samples * width))
            label = f'{width:.1%} ({wrong_samples}/{int(total_samples)})'
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'  {label}', va='center', fontsize=9)
    
    # Customize plot
    ax.set_title(f'Top {len(top_classes)} Most Misclassified Languages',
                pad=20, fontsize=14, fontweight='bold')
    ax.set_xlabel('Misclassification Rate', labelpad=10, fontsize=12)
    ax.set_ylabel('Language', labelpad=10, fontsize=12)
    
    # Set y-axis ticks and labels
    ax.set_yticks(range(len(top_classes)))
    ax.set_yticklabels(top_classes, fontsize=10)
    
    # Customize grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add summary statistics with error handling
    valid_rates = misclassification_rates[~np.isnan(misclassification_rates)]
    stats_text = (f'Total Classes: {len(label_encoder.classes_)}\n'
                 f'Avg. Misclass. Rate: {np.mean(valid_rates):.1%}\n'
                 f'Med. Misclass. Rate: {np.median(valid_rates):.1%}')
    
    fig.text(0.98, 0.98, stats_text,
             fontsize=9, family='monospace',
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.5),
             ha='right', va='top')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'top_{top_n}_misclassified_classes.png'),
                bbox_inches='tight',
                pad_inches=0.5,
                dpi=300)
    plt.close()

def main():
    setupFonts() #for plots

    # Initialize the Dask Client and Cluster
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    old_stdout = sys.stdout
    output_capture = StringIO()
    sys.stdout = output_capture
    
    try:
        cluster = LocalCluster(processes=True)
        client = Client(cluster)

        # Load combined training and test data
        x_train, y_train, x_test, y_test = loadCombineDatasets()

        # Get unique labels from both train and test sets
        train_labels = set(y_train)
        test_labels = set(y_test)
        
        # Find common labels between train and test
        common_labels = sorted(train_labels.intersection(test_labels))
        print(f"\nTotal number of common languages: {len(common_labels)}")

        # Filter training data to include only common labels
        x_train_filtered = []
        y_train_filtered = []
        for x, y in tqdm(zip(x_train, y_train), 
                        desc="Processing Training Data", 
                        total=len(x_train)):
            if y in common_labels:
                x_train_filtered.append(x)
                y_train_filtered.append(y)

        # Filter test data similarly
        x_test_filtered = []
        y_test_filtered = []
        for x, y in tqdm(zip(x_test, y_test),
                        desc="Processing Test Data",
                        total=len(x_test)):
            if y in common_labels:
                x_test_filtered.append(x)
                y_test_filtered.append(y)

        # Update label encoder with common labels
        label_encoder = LabelEncoder()
        label_encoder.fit(common_labels)
        y_train_encoded = np.array(label_encoder.transform(y_train_filtered))
        y_test_encoded = np.array(label_encoder.transform(y_test_filtered))

        # Vectorize the text data using character bigrams (n-grams)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2), max_features=10000)
        X_vectorized = vectorizer.fit_transform(x_train_filtered)

        # Check if the matrix is in CSR format and convert if necessary
        if not sp.isspmatrix_csr(X_vectorized):
            X_vectorized_csr = X_vectorized.tocsr()
        else:
            X_vectorized_csr = X_vectorized

        # Train model using Naive Bayes (parallelize training using `joblib`)
        def train_model(X, y):
            clf = MultinomialNB()
            clf.fit(X, y)
            return clf

        # Parallelize training on chunks (splitting data for multi-core processing)
        n_chunks = 4
        chunk_size = len(x_train_filtered) // n_chunks

        futures = []
        for i in range(n_chunks):
            future = client.submit(
                train_model,
                X_vectorized_csr[i * chunk_size: (i + 1) * chunk_size],
                y_train_encoded[i * chunk_size: (i + 1) * chunk_size]
            )
            futures.append(future)
        
        # Wait for results and gather models
        models = [future.result() for future in futures]

        # Clean up Dask client and cluster
        client.close()
        cluster.close()

        # Ensure all models are valid and combine predictions
        valid_models = [model for model in models if model is not None]
        if not valid_models:
            raise ValueError("No valid models were trained")
        
        final_model = valid_models[0]  # Use the first valid model (can also consider an ensemble approach)

        # Load test data and process it similarly
        x_test_subset = []
        y_test_subset = []

        with open("wili-2018/x_test.txt", encoding="utf-8") as f:
            x_test = [line.strip() for line in f]

        with open("wili-2018/y_test.txt", encoding="utf-8") as f:
            y_test = [line.strip() for line in f]

        for text, label in tqdm(zip(x_test, y_test), desc="Processing Test Data", total=len(x_test)):
            if label in common_labels:
                x_test_subset.append(text)
                y_test_subset.append(label)

        y_test_encoded = label_encoder.transform(y_test_subset)
        X_test_vectorized = vectorizer.transform(x_test_subset)

        # Make predictions on test data
        y_pred = final_model.predict(X_test_vectorized)

        # Evaluate the model
        accuracy = accuracy_score(y_test_encoded, y_pred)
        print(f"\nModel Evaluation Results")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification Report
        print("\nClassification Report:")
        print("=" * 50)
        print(classification_report(y_test_encoded, y_pred))
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        print("=" * 50)
        print(confusion_matrix(y_test_encoded, y_pred))
        
        # Matthews Correlation Coefficient (MCC)
        mcc = matthews_corrcoef(y_test_encoded, y_pred)
        print("\nMatthews Correlation Coefficient (MCC):")
        print("=" * 50)
        print(f"MCC: {mcc:.4f}")

        # Log-Loss
        # Assuming you have predicted probabilities available
        y_pred_prob = final_model.predict_proba(X_test_vectorized)
        log_loss_value = log_loss(y_test_encoded, y_pred_prob, labels=range(len(label_encoder.classes_)))
        print("\nLog-Loss:")
        print("=" * 50)
        print(f"Log-Loss: {log_loss_value:.4f}")
        
        # ROC AUC Scores (if applicable)
        print("\nROC AUC Scores (One-vs-Rest):")
        print("=" * 50)
        try:
            auc_scores = []
            for i in range(len(label_encoder.classes_)):
                auc = roc_auc_score(y_test_encoded == i, y_pred_prob[:, i])
                auc_scores.append((label_encoder.classes_[i], auc))
                print(f"Class {label_encoder.classes_[i]}: AUC = {auc:.4f}")
            print(f"\nAverage AUC: {np.mean([score[1] for score in auc_scores]):.4f}")
        except Exception as e:
            print(f"Error calculating AUC: {str(e)}")
                
        # Misclassified Samples
        print("\nPlotting Misclassified Samples...")
        plotMisclassifiedSamples(x_test, y_test_encoded, y_pred, output_dir)

        # Save captured output
        sys.stdout = old_stdout
        with open(os.path.join(output_dir, f'evaluation_results.txt'), 'w') as f:
            f.write(output_capture.getvalue())

        plotResults(y_test_encoded, y_pred, label_encoder, label_encoder.classes_, output_dir)
        
        y_test_decoded = label_encoder.inverse_transform(y_test_encoded)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        
        plotTopClasses(y_test_decoded, y_pred_decoded, label_encoder, label_encoder.classes_, metric='f1-score')

        unique_classes = np.unique(np.concatenate([y_test_decoded, y_pred_decoded]))
        test_proba = final_model.predict_proba(X_test_vectorized)
        
        printMetrics(y_test_decoded, y_pred_decoded, test_proba, unique_classes)
        plotClassDistribution(y_train_filtered, y_test_filtered, label_encoder, output_dir)
        plotMisclassifiedSamples(x_test, y_test_decoded, y_pred_decoded, output_dir)
        plotTopMisclassifiedClasses(y_test_decoded, y_pred_decoded, label_encoder, output_dir, top_n=10)
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Error occurred: {str(e)}")
        raise e
    finally:
        sys.stdout = old_stdout

if __name__ == '__main__':
    freeze_support()  # Ensure multiprocessing works on Windows
    main()
