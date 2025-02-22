import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_results(model_files, metrics):
    """Load results from multiple CSV files and store metric values for each model."""
    data = []

    for model, file in model_files.items():
        try:
            df = pd.read_csv(file)
            
            for metric in metrics:
                for value in df[metric]:  # Collect all metric values, not just the mean
                    data.append({'Model': model, 'Metric': metric, 'Score': value})
        except Exception as e:
            print(f"Error loading {file} for {model}: {e}")

    return pd.DataFrame(data)

def plot_boxplots(df, metrics):
    """Generate separate boxplots for each metric comparing all models."""
    sns.set(style="whitegrid")

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="Model", y="Score", data=df[df["Metric"] == metric], palette="muted")
        plt.title(f"Comparison of {metric} Across Models")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=15)  # Rotate model names if needed
        plt.show()

def main():
    # Define model names and corresponding CSV file paths
    model_files = {
        "T5-Base": "C:\\Users\\97254\\Desktop\\niv\\Github projects\\NLP-Final-Project---Dreams-Interpreter\\ALL_RESULTS_METRICS\\t5_metrics_results.csv",
        #"T5-Small": "C:\\Users\\97254\\Desktop\\niv\\Github projects\\NLP-Final-Project---Dreams-Interpreter\\ALL_RESULTS_METRICS\\t5_small_metrics_results.csv",
        "GPT-2": "C:\\Users\\97254\\Desktop\\niv\\Github projects\\NLP-Final-Project---Dreams-Interpreter\\ALL_RESULTS_METRICS\\gpt2_metrics_results (1).csv",
        "BERT": "C:\\Users\\97254\\Desktop\\niv\\Github projects\\NLP-Final-Project---Dreams-Interpreter\\ALL_RESULTS_METRICS\\bert_metrics_results.csv"
    }

    # Define the metrics to compare
    metrics = ['BLEU', 'Perplexity', 'BERTScore', 'ROUGE-L']

    # Load results
    df = load_model_results(model_files, metrics)

    # Plot separate boxplots for each metric
    plot_boxplots(df, metrics)

if __name__ == "__main__":
    main()
