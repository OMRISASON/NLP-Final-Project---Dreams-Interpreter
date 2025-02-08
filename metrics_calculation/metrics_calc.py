
import torch
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from rouge_score import rouge_scorer  # Import ROUGE scorer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def calculate_perplexity(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)  # Move inputs to GPU
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return 2 ** loss  # Perplexity formula

def evaluate_text_metrics(tokenizer, model, input_csv, output_csv):
    # Move model to GPU
    model.to(device)
    
    # Load data from CSV file
    data = pd.read_csv(input_csv)
    
    # Extract columns
    dreams = data["Dream"]
    original_interpretations = data['Original_Interpretation']
    generated_interpretations = data['Generated_Interpretation']
    
    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Smoothing for BLEU
    smoother = SmoothingFunction()

    # Compute metrics for each row
    metrics_data = []
    for dream, original, generated in zip(dreams, original_interpretations, generated_interpretations):
        bleu = sentence_bleu([original.split()], generated.split(), smoothing_function=smoother.method1)
        perplexity = calculate_perplexity(generated, tokenizer, model)
        
        # Move BERTScore tensors to device
        bert_p, bert_r, bert_f = bert_score([generated], [original], lang='en', device=device)

        # Calculate ROUGE-L F1 Score
        rouge_scores = rouge.score(original, generated)
        rouge_l_f1 = rouge_scores['rougeL'].fmeasure  # Extract ROUGE-L F1 score

        metrics_data.append({
            "Dream": dream,
            'Original': original,
            'Generated': generated,
            'BLEU': bleu,
            'Perplexity': perplexity,
            'BERTScore': bert_f.mean().item(),  # Move tensor to CPU and extract value
            'ROUGE-L': rouge_l_f1  # Add ROUGE-L score
        })
    
    # Save results to the specified output CSV file
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(output_csv, index=False)
    
    print(f"Metrics saved to '{output_csv}'")
