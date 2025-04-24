import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    try:
        return pd.read_csv(file_path, 
                          on_bad_lines='warn',
                          low_memory=False)
    except pd.errors.ParserError as e:
        print(f"Error reading the CSV file: {e}")
        print("Attempting to read the file with more lenient settings...")
        return pd.read_csv(file_path, 
                          on_bad_lines='skip',
                          low_memory=False)

def calculate_simplification_stats(df):
    total_sentences = len(df)
    
    # Calculate individual changes
    deletion_only = (df['deletion_only'] == 'Yes').sum()
    paraphrase_only = (df['paraphrase_only'] == 'Yes').sum()
    both = (df['deletion_and_paraphrase'] == 'Yes').sum()
    
    # Calculate no change (sentences that had none of the above)
    has_any_change = ((df['deletion_only'] == 'Yes') | 
                     (df['paraphrase_only'] == 'Yes') | 
                     (df['deletion_and_paraphrase'] == 'Yes'))
    no_change = (~has_any_change).sum()
    
    # Calculate percentages
    stats = {
        'Deletion Only': {
            'count': deletion_only,
            'percentage': (deletion_only / total_sentences) * 100
        },
        'Paraphrase Only': {
            'count': paraphrase_only,
            'percentage': (paraphrase_only / total_sentences) * 100
        },
        'Deletion & Paraphrase': {
            'count': both,
            'percentage': (both / total_sentences) * 100
        },
        'No Change': {
            'count': no_change,
            'percentage': (no_change / total_sentences) * 100
        }
    }
    
    return stats, total_sentences

def plot_model_comparison(all_stats, output_dir):
    fig = plt.figure(figsize=(15, 8))
    
    # Create main axis for the bar plot
    ax = plt.subplot2grid((1, 4), (0, 0), colspan=3)
    
    # Get all percentages for plotting
    all_percentages = {model: {cat: model_stats[cat]['percentage'] 
                             for cat in model_stats.keys()} 
                      for model, model_stats in all_stats.items()}
    
    # Define categories and models
    categories = ['Deletion Only', 'Paraphrase Only', 'Deletion & Paraphrase', 'No Change']
    models = list(all_percentages.keys())
    
    # Set up the bars
    x = np.arange(len(categories))
    width = 0.15
    n_models = len(models)
    offsets = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)
    
    # Plot bars for each model
    for i, (model, percentages) in enumerate(all_percentages.items()):
        values = [percentages[cat] for cat in categories]
        bars = ax.bar(x + offsets[i], values, width, label=model)
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', rotation=0)
    
    # Customize the bar plot
    ax.set_ylabel('Percentage of Sentences')
    ax.set_title('Comparison of Change Types Across Language Models')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    
    # Add legend
    ax.legend(title='Language Models')
    
    # Add explanation text on the right side
    explanation = plt.figtext(0.76, 0.5, 
        "How to read this plot:\n\n"
        "1. Each group shows a type of change:\n"
        "   - Deletion Only\n"
        "   - Paraphrase Only\n"
        "   - Deletion & Paraphrase\n"
        "   - No Change\n\n"
        "2. Different colored bars represent\n"
        "   different language models\n\n"
        "3. The height of each bar shows the\n"
        "   percentage of sentences that\n"
        "   underwent that type of change\n\n"
        "4. Percentages are calculated as:\n"
        "   (count of change type / total\n"
        "   sentences) × 100\n\n"
        "5. The exact percentage is shown\n"
        "   on top of each bar",
        fontsize=9, 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
        va='center'
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def analyze_cross_model_categorization(base_path, output_base_dir):
    # Read all model output files
    model_files = [f for f in os.listdir(base_path) if f.endswith('_simplified_output.csv')]
    
    # Read all dataframes
    dfs = {model.replace('_simplified_output.csv', ''): pd.read_csv(os.path.join(base_path, model)) 
           for model in model_files}
    
    # Prepare a combined dataframe with model column
    combined_data = []
    for model, df in dfs.items():
        model_df = df.copy()
        model_df['model'] = model
        combined_data.append(model_df)
    
    combined_df = pd.concat(combined_data)
    
    # Group by text_id and find disagreements
    def find_disagreements(group):
        # Only consider rows with multiple models
        if len(group) < 2:
            return None
        
        # Categories to compare
        category_cols = ['deletion_only', 'paraphrase_only', 'deletion_and_paraphrase']
        
        # Check if categorizations differ across models
        unique_cats = group[category_cols].drop_duplicates()
        
        # If more than one unique categorization, return details
        if len(unique_cats) > 1:
            result = {
                'text_id': group['text_id'].iloc[0],
                'adv_text': group['adv_text'].iloc[0],
                'ele_text': group['ele_text'].iloc[0],
                'disagreeing_models': ', '.join(group['model'].unique()),
                'categorizations': group[['model'] + category_cols].to_dict('records')
            }
            return result
        
        return None
    
    # Find sentences with different categorizations-
    disagreements = combined_df.groupby(['text_id', 'adv_text']).apply(find_disagreements).dropna()
    
    # Convert to DataFrame and save
    if not disagreements.empty:
        disagreements_df = pd.DataFrame(disagreements.tolist())
        disagreements_path = os.path.join(output_base_dir, 'model_categorization_disagreements.csv')
        disagreements_df.to_csv(disagreements_path, index=False)
        
        print(f"Total sentences with different categorizations: {len(disagreements_df)}")
        print(f"Disagreements saved to: {disagreements_path}")
    else:
        print("No model categorization disagreements found.")

def main():
    # Define paths
    base_path = r'C:/Users/raque/Desktop/LaCC_READABILITY/Readability/output_analysis_LLM_answers/LLM_answers/output_csv'
    output_base_dir = r'C:/Users/raque/Desktop/LaCC_READABILITY/Readability/output_analysis_LLM_answers/LLM_answers/simplified_analysis'
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Define models to analyze
    models = ['llama', 'gpt4o', 'gpt4-mini', 'gemini']
    
    # Store stats for all models
    all_stats = {}
    
    # Analyze each model
    for model in models:
        input_file = os.path.join(base_path, f'{model}_simplified_output.csv')
        
        if os.path.exists(input_file):
            print(f"\nAnalyzing {model}:")
            df = load_data(input_file)
            stats, total_sentences = calculate_simplification_stats(df)
            all_stats[model] = stats
            
            # Print detailed statistics for each model
            print(f"Total sentences analyzed: {total_sentences}")
            print("\nBreakdown by change type:")
            for change_type, data in stats.items():
                print(f"{change_type}:")
                print(f"  Count: {data['count']} sentences")
                print(f"  Percentage: {data['percentage']:.1f}%")
        else:
            print(f"Warning: No output file found for {model}")
    
    # Generate the comparison plot
    if len(all_stats) > 1:
        print("\nGenerating comparison plot...")
        plot_model_comparison(all_stats, output_base_dir)
    
    print("\nAnalysis complete. Results saved in:", output_base_dir)

    print("\nPerforming cross-model categorization analysis...")
    analyze_cross_model_categorization(base_path, output_base_dir)
    
    print("\nAnalysis complete. Results saved in:", output_base_dir)

if __name__ == "__main__":
    main()

