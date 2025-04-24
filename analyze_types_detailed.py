import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_llm_data(base_dir):
    """
    Load CSV files for each LLM from the specified directory.
    """
    llm_data = {}
    expected_files = [
        'llama_detailed_output.csv',
        'gpt4o_detailed_output.csv',
        'gpt4-mini_detailed_output.csv',
        'gemini_detailed_output.csv'
    ]
    
    print(f"Loading files from: {base_dir}")
    for filename in expected_files:
        if os.path.exists(os.path.join(base_dir, filename)):
            llm_name = filename.replace('_detailed_output.csv', '')
            file_path = os.path.join(base_dir, filename)
            print(f"Loading {filename}...")
            df = pd.read_csv(file_path)
            # Remove rows with parsing errors or empty data
            df = df.dropna(subset=['text_id', 'adv_text'])
            llm_data[llm_name] = df
            print(f"Successfully loaded {filename}")
        else:
            print(f"Warning: {filename} not found in {base_dir}")
    
    return llm_data

def calculate_change_percentages(llm_data):
    """
    Calculate the percentage of sentences with each change type for each LLM.
    For each change type, counts number of 'yes' occurrences and divides by total sentences.
    """
    change_types = [
        'vocabulary', 'syntax', 'rephrasing', 'sentence_splitting', 
        'sentence_merging', 'thematic_shift', 'deletion', 'addition', 
        'figurative_language_simplification', 'elaboration_clarification', 
        'personalization', 'connective_changes'
    ]
    
    change_percentages = []
    
    for llm_name, df in llm_data.items():
        print(f"\nCalculating for LLM: {llm_name}")
        llm_percentages = {'LLM': llm_name}
        total_sentences = len(df)
        print(f"Total sentences: {total_sentences}")
        
        for change_type in change_types:
            # Count how many times this change type was applied (count 'yes' occurrences)
            applied_column = f'{change_type}_applied'
            yes_count = (df[applied_column].str.lower() == 'yes').sum()
            
            # Print the count of 'yes' occurrences
            print(f"Change Type: {change_type}, Yes Count: {yes_count}")
            
            # Calculate percentage
            percentage = (yes_count / total_sentences) * 100
            print(f"Change Type: {change_type}, Percentage: {percentage:.2f}%")
            llm_percentages[change_type] = percentage
        
        change_percentages.append(llm_percentages)
    
    return pd.DataFrame(change_percentages)

def plot_llm_comparison(change_percentages, output_dir):
    """Create a bar plot comparing change percentages across LLMs."""
    plt.figure(figsize=(15, 8))
    change_types = [col for col in change_percentages.columns if col != 'LLM']
    
    x = np.arange(len(change_types))
    width = 0.2
    
    for i, (idx, row) in enumerate(change_percentages.iterrows()):
        llm = row['LLM']
        percentages = [row[ct] for ct in change_types]
        plt.bar(x + i * width - (width * 1.5), percentages, width, label=llm)
    
    plt.xlabel('Change Types')
    plt.ylabel('Percentage of Sentences')
    plt.title('Percentage of Sentences with Each Change Type')
    plt.ylim(0, 100)  # Set y-axis from 0 to 100
    plt.xticks(x, change_types, rotation=45, ha='right')
    plt.legend(title='LLM', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add explanation text
    explanation = (
        "This plot shows what percentage of sentences had each type of change applied.\n"
        "Calculated as: (number of sentences with the change / total number of sentences) × 100"
    )
    plt.figtext(0.99, -0.15, explanation, ha='right', va='center', wrap=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llm_comparison.png'), bbox_inches='tight', 
                dpi=300, pad_inches=0.5)
    plt.close()


def plot_avg_changes_per_sentence(llm_data, output_dir):
    """
    Create a bar plot showing the average number of changes per sentence 
    based on the 'count' columns for each change type.
    """
    change_types = [
        'vocabulary', 'syntax', 'rephrasing', 'sentence_splitting', 
        'sentence_merging', 'thematic_shift', 'deletion', 'addition', 
        'figurative_language_simplification', 'elaboration_clarification', 
        'personalization', 'connective_changes'
    ]
    
    avg_changes = []
    
    for llm_name, df in llm_data.items():
        print(f"\nCalculating average changes for LLM: {llm_name}")
        llm_averages = {'LLM': llm_name}
        total_sentences = len(df)
        print(f"Total sentences: {total_sentences}")
        
        for change_type in change_types:
            count_column = f'{change_type}_count'
            # Sum the count column and divide by total sentences to get the average
            total_count = pd.to_numeric(df[count_column], errors='coerce').fillna(0).sum()
            
            # Print the total count for this change type
            print(f"Change Type: {change_type}, Total Count: {total_count}")
            
            average_changes = total_count / total_sentences
            print(f"Change Type: {change_type}, Average Changes per Sentence: {average_changes:.2f}")
            llm_averages[change_type] = average_changes
        
        avg_changes.append(llm_averages)
    
    avg_df = pd.DataFrame(avg_changes)
    
    plt.figure(figsize=(15, 8))
    x = np.arange(len(change_types))
    width = 0.2
    
    for i, (idx, row) in enumerate(avg_df.iterrows()):
        llm = row['LLM']
        averages = [row[ct] for ct in change_types]
        plt.bar(x + i * width - (width * 1.5), averages, width, label=llm)
    
    plt.xlabel('Change Types')
    plt.ylabel('Average Changes per Sentence')
    plt.title('Average Changes per Sentence by Type')
    plt.ylim(0, max(avg_df[change_types].max()) + 1)  # Set y-axis with an appropriate range
    plt.xticks(x, change_types, rotation=45, ha='right')
    plt.legend(title='LLM', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add explanation text
    explanation = (
        "This plot shows the average number of times each change type was applied per sentence.\n"
        "Calculated as: (total count of each change type / total number of sentences)\n"
        "Higher values indicate more frequent application of that change type per sentence."
    )
    plt.figtext(0.99, -0.15, explanation, ha='right', va='center', wrap=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_changes_per_sentence_bar_plot.png'), 
                bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close()

def create_changes_heatmap(llm_data, output_dir):
    """Create heatmaps showing changes per text ID."""
    change_types = [
        'vocabulary', 'syntax', 'rephrasing', 'sentence_splitting', 
        'sentence_merging', 'thematic_shift', 'deletion', 'addition', 
        'figurative_language_simplification', 'elaboration_clarification', 
        'personalization', 'connective_changes'
    ]
    
    for llm_name, df in llm_data.items():
        # Create matrix of change frequencies per text_id
        change_matrix = pd.DataFrame(index=df['text_id'].unique())
        
        for change_type in change_types:
            column = f'{change_type}_applied'
            # Convert to percentage
            change_matrix[change_type] = df.groupby('text_id')[column].apply(
                lambda x: (x.str.lower() == 'yes').mean() * 100
            )
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(change_matrix, cmap='YlOrRd', annot=True, fmt='.1f',
                   cbar_kws={'label': 'Percentage of Sentences with Changes'},
                   vmin=0, vmax=100)  # Set scale from 0 to 100
        plt.title(f'{llm_name} Changes by Text ID')
        plt.xlabel('Change Types')
        plt.ylabel('Text ID')
        plt.xticks(rotation=45, ha='right')
        
        # Add explanation text
        explanation = (
            f"This heatmap shows the percentage of sentences in each text that underwent specific changes by {llm_name}.\n"
            "Each row represents a different text, and each column represents a type of change.\n"
            "Values range from 0% (no sentences changed) to 100% (all sentences changed).\n"
            "Darker colors indicate a higher percentage of sentences affected by that change type."
        )
        plt.figtext(0.99, -0.05, explanation, ha='right', va='center', wrap=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{llm_name}_changes_per_text_id_heatmap.png'),
                    bbox_inches='tight', dpi=300, pad_inches=0.5)
        plt.close()

def create_changes_per_paragraph_heatmap(llm_data, output_dir):
    """
    Create heatmaps showing number of changes per paragraph.
    Maintains the original order of text_id in the CSV.
    """
    change_types = [
        'vocabulary', 'syntax', 'rephrasing', 'sentence_splitting', 
        'sentence_merging', 'thematic_shift', 'deletion', 'addition', 
        'figurative_language_simplification', 'elaboration_clarification', 
        'personalization', 'connective_changes'
    ]
    
    for llm_name, df in llm_data.items():
        # Prepare data for heatmap
        paragraph_changes = []
        
        # Iterate through unique text_id in the order they appear
        for text_id in df['text_id'].unique():
            # Select rows for this text_id
            paragraph_group = df[df['text_id'] == text_id]
            
            paragraph_row = {'text_id': text_id}
            
            # Count changes for each change type in this paragraph
            for change_type in change_types:
                count_column = f'{change_type}_count'
                
                # Sum the count of changes for this text_id
                paragraph_row[change_type] = (
                    pd.to_numeric(paragraph_group[count_column], errors='coerce')
                    .fillna(0)
                    .sum()
                )
            
            paragraph_changes.append(paragraph_row)
        
        # Convert to DataFrame
        paragraph_changes_df = pd.DataFrame(paragraph_changes)
        
        # Drop columns with only zeros
        columns_to_plot = [col for col in change_types if paragraph_changes_df[col].sum() > 0]
        
        # Select data to plot
        plot_data = paragraph_changes_df.set_index('text_id')[columns_to_plot]
        
        # Plot heatmap
        plt.figure(figsize=(15, 10))
        sns.heatmap(plot_data, cmap='YlOrRd', annot=True, fmt='.1f',
                   cbar_kws={'label': 'Number of Changes per Paragraph'},
                   vmin=0)
        plt.title(f'{llm_name} Changes per Paragraph')
        plt.xlabel('Change Types')
        plt.ylabel('Text ID (Original Order)')
        plt.xticks(rotation=45, ha='right')
        
        # Add explanation text
        explanation = (
            f"This heatmap shows the number of changes per paragraph by {llm_name}.\n"
            "Each cell represents the total number of changes for a specific change type in a paragraph.\n"
            "Darker colors indicate a higher number of changes.\n"
            "Rows represent texts in their original CSV order, columns represent change types."
        )
        plt.figtext(0.99, -0.05, explanation, ha='right', va='center', wrap=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{llm_name}_changes_per_paragraph_heatmap.png'),
                    bbox_inches='tight', dpi=300, pad_inches=0.5)
        plt.close()

def create_changes_per_sentence_heatmap(llm_data, output_dir):
    """
    Create heatmaps showing number of changes per sentence.
    Maintains the original order of the CSV.
    """
    change_types = [
        'vocabulary', 'syntax', 'rephrasing', 'sentence_splitting', 
        'sentence_merging', 'thematic_shift', 'deletion', 'addition', 
        'figurative_language_simplification', 'elaboration_clarification', 
        'personalization', 'connective_changes'
    ]
    
    for llm_name, df in llm_data.items():
        # Create a copy of the dataframe to preserve original order
        df_copy = df.copy()
        
        # Prepare data for heatmap
        sentence_changes = []
        
        # Iterate through the original dataframe rows
        for idx, row in df_copy.iterrows():
            sentence_row = {
                'index': idx,
                'text_id': row['text_id'], 
                'adv_id': row['adv_id']
            }
            
            # Count changes for each change type
            for change_type in change_types:
                count_column = f'{change_type}_count'
                
                # Convert to numeric, handling any non-numeric values
                sentence_row[change_type] = (
                    pd.to_numeric(row[count_column], errors='coerce')
                    or 0
                )
            
            sentence_changes.append(sentence_row)
        
        # Convert to DataFrame
        sentence_changes_df = pd.DataFrame(sentence_changes)
        
        # Drop columns with only zeros
        columns_to_plot = [col for col in change_types if sentence_changes_df[col].sum() > 0]
        
        # Select data to plot
        plot_data = sentence_changes_df.set_index(['index', 'text_id', 'adv_id'])[columns_to_plot]
        
        # Plot heatmap
        plt.figure(figsize=(15, 12))
        sns.heatmap(plot_data, cmap='YlOrRd', annot=True, fmt='.1f',
                   cbar_kws={'label': 'Number of Changes per Sentence'},
                   vmin=0)
        plt.title(f'{llm_name} Changes per Sentence')
        plt.xlabel('Change Types')
        plt.ylabel('Original Index, Text ID, Sentence ID')
        plt.xticks(rotation=45, ha='right')
        
        # Add explanation text
        explanation = (
            f"This heatmap shows the number of changes per sentence by {llm_name}.\n"
            "Each cell represents the total number of changes for a specific change type in a sentence.\n"
            "Darker colors indicate a higher number of changes.\n"
            "Rows represent sentences in their original CSV order."
        )
        plt.figtext(0.99, -0.05, explanation, ha='right', va='center', wrap=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{llm_name}_changes_per_sentence_heatmap.png'),
                    bbox_inches='tight', dpi=300, pad_inches=0.5)
        plt.close()

def plot_change_type_statistics(llm_data, output_dir):
    """Create statistical plots for each change type."""
    change_types = [
        'vocabulary', 'syntax', 'rephrasing', 'sentence_splitting', 
        'sentence_merging', 'thematic_shift', 'deletion', 'addition', 
        'figurative_language_simplification', 'elaboration_clarification', 
        'personalization', 'connective_changes'
    ]
    
    for llm_name, df in llm_data.items():
        # Calculate statistics as percentages
        stats = []
        for change_type in change_types:
            column = f'{change_type}_applied'
            values = (df[column].str.lower() == 'yes').astype(float) * 100
            stats.append({
                'Change Type': change_type,
                'Mean': values.mean(),
                'Median': values.median(),
                'Std Dev': values.std()
            })
        
        stats_df = pd.DataFrame(stats)
        
        # Create plot
        plt.figure(figsize=(15, 8))
        x = np.arange(len(change_types))
        width = 0.25
        
        plt.bar(x - width, stats_df['Mean'], width, label='Mean', color='blue', alpha=0.7)
        plt.bar(x, stats_df['Median'], width, label='Median', color='green', alpha=0.7)
        plt.bar(x + width, stats_df['Std Dev'], width, label='Std Dev', color='red', alpha=0.7)
        
        plt.xlabel('Change Types')
        plt.ylabel('Percentage')
        plt.title(f'{llm_name} Change Type Statistics')
        plt.ylim(0, 100)  # Set y-axis from 0 to 100
        plt.xticks(x, change_types, rotation=45, ha='right')
        plt.legend()
        
        # Add explanation text
        explanation = (
            f"Statistics for {llm_name} showing the distribution of changes across sentences.\n"
            "Mean: Average percentage of sentences with each change type\n"
            "Median: Middle value of the percentage distribution\n"
            "Std Dev: Spread of the percentage values around the mean"
        )
        plt.figtext(0.99, -0.15, explanation, ha='right', va='center', wrap=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{llm_name}_change_type_statistics_plot.png'), 
                    bbox_inches='tight', dpi=300, pad_inches=0.5)
        plt.close()

def plot_correlation_matrix(llm_data, output_dir):
    """Create correlation matrix heatmaps for change types."""
    change_types = [
        'vocabulary', 'syntax', 'rephrasing', 'sentence_splitting', 
        'sentence_merging', 'thematic_shift', 'deletion', 'addition', 
        'figurative_language_simplification', 'elaboration_clarification', 
        'personalization', 'connective_changes'
    ]
    
    for llm_name, df in llm_data.items():
        # Create correlation matrix
        corr_data = pd.DataFrame()
        for change_type in change_types:
            column = f'{change_type}_applied'
            corr_data[change_type] = (df[column].str.lower() == 'yes').astype(float)
        
        correlation_matrix = corr_data.corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, fmt='.2f')
        plt.title(f'{llm_name} Change Type Correlations')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add explanation text
        explanation = (
            f"This correlation matrix shows how different types of changes are related to each other in {llm_name}'s outputs.\n"
            "Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation):\n"
            "• +1 (dark red): Changes that tend to occur together\n"
            "• 0 (white): Changes that occur independently\n"
            "• -1 (dark blue): Changes that tend to not occur together\n"
            "The diagonal always shows 1.0 as each change type perfectly correlates with itself."
        )
        plt.figtext(0.99, -0.05, explanation, ha='right', va='center', wrap=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{llm_name}_change_type_correlation_matrix.png'),
                    bbox_inches='tight', dpi=300, pad_inches=0.5)
        plt.close()

def main():
    # Set directories
    base_input_dir = r'C:/Users/raque/Desktop/LaCC_READABILITY/Readability/output_analysis_LLM_answers/LLM_answers/output_csv'
    base_output_dir = r'C:/Users/raque/Desktop/LaCC_READABILITY/Readability/output_analysis_LLM_answers/LLM_answers/output_plots'
    
    # Create output directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load data
    llm_data = load_llm_data(base_input_dir)
    
    if not llm_data:
        print("Error: No data was loaded. Please check your input directory and file names.")
        return
    
    # Calculate change percentages
    change_percentages = calculate_change_percentages(llm_data)
    
    # Generate all plots
    print("\nGenerating visualizations...")
    
    print("1. Creating LLM comparison plot...")
    plot_llm_comparison(change_percentages, base_output_dir)
    
    print("2. Creating average changes per sentence plot...")
    plot_avg_changes_per_sentence(llm_data, base_output_dir)
    
    print("3. Creating changes per text ID heatmaps...")
    create_changes_heatmap(llm_data, base_output_dir)
    
    print("4. Creating change type statistics plots...")
    plot_change_type_statistics(llm_data, base_output_dir)
    
    print("5. Creating correlation matrices...")
    plot_correlation_matrix(llm_data, base_output_dir)

    print("6. Creating changes per paragraph heatmaps...")
    create_changes_per_paragraph_heatmap(llm_data, base_output_dir)
    
    print("7. Creating changes per sentence heatmaps...")
    create_changes_per_sentence_heatmap(llm_data, base_output_dir)
    
    print("\nAll visualizations have been generated successfully!")
    print(f"\nOutput files can be found in: {base_output_dir}")

if __name__ == "__main__":
    main()