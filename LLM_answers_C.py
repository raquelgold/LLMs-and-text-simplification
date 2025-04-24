import os
import json
import pandas as pd
import time
from groq import Groq
from openai import OpenAI
import google.generativeai as genai

# Set up API clients
def read_api_key(file_path):
    with open(file_path) as f:
        return f.read().strip()

GROQ_KEY_PATH = os.path.join('API_keys', 'groq_key.txt')
OPENAI_KEY_PATH = os.path.join('API_keys', 'open_ai_Readability_key.txt')
GEMINI_KEY_PATH = os.path.join('API_keys', 'gemini_api_key.txt')

GROQ_API_KEY = read_api_key(GROQ_KEY_PATH)
OPENAI_API_KEY = read_api_key(OPENAI_KEY_PATH)
GEMINI_API_KEY = read_api_key(GEMINI_KEY_PATH)

groq_client = Groq(api_key=GROQ_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Taxonomy-based prompt generator
def get_taxonomy_analysis_prompt(sample_text):
    return f"""
You are an expert in linguistics. 
I will provide you with a taxonomy of operations that can be performed on sentences. 
The taxonomy is composed of two sets of operations: computational operations and computational operation combinations.

Here are the operations contained in the set of computational operations:
(C1) Move
(C2) Insert proposition
(C3) Delete proposition
(C4) Insert modifier
(C5) Delete modifier
(C6) Insert for consistency
(C7) Delete for consistency
(C8) Insert other
(C9) Delete other
(C10) Replace with synonym
(C11) Replace with hyperonym
(C12) Replace with hyponym
(C13) Replace singular with plural
(C14) Replace plural with singular
(C15) Replace segment with a pronoun
(C16) Replace pronoun with its antecedent
(C17) Modify verbal features

Here are the operations contained in the set of computational operation combinations: 
(CC1) Active to passive
(CC2) Passive to active
(CC3) Part-of-speech change
(CC4) Split
(CC5) Merge
(CC6) To impersonal form
(CC7) To personal form
(CC8) Affirmation to negation
(CC9) Negation to affirmation

Given the above taxonomy, what are the operations used to transform:
Advanced sentence: "{adv_text}" into Elementary sentence: "{ele_text}"

Provide your analysis in JSON format, using the following structure. Remember, all string values MUST be on a single line without any line breaks.

Important formatting instructions:
- Ensure the JSON is valid and well-formed.
- Do not include any markdown formatting (e.g., no ```json markers).
- CRITICAL: Do not insert line breaks or newlines in any string values in the JSON. All strings MUST be continuous, single-line values.
- Ensure that every key-value pair is complete, with no trailing commas or missing values.
- Ensure that the JSON does not include empty details (e.g., remove empty entries in the "details" field if no change is present).
- If there is no elementary sentence corresponding to the advanced sentence, the response should be "omitted".

{{
  "analysis": {{
    "advanced": "Full advanced text on a single line",
    "elementary": "Full elementary text on a single line",
    "changes": {{
        "C1 Move": {{"applied": true/false, "details": ["Specific significant C1 Move changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C2 Insert proposition": {{"applied": true/false, "details": ["Specific significant C2 Insert proposition changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C3 Delete proposition": {{"applied": true/false, "details": ["Specific significant C3 Delete proposition changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C4 Insert modifier": {{"applied": true/false, "details": ["Specific significant C4 Insert modifier changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C5 Delete modifier": {{"applied": true/false, "details": ["Specific significant C5 Delete modifier changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C6 Insert for consistency": {{"applied": true/false, "details": ["Specific significant C6 Insert for consistency changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C7 Delete for consistency": {{"applied": true/false, "details": ["Specific significant C7 Delete for consistency changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C8 Insert other": {{"applied": true/false, "details": ["Specific significant C8 Insert other changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C9 Delete other": {{"applied": true/false, "details": ["Specific significant C9 Delete other changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C10 Replace with synonym": {{"applied": true/false, "details": ["Specific significant C10 Replace with synonym changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C11 Replace with hyperonym": {{"applied": true/false, "details": ["Specific significant C11 Replace with hyperonym changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C12 Replace with hyponym": {{"applied": true/false, "details": ["Specific significant C12 Replace with hyponym changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C13 Replace singular with plural": {{"applied": true/false, "details": ["Specific significant C13 Replace singular with plural changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C14 Replace plural with singular": {{"applied": true/false, "details": ["Specific significant C14 Replace plural with singular changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C15 Replace segment with a pronoun": {{"applied": true/false, "details": ["Specific significant C15 Replace segment with a pronoun changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C16 Replace pronoun with its antecedent": {{"applied": true/false, "details": ["Specific significant C16 Replace pronoun with its antecedent changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "C17 Modify verbal features": {{"applied": true/false, "details": ["Specific significant C17 Modify verbal features changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "CC1 Active to passive": {{"applied": true/false, "details": ["Specific significant CC1 Active to passive changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "CC2 Passive to active": {{"applied": true/false, "details": ["Specific significant CC2 Passive to active changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "CC3 Part-of-speech change": {{"applied": true/false, "details": ["Specific significant CC3 Part-of-speech change changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "CC4 Split": {{"applied": true/false, "details": ["Specific significant CC4 Split changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "CC5 Merge": {{"applied": true/false, "details": ["Specific significant CC5 Merge changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "CC6 To impersonal form": {{"applied": true/false, "details": ["Specific significant CC6 To impersonal form changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "CC7 To personal form": {{"applied": true/false, "details": ["Specific significant CC7 To personal form changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "CC8 Affirmation to negation": {{"applied": true/false, "details": ["Specific significant CC8 Affirmation to negation changes all on one line], "count": number of times this category was applied in this pair of sentences}},
        "CC9 Negation to affirmation": {{"applied": true/false, "details": ["Specific significant CC9 Negation to affirmation changes all on one line], "count": number of times this category was applied in this pair of sentences}}
    }}
  }}
}}

**Important**: 
- Analyze only the given transformation without introducing hypothetical examples.
- Provide just one JSON object per response.
- Check that total operations is the sum of all counts.
- Ensure all keys are present, even if counts are 0 or details are empty.
- Avoid redundant or additional analysis.
"""

def get_llama_response(sample_text):
    prompt = get_taxonomy_analysis_prompt(sample_text)
    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert in linguistics."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=2000
    )
    return response.choices[0].message.content

def get_gpt4_response(sample_text, model_name):
    prompt = get_taxonomy_analysis_prompt(sample_text)
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert in linguistics."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=2000
    )
    return response.choices[0].message.content

def get_gemini_response(sample_text):
    prompt = get_taxonomy_analysis_prompt(sample_text)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

def parse_json_response(analysis_result):
    cleaned_result = analysis_result.strip().replace('```json', '').replace('```', '').strip()
    try:
        analysis_data = json.loads(cleaned_result)
        print("JSON successfully parsed.")
        return analysis_data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print("Raw response:", cleaned_result)
        return None
    
def process_aligned_data(file_path):
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        text_id = row['text_id']
        adv_idx = row['Adv_idx'] if pd.notna(row['Adv_idx']) else None
        ele_idx = row['Ele_idx'] if pd.notna(row['Ele_idx']) else None
        adv_text = row['Text Adv Sentence'] if pd.notna(row['Text Adv Sentence']) else ''
        ele_text = row['Text Ele Sentence'] if pd.notna(row['Text Ele Sentence']) else ''
        
        non_corresponding = {
            "adv": [adv_text] if pd.isna(row['Ele_idx']) and pd.notna(row['Text Adv Sentence']) else [],
            "ele": [ele_text] if pd.isna(row['Adv_idx']) and pd.notna(row['Text Ele Sentence']) else []
        }
        
        yield text_id, adv_idx, ele_idx, adv_text, ele_text, non_corresponding

def write_taxonomy_result_to_csv(result, llm_name):
    
    base_dir = r'C:/Users/raque/Desktop/LaCC_READABILITY/Readability/output_analysis_LLM_answers/LLM_answers/LLM_C_analysis'

    # Create a specific directory for each model
    model_dir = os.path.join(base_dir, f'{llm_name}_C_Output')
    os.makedirs(model_dir, exist_ok=True)

    # Define the output file path within the model-specific directory
    output_file = os.path.join(model_dir, f'{llm_name}_C_output.csv')

    rows = []
    if isinstance(result, dict) and 'error' in result:  # Error message
        rows.append({
            'text_id': result.get('text_id', ''),
            'adv_id': result.get('adv_id', ''),
            'adv_text': '',
            'ele_id': result.get('ele_id', ''),
            'ele_text': '',
            'error': result.get('error', 'Unknown error'),
            'non_corresponding': 'No',
            'non_corresponding_details': ''
        })
    else:
        analysis = result['analysis']
        
        # Combine non-corresponding sentence information
        non_corresponding_status = 'Yes' if result['non_corresponding']['adv'] or result['non_corresponding']['ele'] else 'No'
        non_corresponding_details = []
        if result['non_corresponding']['adv']:
            non_corresponding_details.extend([f"Deleted: {sent}" for sent in result['non_corresponding']['adv']])
        if result['non_corresponding']['ele']:
            non_corresponding_details.extend([f"Added: {sent}" for sent in result['non_corresponding']['ele']])
        
        row = {
            'text_id': result['text_id'],
            'adv_id': result['adv_id'],
            'adv_text': analysis['advanced'],
            'ele_id': result['ele_id'],
            'ele_text': analysis['elementary'],
            'error': '',
            'non_corresponding': non_corresponding_status,
            'non_corresponding_details': '; '.join(non_corresponding_details)
        }
        
        change_types = [
            "C1 Move", "C2 Insert proposition", "C3 Delete proposition", "C4 Insert modifier", "C5 Delete modifier",
            "C6 Insert for consistency", "C7 Delete for consistency", "C8 Insert other", "C9 Delete other",
            "C10 Replace with synonym", "C11 Replace with hyperonym", "C12 Replace with hyponym",
            "C13 Replace singular with plural", "C14 Replace plural with singular", "C15 Replace segment with a pronoun",
            "C16 Replace pronoun with its antecedent", "C17 Modify verbal features",
            "CC1 Active to passive", "CC2 Passive to active", "CC3 Part-of-speech change", "CC4 Split", "CC5 Merge",
            "CC6 To impersonal form", "CC7 To personal form", "CC8 Affirmation to negation", "CC9 Negation to affirmation"
        ]
        
        for change_type in change_types:
            change_data = analysis['changes'].get(change_type, {"applied": False, "details": [], "count": 0})
            row[f'{change_type}_applied'] = 'Yes' if change_data['applied'] else 'No'
            row[f'{change_type}_details'] = '; '.join(change_data['details']) if change_data['applied'] else ''
            row[f'{change_type}_count'] = change_data['count']  # Add the count column for each change type
    
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, mode='a', header=False, index=False)

# Main processing function
def process_text_pair(adv_text, ele_text, text_id, adv_id, ele_id, non_corresponding):
    sample_text = {"adv": adv_text, "ele": ele_text}
    
    # Define model configurations
    models = [
        ('llama', lambda x: get_llama_response(x)),
        # ('mistral', lambda x: get_mistral_response(x)),
        ('gpt4o', lambda x: get_gpt4_response(x, "gpt-4o")),
        ('gpt4-mini', lambda x: get_gpt4_response(x, "gpt-4o-mini")),
        ('gemini', lambda x: get_gemini_response(x))
    ]
    
    for model_name, get_response in models:
        try:
            print(f"Processing with {model_name}...")
            result = get_response(sample_text)
            parsed_result = parse_json_response(result)
            
            if parsed_result:
                parsed_result['text_id'] = text_id
                parsed_result['adv_id'] = adv_id
                parsed_result['ele_id'] = ele_id
                parsed_result['non_corresponding'] = non_corresponding
                write_taxonomy_result_to_csv(parsed_result, model_name)
                print(f"Processed and wrote results for {model_name} - text_id: {text_id}, adv_id: {adv_id}, ele_id: {ele_id}")
            else:
                error_message = {
                    'text_id': text_id,
                    'adv_id': adv_id,
                    'ele_id': ele_id,
                    'error': f"Failed to parse {model_name} result",
                    'non_corresponding': non_corresponding
                }
                write_taxonomy_result_to_csv(error_message, model_name)
                print(f"Failed to parse {model_name} result for text_id: {text_id}, adv_id: {adv_id}, ele_id: {ele_id}")
        
        except Exception as e:
            error_message = {
                'text_id': text_id,
                'adv_id': adv_id,
                'ele_id': ele_id,
                'error': str(e),
                'non_corresponding': non_corresponding
            }
            write_taxonomy_result_to_csv(error_message, model_name)
            print(f"Error processing {model_name} for text_id: {text_id}, adv_id: {adv_id}, ele_id: {ele_id}: {str(e)}")
        
        # Add delay between API calls to respect rate limits
        time.sleep(5)


if __name__ == "__main__":
    print("Running text analysis script using multiple LLM models:")
    print("- LLaMA 3.1 70B Versatile")
    print("- GPT-4o")
    print("- GPT-4o Mini")
    print("- Gemini 1.5 Flash")
    
    aligned_data_path = 'C:/Users/raque/Desktop/LaCC_READABILITY/Readability/src/Alignment_Sentences/data/aligned_sentences_with_NA.csv'
    
    # Read the DataFrame
    df = pd.read_csv(aligned_data_path)
    
    # Randomly sample 200 rows
    sampled_df = df.sample(n=200, random_state=42)  # random_state ensures reproducibility
    
    # Modify process_aligned_data to work with the sampled DataFrame
    def process_sampled_data(sampled_dataframe):
        for _, row in sampled_dataframe.iterrows():
            text_id = row['text_id']
            adv_idx = row['Adv_idx'] if pd.notna(row['Adv_idx']) else None
            ele_idx = row['Ele_idx'] if pd.notna(row['Ele_idx']) else None
            adv_text = row['Text Adv Sentence'] if pd.notna(row['Text Adv Sentence']) else ''
            ele_text = row['Text Ele Sentence'] if pd.notna(row['Text Ele Sentence']) else ''
            
            non_corresponding = {
                "adv": [adv_text] if pd.isna(row['Ele_idx']) and pd.notna(row['Text Adv Sentence']) else [],
                "ele": [ele_text] if pd.isna(row['Adv_idx']) and pd.notna(row['Text Ele Sentence']) else []
            }
            
            yield text_id, adv_idx, ele_idx, adv_text, ele_text, non_corresponding
    
    for text_id, adv_id, ele_id, adv_text, ele_text, non_corresponding in process_sampled_data(sampled_df):
        print(f"\nProcessing text_id: {text_id}, adv_id: {adv_id}, ele_id: {ele_id}")
        
        process_text_pair(adv_text, ele_text, text_id, adv_id, ele_id, non_corresponding)
        
        # Add delay between processing different text pairs
        time.sleep(5)
    
    print("\nScript execution completed.")
