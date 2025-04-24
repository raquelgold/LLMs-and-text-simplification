import os
import json
import pandas as pd
import time
from groq import Groq
from openai import OpenAI
import google.generativeai as genai

# Set up API clients (unchanged)
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

# Simplified analysis prompt
def get_analysis_prompt(sample_text):
    prompt = f"""
Analyze the relationship between two versions of a text and determine which simplification strategy is used:

Deletion Only:
Information present in the advanced text is completely removed in the simplified version. No new words are added, and the remaining content is unchanged. Minor rewording (e.g., changing "though" to "but") should not be categorized as deletion.

Paraphrase Only:
All information is preserved but expressed differently in the simplified version. Minor word substitutions (e.g., "though" to "but") are acceptable here if they preserve meaning without removing or adding information.

Deletion + Paraphrase:
Some information is removed, and the remaining content is reworded. For this to apply, both conditions must be met:

A phrase, clause, or meaningful unit is omitted.
Remaining content is rephrased or rewritten.
Rules:

At most one category can be correct for each pair.
Changes like replacing a word with a synonym (e.g., "though" to "but") alone should not be classified as deletion or deletion + paraphrase.
When evaluating borderline cases, prioritize identifying information loss over minor syntactic or lexical changes.

Examples:

Deletion Only:
Advanced: "This article is a list of the 50 U.S. states and the District of Columbia ordered by population density."
Elementary: "This is a list of the 50 U.S. states, ordered by population density."
Analysis: Deletion Only
Explanation: Removes "and the District of Columbia" without rephrasing the remaining text.

Paraphrase Only:
Advanced: "In 2002, both Russia and China also had prison populations in excess of 1 million."
Elementary: "In 2002, both Russia and China also had over 1 million people in prison."
Analysis: Paraphrase Only
Explanation: Rewords "prison populations in excess of 1 million" as "over 1 million people in prison" while preserving all information.

Deletion + Paraphrase:
Advanced: "All adult Muslims, with exceptions for the infirm, are required to offer Salat prayers five times daily."
Elementary: "All adult Muslims should do Salat prayers five times a day."
Analysis: Deletion + Paraphrase
Explanation: Removes "with exceptions for the infirm" AND rephrases "are required to offer" as "should do" and "daily" as "a day."

Advanced Text: {sample_text['adv']}
Elementary Text: {sample_text['ele']}

Return only valid JSON in this format, with no additional text:
{{
    "analysis": {{
        "advanced": "Full advanced text",
        "elementary": "Full elementary text",
        "deletion_only": true/false,
        "paraphrase_only": true/false,
        "deletion_and_paraphrase": true/false,
        "explanation": "Brief explanation of why this classification was chosen"
    }}
}}

Ensure only one category is marked as true and the others as false.
"""
    return prompt

def get_llama_response(sample_text):
    prompt = get_analysis_prompt(sample_text)
    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes text simplifications."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=2000
    )
    return response.choices[0].message.content

def get_gpt4_response(sample_text, model_name):
    prompt = get_analysis_prompt(sample_text)
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes text simplifications."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=2000
    )
    return response.choices[0].message.content

def get_gemini_response(sample_text):
    prompt = get_analysis_prompt(sample_text)
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
    
# Modified result writing function
def write_result_to_csv(result, llm_name):
    base_dir = r'C:/Users/raque/Desktop/LaCC_READABILITY/Readability/output_analysis_LLM_answers/LLM_answers/output_csv'
    
    os.makedirs(base_dir, exist_ok=True)
    
    output_file = os.path.join(base_dir, f'{llm_name}_simplified_output.csv')
    
    rows = []
    if isinstance(result, dict) and 'error' in result:
        rows.append({
            'text_id': result.get('text_id', ''),
            'adv_id': result.get('adv_id', ''),
            'ele_id': result.get('ele_id', ''),
            'adv_text': '',
            'ele_text': '',
            'deletion_only': '',
            'paraphrase_only': '',
            'deletion_and_paraphrase': '',
            'explanation': result.get('error', 'Unknown error'),
            'error': result.get('error', 'Unknown error')
        })
    else:
        analysis = result['analysis']
        row = {
            'text_id': result['text_id'],
            'adv_id': result['adv_id'],
            'ele_id': result['ele_id'],
            'adv_text': analysis['advanced'],
            'ele_text': analysis['elementary'],
            'deletion_only': 'Yes' if analysis['deletion_only'] else 'No',
            'paraphrase_only': 'Yes' if analysis['paraphrase_only'] else 'No',
            'deletion_and_paraphrase': 'Yes' if analysis['deletion_and_paraphrase'] else 'No',
            'explanation': analysis['explanation'],
            'error': ''
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, mode='a', header=False, index=False)

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

def process_text_pair(adv_text, ele_text, text_id, adv_id, ele_id, non_corresponding):
    sample_text = {"adv": adv_text, "ele": ele_text}
    
    # Define model configurations
    models = [
        ('llama', lambda x: get_llama_response(x)),
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
                write_result_to_csv(parsed_result, model_name)
                print(f"Processed and wrote results for {model_name} - text_id: {text_id}, adv_id: {adv_id}, ele_id: {ele_id}")
            else:
                error_message = {
                    'text_id': text_id,
                    'adv_id': adv_id,
                    'ele_id': ele_id,
                    'error': f"Failed to parse {model_name} result",
                    'non_corresponding': non_corresponding
                }
                write_result_to_csv(error_message, model_name)
                print(f"Failed to parse {model_name} result for text_id: {text_id}, adv_id: {adv_id}, ele_id: {ele_id}")
        
        except Exception as e:
            error_message = {
                'text_id': text_id,
                'adv_id': adv_id,
                'ele_id': ele_id,
                'error': str(e),
                'non_corresponding': non_corresponding
            }
            write_result_to_csv(error_message, model_name)
            print(f"Error processing {model_name} for text_id: {text_id}, adv_id: {adv_id}, ele_id: {ele_id}: {str(e)}")
        
        # Add delay between API calls to respect rate limits
        time.sleep(5)

if __name__ == "__main__":
    print("Running simplified text analysis script using multiple LLM models:")
    print("- LLaMA 3.1 70B Versatile")
    print("- Mistral Instruct 7B")
    print("- GPT-4o")
    print("- GPT-4o Mini")
    print("- Gemini 1.5 Flash")
    
    aligned_data_path = 'C:/Users/raque/Desktop/LaCC_READABILITY/Readability/src/Alignment_Sentences/data/aligned_sentences_with_NA.csv'
    
    for text_id, adv_id, ele_id, adv_text, ele_text, non_corresponding in process_aligned_data(aligned_data_path):
        print(f"\nProcessing text_id: {text_id}, adv_id: {adv_id}, ele_id: {ele_id}")
        
        process_text_pair(adv_text, ele_text, text_id, adv_id, ele_id, non_corresponding)
        
        # Add delay between processing different text pairs
        time.sleep(5)
    
    print("\nScript execution completed.")