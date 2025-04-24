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
You are an expert in linguistics. I will provide you with a taxonomy of operations that can
be performed on sentences. The taxonomy is composed of two set of strategies: the first set
contains the surface strategies and the second set contains the content strategies.
The surface strategies are categorized into 7 categories of operations: ”replacement”, ”deletion”,
”addition”, ”integration”, ”splitting”, ”move” and ”no transformation”. Here are the operations
contained in each of these 7 categories:

- Replacement:
(S1) Replace at punctuation level
(S2) Replace at word level
(S3) Replace at phrase level
(S4) Replace at clause level
(S5) Replace at sentence level

- Deletion:
(S6) Delete at punctuation level
(S7) Delete at word level
(S8) Delete at phrase level
(S9) Delete at clause level
(S10) Delete at sentence level

- Addition:
(S11) Add at punctuation level
(S12) Add at word level
(S13) Add at phrase level
(S14) Add at clause level
(S15) Add at sentence level

- Integration:
(S16) Integrate two sentences
(S17) Integrate more than two sentences

Splitting:
(S18) Split by phrase
(S19) Split by clause

- Move:
(S20) Move constituents
(S21) Move a sentence

- No transformation:
(S22) Use an identical sentence

The content strategies are categorized into 5 categories of operations: ”no content change”,
”content deletion”, ”content addition”, ”content change” and ”document-level adjustment”. Here
are the operations contained in each of these 5 categories:

 - No content change:
(C1) Transform syntactic structure
(C2) Paraphrase into an abbreviation
(C3) Paraphrase into a non-abbreviation
(C4) Paraphrase into standard form
(C5) Remain unchanged

- Content deletion:
(C6) Delete introduction / conclusion
(C7) Delete a parallel element
(C8) Delete information for cohesion
(C9) Delete a modifier
(C10) Delete important information
(C11) Delete detail / extra information

- Content addition:
(C12) Add introduction / conclusion
(C13) Add a parallel element
(C14) Add contextual information
(C15) Add information for cohesion
(C16) Add a modifier
(C17) Add detail / extra information

- Content change:
(C18) Change aspect
(C19) Change modality
(C20) Paraphrase into a similar phrase
(C21) Paraphrase into an explanatory expression
(C22) Paraphrase into a direct expression
(C23) Paraphrase into a brief expression
(C24) Paraphrase into a concrete expression
(C25) Paraphrase into an essential point
(C26) Paraphrase into a different view

Document-level adjustment:
(C27) Change information flow
(C28) Delete for adjustment
(C29) Add for adjustment
(C30) Paraphrase for adjustment

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
        "S1 Replace at punctuation level": {{ "applied": true/false, "details": ["Specific S1 Replace changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S2 Replace at word level": {{ "applied": true/false, "details": ["Specific S2 Replace changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S3 Replace at phrase level": {{ "applied": true/false, "details": ["Specific S3 Replace changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S4 Replace at clause level": {{ "applied": true/false, "details": ["Specific S4 Replace changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S5 Replace at sentence level": {{ "applied": true/false, "details": ["Specific S5 Replace changes wrritten all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S6 Delete at punctuation level": {{ "applied": true/false, "details": ["Specific S6 Delete changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S7 Delete at word level": {{ "applied": true/false, "details": ["Specific S7 Delete changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S8 Delete at phrase level": {{ "applied": true/false, "details": ["Specific S8 Delete changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S9 Delete at clause level": {{ "applied": true/false, "details": ["Specific S9 Delete changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S10 Delete at sentence level": {{ "applied": true/false, "details": ["Specific S10 Delete changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S11 Add at punctuation level": {{ "applied": true/false, "details": ["Specific S11 Add changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S12 Add at word level": {{ "applied": true/false, "details": ["Specific S12 Add changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S13 Add at phrase level": {{ "applied": true/false, "details": ["Specific S13 Add changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S14 Add at clause level": {{ "applied": true/false, "details": ["Specific S14 Add changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S15 Add at sentence level": {{ "applied": true/false, "details": ["Specific S15 Add changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S16 Integrate two sentences": {{ "applied": true/false, "details": ["Specific S16 Integrate changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S17 Integrate more than two sentences": {{ "applied": true/false, "details": ["Specific S17 Integrate changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S18 Split by phrase": {{ "applied": true/false, "details": ["Specific S18 Split changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S19 Split by clause": {{ "applied": true/false, "details": ["Specific S19 Split changes writtten all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S20 Move constituents": {{ "applied": true/false, "details": ["Specific S20 Move changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S21 Move a sentence": {{ "applied": true/false, "details": ["Specific S21 Move changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "S22 Use an identical sentence": {{ "applied": true/false, "details": ["Specific S22 No transformation changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C1 Transform syntactic structure": {{ "applied": true/false, "details": ["Specific C1 Transform changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C2 Paraphrase into an abbreviation": {{ "applied": true/false, "details": ["Specific C2 Paraphrase changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C3 Paraphrase into a non-abbreviation": {{ "applied": true/false, "details": ["Specific C3 Paraphrase changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C4 Paraphrase into standard form": {{ "applied": true/false, "details": ["Specific C4 Paraphrase changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C5 Remain unchanged": {{ "applied": true/false, "details": ["Specific C5 No change details written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C6 Delete introduction / conclusion": {{ "applied": true/false, "details": ["Specific C6 Delete changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C7 Delete a parallel element": {{ "applied": true/false, "details": ["Specific C7 Delete changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C8 Delete information for cohesion": {{ "applied": true/false, "details": ["Specific C8 Delete changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C9 Delete a modifier": {{ "applied": true/false, "details": ["Specific C9 Delete changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C10 Delete important information": {{ "applied": true/false, "details": ["Specific C10 Delete changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C11 Delete detail / extra information": {{ "applied": true/false, "details": ["Specific C11 Delete changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C12 Add introduction / conclusion": {{ "applied": true/false, "details": ["Specific C12 Add changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C13 Add a parallel element": {{ "applied": true/false, "details": ["Specific C13 Add changes writtten all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C14 Add contextual information": {{ "applied": true/false, "details": ["Specific C14 Add changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C15 Add information for cohesion": {{ "applied": true/false, "details": ["Specific C15 Add changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C16 Add a modifier": {{ "applied": true/false, "details": ["Specific C16 Add changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C17 Add detail / extra information": {{ "applied": true/false, "details": ["Specific C17 Add changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C18 Change aspect": {{ "applied": true/false, "details": ["Specific C18 Change changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C19 Change modality": {{ "applied": true/false, "details": ["Specific C19 Change changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C20 Paraphrase into a similar phrase": {{ "applied": true/false, "details": ["Specific C20 Paraphrase changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C21 Paraphrase into an explanatory expression": {{ "applied": true/false, "details": ["Specific C21 Paraphrase changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C22 Paraphrase into a direct expression": {{ "applied": true/false, "details": ["Specific C22 Paraphrase changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C23 Paraphrase into a brief expression": {{ "applied": true/false, "details": ["Specific C23 Paraphrase changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C24 Paraphrase into a concrete expression": {{ "applied": true/false, "details": ["Specific C24 Paraphrase changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C25 Paraphrase into an essential point": {{ "applied": true/false, "details": ["Specific C25 Paraphrase changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C26 Paraphrase into a different view": {{ "applied": true/false, "details": ["Specific C26 Paraphrase changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C27 Change information flow": {{ "applied": true/false, "details": ["Specific C27 Adjustment changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C28 Delete for adjustment": {{ "applied": true/false, "details": ["Specific C28 Adjustment changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C29 Add for adjustment": {{ "applied": true/false, "details": ["Specific C29 Adjustment changes written all on one line"], "count": number of times this category is applied in this pair of sentences }},
        "C30 Paraphrase for adjustment": {{ "applied": true/false, "details": ["Specific C30 Adjustment changes written all on one line"], "count": number of times this category is applied in this pair of sentences }}
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
    
    base_dir = r'C:/Users/raque/Desktop/LaCC_READABILITY/Readability/output_analysis_LLM_answers/LLM_answers/LLM_Y_analysis'

    # Create a specific directory for each model
    model_dir = os.path.join(base_dir, f'{llm_name}_Y_Output')
    os.makedirs(model_dir, exist_ok=True)

    # Define the output file path within the model-specific directory
    output_file = os.path.join(model_dir, f'{llm_name}_Y_output.csv')

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
            "S1 Replace at punctuation level", "S2 Replace at word level", "S3 Replace at phrase level",
            "S4 Replace at clause level", "S5 Replace at sentence level", "S6 Delete at punctuation level",
            "S7 Delete at word level", "S8 Delete at phrase level", "S9 Delete at clause level",
            "S10 Delete at sentence level", "S11 Add at punctuation level", "S12 Add at word level",
            "S13 Add at phrase level", "S14 Add at clause level", "S15 Add at sentence level",
            "S16 Integrate two sentences", "S17 Integrate more than two sentences", "S18 Split by phrase",
            "S19 Split by clause", "S20 Move constituents", "S21 Move a sentence", "S22 Use an identical sentence",
            "C1 Transform syntactic structure", "C2 Paraphrase into an abbreviation", "C3 Paraphrase into a non-abbreviation",
            "C4 Paraphrase into standard form", "C5 Remain unchanged", "C6 Delete introduction / conclusion",
            "C7 Delete a parallel element", "C8 Delete information for cohesion", "C9 Delete a modifier",
            "C10 Delete important information", "C11 Delete detail / extra information", "C12 Add introduction / conclusion",
            "C13 Add a parallel element", "C14 Add contextual information", "C15 Add information for cohesion",
            "C16 Add a modifier", "C17 Add detail / extra information", "C18 Change aspect", "C19 Change modality",
            "C20 Paraphrase into a similar phrase", "C21 Paraphrase into an explanatory expression",
            "C22 Paraphrase into a direct expression", "C23 Paraphrase into a brief expression",
            "C24 Paraphrase into a concrete expression", "C25 Paraphrase into an essential point",
            "C26 Paraphrase into a different view", "C27 Change information flow", "C28 Delete for adjustment",
            "C29 Add for adjustment", "C30 Paraphrase for adjustment"
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
