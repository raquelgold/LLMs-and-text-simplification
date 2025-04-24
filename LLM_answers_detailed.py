import os
import json
import pandas as pd
import time
from groq import Groq
from openai import OpenAI
import google.generativeai as genai
import re

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

# Analysis prompt (unchanged)
def get_analysis_prompt(sample_text):
    prompt = f"""
Analyze the differences between the following advanced and elementary versions of a text with precision and accuracy. Focus only on clear, significant changes. Avoid over-analysis or identifying changes that are not explicitly present. Provide a concise analysis addressing the following aspects:

1. Vocabulary Changes: 
   - Identify only clear, one-word substitutions or small phrase changes (up to three words).
   - Present changes in the format "word/phrase A → word/phrase B".
   - Only include changes that significantly alter the meaning or complexity of the text.
   - If a word or phrase in the advanced text is entirely omitted in the elementary text, categorize it under "Deletion" rather than vocabulary change.
   - Be very precise: if a word is part of a larger rephrasing or deletion, it should not be listed here.
   - Examples:
     - Correct: "utilize → use"
     - Incorrect: "quickly ran → moved fast" (this is rephrasing, not a vocabulary change)
     - Incorrect: "meticulous → omitted" (this is a deletion, not a vocabulary change)

2. Syntax Changes:
   - Note only major alterations in sentence structure that affect readability or complexity.
   - Focus on specific structural changes, such as:
     a) Shifts in verb tense (e.g., from past to present, or future to past).
     b) Changes in voice (from passive to active or active to passive).
   - **Important**: Distinguish between verb tense changes and changes in voice (active/passive). 
     - Only mark a change as "voice" if the sentence structure has been altered to change the subject and object roles (i.e., who is performing the action).
     - Example: Passive voice in "The project was completed by the team" changed to active voice in "The team completed the project."
     - **Incorrect**: "badgered... into promising" → "asked... to install" (this is a change in verb and structure, not a passive-to-active shift).
   - Count tense changes (without a shift in subject/object roles) as separate from voice changes and note them explicitly as "verb tense shifts."
   - Identify cases where passive voice constructions in the advanced text are changed to active voice in the elementary version or vice versa, but ensure these cases involve changes in the grammatical roles of subject and object.
   - If quoted phrases in the advanced text are presented without quotation marks in the elementary text, consider this a structural change, but avoid classifying it as a change in voice.
   - Ignore minor changes in word order unless they significantly impact the meaning.
   - Do not confuse syntax changes with rephrasing or connective changes.
   - Count each change made to sentence structure, verb tense, or voice as a separate instance, ensuring only clear and impactful changes are counted.

3. Rephrasing: 
   - Identify significant rewording of phrases or sentences that alter the style or complexity of the text.
   - Focus on changes that affect more than half of a sentence.
   - If a phrase in the advanced text is completely omitted in the elementary text, consider it a deletion instead of rephrasing.
   - Examples:
     - Correct: "The project was completed ahead of schedule → We finished the project early"
     - Incorrect: "The intricate analysis" → omitted (this is a deletion, not rephrasing)

4. Sentence Splitting: 
   - Identify only clear instances where one complex advanced sentence is divided into two or more simpler sentences in the elementary version.
   - Confirm that the advanced version has a single sentence and the elementary version has multiple sentences conveying the same content.
   - Avoid counting cases as "splits" when the number of sentences in both versions is the same.
   - If both the advanced and elementary texts contain a single sentence, do not mark it as a split. Instead, look for other structural changes like rephrasing or syntax.
   - Count each split as a separate instance only when verified that one complex sentence was transformed into multiple simpler sentences.

5. Sentence Merging: 
   - Identify only clear instances where multiple advanced sentences are combined into a single elementary sentence.
   - Confirm that the advanced version has multiple sentences and that the elementary version combines them into one.
   - Avoid counting cases as "merges" when the number of sentences in both versions is the same.
   - If both the advanced and elementary texts contain a single sentence, do not mark it as a merge. Instead, look for other structural changes like rephrasing or syntax.
   - Count each merge as a separate instance only when verified that multiple sentences were transformed into a single sentence.

6. Thematic Shift: 
   - Note only major changes in focus, perspective, or subject between the advanced and elementary versions.
   - This should involve a clear shift in the main topic or point of view.
   - Count each thematic shift as a separate instance.

7. Deletion: 
   - Identify significant information, phrases, or words present in the advanced text that are entirely omitted in the elementary version.
   - If a vocabulary item or phrase is completely missing in the elementary text, treat it as a deletion rather than a vocabulary change or rephrasing.
   - Only include deletions where no equivalent information is present in the elementary text.
   - Examples:
     - Correct: "The intricate process of photosynthesis involves... [detailed explanation]" → [No mention of photosynthesis at all]
     - Correct: "meticulous → omitted" (since the word is entirely missing in the elementary text)
     - Incorrect: "The intricate process of photosynthesis" → "How plants make food" (this is rephrasing, not deletion)

8. Addition: 
   - Identify significant new information or explanations in the elementary text that are not present in any form in the advanced version.
   - Focus on additions that provide entirely new content, not just simplified explanations of existing content.
   - Count each individual addition as a separate instance.
   - Examples:
     - Correct: [No mention of side effects] → "This medication may cause drowsiness and nausea."
     - Incorrect: "Photosynthesis" → "Photosynthesis is how plants make their food" (this is elaboration, not addition)

9. Simplification of Figurative Language: 
   - Note instances where metaphors, similes, or other figurative expressions are simplified or replaced with more literal language.
   - Count each simplification of figurative language as a separate instance.

10. Elaboration and Clarification: 
    - Identify extra details added in the elementary version to make certain concepts clearer.
    - Note instances where names or subjects replace pronouns to avoid ambiguity.
    - Count each added detail or clarification as a separate instance.

11. Personalization or Audience Accommodation: 
    - Note the addition of more specific or personal words to clarify the subject and reduce ambiguity.
    - Count each instance of personalization or audience accommodation as a separate change.

12. Connective Changes: 
    - Identify changes in the way clauses or phrases within a sentence are connected.
    - Note significant additions, removals, or substitutions of:
      a) Coordinating conjunctions (e.g., and, but, or)
      b) Subordinating conjunctions (e.g., because, although, while)
      c) Transitional phrases (e.g., however, therefore, in addition)
      d) Commas that affect the relationship between clauses within a sentence
      e) Semicolons or colons that join related independent clauses within a single sentence
    - Focus on changes that alter the logical flow or complexity within a sentence.
    - Do NOT include changes that involve splitting or merging sentences - these belong in the Sentence Splitting or Sentence Merging categories.
    - Count how many times the connective changes are made within the text.
    - Examples:
      - Correct: "The project was challenging and time-consuming." → "The project was challenging but rewarding."
      - Correct: "Despite the rain, we continued our hike." → "We continued our hike although it was raining."
      - Incorrect: "He was late. He missed the bus." → "He was late because he missed the bus." (This is sentence merging, not a connective change)

For each change identified, provide a brief, clear explanation. Do not impose or infer changes that are not explicitly present in the text. If you're unsure about a change, do not include it.

Before finalizing your analysis, review each identified change and ask yourself:
1. For vocabulary changes: Is this truly a one-to-one word or small phrase substitution, or part of a larger rephrasing?
2. For rephrasing: Does this change affect a significant portion of the sentence, or is it just a minor word change?
3. For deletions: Is this information completely absent from the elementary text, or just expressed differently?
4. For additions: Is this entirely new information, or just a clarification of existing content?
5. For connective changes: Does this change alter how clauses or phrases are connected within a single sentence, without involving sentence splitting or merging?

If you're not confident that a change fits these criteria, do not include it in your analysis.

Important guidelines:
- For vocabulary and rephrasing: If the elementary text entirely omits a word or phrase from the advanced text, classify it as a deletion instead of a vocabulary change or rephrasing.
- For sentence splitting and merging: Only apply these categories if the count of sentences explicitly changes between the advanced and elementary versions. Confirm each case by reviewing the sentence count and structure carefully.
- Before categorizing any change as splitting or merging, verify that it accurately represents a transformation in sentence count or structure.
- Ensure the JSON output is valid, well-formed, and follows the requested structure strictly.
  
Important formatting instructions:
- Ensure the JSON is valid and well-formed.
- Do not include any markdown formatting (e.g., no ```json markers).
- CRITICAL: Do not insert line breaks or newlines in any string values in the JSON. All strings MUST be continuous, single-line values.
- If a string value is long, it should continue on the same line without any line breaks.
- If there is no valid substitution (e.g., in a vocabulary change), use the placeholder "omitted" for the missing part.
- Ensure that every key-value pair is complete, with no trailing commas or missing values.
- Ensure that the JSON does not include empty details (e.g., remove empty entries in the "details" field if no change is present).

Advanced Text: {sample_text['adv']}

Elementary Text: {sample_text['ele']}

Provide your analysis in JSON format, using the following structure. Remember, all string values MUST be on a single line without any line breaks.
PLEASE BE CAREFUL WITH THE COMAS SO THAT THE JSON FORMAT IS CORRECT.

{{
  "analysis": {{
    "advanced": "Full advanced text on a single line",
    "elementary": "Full elementary text on a single line",
    "changes": {{
      "vocabulary": {{"applied": true/false, "details": ["word/phrase A → word/phrase B", "another change"], "count": number}},
      "syntax": {{"applied": true/false, "details": ["Specific major structure changes all on one line"], "count": number}},
      "rephrasing": {{"applied": true/false, "details": ["Specific significant rewording all on one line"], "count": number}},
      "sentence_splitting": {{"applied": true/false, "details": ["Specific instances all on one line"], "count": number}},
      "sentence_merging": {{"applied": true/false, "details": ["Specific instances all on one line"], "count": number}},
      "thematic_shift": {{"applied": true/false, "details": ["Specific major topic or perspective changes all on one line"], "count": number}},
      "deletion": {{"applied": true/false, "details": ["Specific significant omissions all on one line"], "count": number}},
      "addition": {{"applied": true/false, "details": ["Specific significant additions all on one line"], "count": number}},
      "figurative_language_simplification": {{"applied": true/false, "details": ["Specific instances of simplified figurative language all on one line"], "count": number}},
      "elaboration_clarification": {{"applied": true/false, "details": ["Specific instances of added details or pronoun clarification all on one line"], "count": number}},
      "personalization": {{"applied": true/false, "details": ["Specific instances of added personal or specific words all on one line"], "count": number}},
      "connective_changes": {{"applied": true/false, "details": ["Specific changes in connections within sentences all on one line"], "count": number}}
    }}
  }}
}}

For each change type:
- Set "applied" to true if any changes of that type are found, false otherwise.
- List all specific changes in the "details" array.
- Set "count" to the total number of distinct changes found for that type. If no changes are found, set "count" to 0.
- Make sure count reflects the actual number of changes made, not just the number of entries in details.


Ensure that the JSON is valid and properly formatted. Return only the JSON data without additional comments or text.
Ensure that the JSON is valid and properly formatted for accurate analysis and evaluation.
Return only the JSON data without additional comments or text afterwards because if not then I can't parse the json response.
Be careful with the comas, and with repeating the same key in the JSON object, since it might cause erros in parsing later on.
"""
    return prompt

# API response functions (unchanged)
def get_llama_response(sample_text):
    prompt = get_analysis_prompt(sample_text)
    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes text simplifications."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
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
        temperature=0.3,
        max_tokens=2000
    )
    return response.choices[0].message.content

def get_gemini_response(sample_text):
    prompt = get_analysis_prompt(sample_text)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# JSON parsing function (unchanged)
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

# Updated function to process aligned data line by line
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

def write_result_to_csv(result, llm_name):
    
    base_dir = r'C:/Users/raque/Desktop/LaCC_READABILITY/Readability/output_analysis_LLM_answers/LLM_answers/output_csv'

    # Create a specific directory for each model
    model_dir = os.path.join(base_dir, f'{llm_name}_Output')
    os.makedirs(model_dir, exist_ok=True)

    # Define the output file path within the model-specific directory
    output_file = os.path.join(model_dir, f'{llm_name}_detailed_output.csv')

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
            'vocabulary', 'syntax', 'rephrasing', 'sentence_splitting', 'sentence_merging',
            'thematic_shift', 'deletion', 'addition', 'figurative_language_simplification',
            'elaboration_clarification', 'personalization', 'connective_changes'
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


# Modified text processing function to include all models
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