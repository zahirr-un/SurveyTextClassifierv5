import pandas as pd
import joblib
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.tokenize import word_tokenize
import time
from datetime import datetime
import os

# Define file paths
SURVEY_DATA_PATH = r"C:\Users\ZAHIRR\OneDrive - UNHCR\Workstation\MENA\00_Data Scientist\Apps\Text Classification\SurveyTexts\Survey_Data\Survey_Data_1.xlsx"
OUTPUT_DATA_FOLDER = r"C:\Users\ZAHIRR\OneDrive - UNHCR\Workstation\MENA\00_Data Scientist\Apps\Text Classification\SurveyTexts\Output_Data"
MODEL_SAVE_FOLDER = r"C:\Users\ZAHIRR\OneDrive - UNHCR\Workstation\MENA\00_Data Scientist\Apps\Text Classification\SurveyTexts\TrainedModels/"  # Directory where models are saved

# Start timer
start_time = time.time()

# Load the models and other components
print("Loading models and other components...")
tokenizer = T5Tokenizer.from_pretrained(os.path.join(MODEL_SAVE_FOLDER, 't5-summarization-tokenizer'))
model = T5ForConditionalGeneration.from_pretrained(os.path.join(MODEL_SAVE_FOLDER, 't5-summarization-model'))
clf = joblib.load(os.path.join(MODEL_SAVE_FOLDER, 'multi_label_classifier.pkl'))
vectorizer = joblib.load(os.path.join(MODEL_SAVE_FOLDER, 'tfidf_vectorizer.pkl'))
mlb = joblib.load(os.path.join(MODEL_SAVE_FOLDER, 'multi_label_binarizer.pkl'))
print("Models and other components loaded.")
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Preprocess text data function
def preprocess_text(text):
    if pd.isnull(text) or len(text.strip()) < 4:
        return ""
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)

# Define the summarization function
def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=20, min_length=5, length_penalty=5.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to select main category based on predefined logic
def select_main_category(detailed_categories):
    priority_order = [
        "Security and Safety",
        "Access to Services",
        "Housing issues",
        "Legal and Policy Factors",
        "Economic Factors",
        "Socioeconomic Factors",
        "Specific Protection Issues",
        "Family and Personal Reasons",
        "Social and Community Factors",
        "Movement regulations",
        "Illegal movement - Transit country",
        "Illegal movement",
        "Pendular movements",
        "Legal movement",
        "Environmental Factors",
        "Legal movement - Transit country",
        "Undecided/Undisclosed",
        "Other"
    ]
    for category in priority_order:
        if category in detailed_categories:
            return category
    if detailed_categories:
        return detailed_categories[0]
    return "Uncategorized"

# Process the survey data
print("Step 1: Loading and preprocessing survey data...")

# Load survey data
survey_data = pd.read_excel(SURVEY_DATA_PATH, sheet_name=0)  # Selecting the first sheet by default

# Filter columns starting with 'translated_'
translated_columns = [col for col in survey_data.columns if col.startswith('translated_')]

# Process each translated column
for col in translated_columns:
    new_col_name_prefix = col.split('translated_')[1]
    
    # Preprocess text data
    survey_data[col] = survey_data[col].apply(preprocess_text)
    
    # Generate summaries for non-empty text data
    print(f"Generating summaries for {col}...")
    survey_data[f'Summary_{new_col_name_prefix}'] = survey_data[col].apply(lambda x: generate_summary(x) if x != "" else "")
    
    # Vectorize the survey data text using the same vectorizer for non-empty text data
    X_survey = vectorizer.transform(survey_data[col] + " " + survey_data[f'Summary_{new_col_name_prefix}'])
    
    # Detect detailed categories for non-empty text data
    print(f"Detecting detailed categories for {col}...")
    detailed_categories_pred = clf.predict(X_survey)
    detailed_categories_pred = mlb.inverse_transform(detailed_categories_pred)
    survey_data[f'Detailed_Category_{new_col_name_prefix}'] = [list(pred) if survey_data[col].iloc[idx] != "" else [] for idx, pred in enumerate(detailed_categories_pred)]
    
    # Select main category for non-empty text data
    print(f"Selecting main categories for {col}...")
    survey_data[f'Main_Category_{new_col_name_prefix}'] = survey_data[f'Detailed_Category_{new_col_name_prefix}'].apply(select_main_category)

    # Insert new columns next to the original translated_ column
    summary_col = survey_data.pop(f'Summary_{new_col_name_prefix}')
    detailed_col = survey_data.pop(f'Detailed_Category_{new_col_name_prefix}')
    main_col = survey_data.pop(f'Main_Category_{new_col_name_prefix}')
    col_idx = survey_data.columns.get_loc(col) + 1
    survey_data.insert(col_idx, f'Summary_{new_col_name_prefix}', summary_col)
    survey_data.insert(col_idx + 1, f'Detailed_Category_{new_col_name_prefix}', detailed_col)
    survey_data.insert(col_idx + 2, f'Main_Category_{new_col_name_prefix}', main_col)

print("All columns processed.")
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Save processed data to output file
output_file_path = f"{OUTPUT_DATA_FOLDER}/processed_survey_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
survey_data.to_excel(output_file_path, index=False)
print(f"Processed data saved to {output_file_path}")

print("All steps completed successfully.")
elapsed_time = time.time() - start_time
print(f"Total elapsed time: {elapsed_time:.2f} seconds")

