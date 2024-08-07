import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import torch
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Define file paths
TRAINING_DATA_PATH = r"C:\Users\ZAHIRR\OneDrive - UNHCR\Workstation\MENA\00_Data Scientist\Apps\Text Classification\SurveyTexts\Training_Data\training_data.xlsx"
OUTPUT_DATA_FOLDER = r"C:\Users\ZAHIRR\OneDrive - UNHCR\Workstation\MENA\00_Data Scientist\Apps\Text Classification\SurveyTexts\Output_Data"
MODEL_SAVE_FOLDER = r"C:\Users\ZAHIRR\OneDrive - UNHCR\Workstation\MENA\00_Data Scientist\Apps\Text Classification\SurveyTexts\TrainedModels"  # Directory where models are saved

# Start timer
start_time = time.time()

# Preprocess text data function
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)

print("Step 1: Loading and preprocessing training data...")

# Load training data
training_data = pd.read_excel(TRAINING_DATA_PATH, sheet_name=0)  # Selecting the first sheet by default
# Verify the number of rows loaded
print(f"Total rows loaded: {len(training_data)}")

# Apply preprocessing to the text columns
training_data['Original Response'] = training_data['Original Response'].apply(preprocess_text)
training_data['Summary'] = training_data['Summary'].apply(preprocess_text)
training_data['Detailed_Category'] = training_data['Detailed_Category'].apply(lambda x: str(x).split(',') if isinstance(x, str) else [])
training_data['Main_Category'] = training_data['Main_Category'].apply(preprocess_text)

# Verify preprocessing
print(f"Total rows after preprocessing: {len(training_data)}")
print(training_data.head())

# Continue with the rest of the script
# Initialize T5 model for text summarization
print("Step 2: Initializing T5 model for text summarization...")
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
model = T5ForConditionalGeneration.from_pretrained('t5-small')
print("T5 model initialized.")
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Define the summarization function
def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=20, min_length=5, length_penalty=5.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Prepare the dataset for training
train_encodings = tokenizer(list(training_data['Original Response']), truncation=True, padding=True, max_length=512)
summary_encodings = tokenizer(list(training_data['Summary']), truncation=True, padding=True, max_length=20)

# Create a dataset class
class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, summaries):
        self.encodings = encodings
        self.summaries = summaries
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.summaries['input_ids'][idx])
        return item
    
    def __len__(self):
        return len(self.summaries['input_ids'])

# Create dataset
train_dataset = SummarizationDataset(train_encodings, summary_encodings)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',            # output directory
    num_train_epochs=5,                # total number of training epochs
    per_device_train_batch_size=16,    # batch size for training (increased from 8)
    warmup_steps=500,                  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                 # strength of weight decay
    logging_dir='./logs',              # directory for storing logs
    logging_steps=10,
    learning_rate=3e-5,                # initial learning rate
    lr_scheduler_type='linear',        # learning rate scheduler type
    gradient_accumulation_steps=1,     # number of steps to accumulate before performing a backward/update pass
    gradient_checkpointing=False,      # whether to use gradient checkpointing
    fp16=True,                         # whether to use mixed precision training
)

# Create Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                'labels': torch.stack([f['labels'] for f in data])},
)

# Train the model
print("Step 3: Training the summarization model...")
trainer.train()
print("Summarization model trained.")
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Generate summaries for training data (for verification)
training_data['Generated_Summary'] = training_data['Original Response'].apply(generate_summary)
print(training_data[['Original Response', 'Summary', 'Generated_Summary']].head())

# Save the model and tokenizer
model.save_pretrained(os.path.join(MODEL_SAVE_FOLDER, 't5-summarization-model'))
tokenizer.save_pretrained(os.path.join(MODEL_SAVE_FOLDER, 't5-summarization-tokenizer'))

# Save summarization review file
summary_review_file_path = f"{OUTPUT_DATA_FOLDER}/summarization_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
training_data[['Original Response', 'Summary', 'Generated_Summary']].to_excel(summary_review_file_path, index=False)
print(f"Summarization review file saved to {summary_review_file_path}")

# Binarize the labels for multi-label classification
print("Step 4: Binarizing labels for multi-label classification...")
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(training_data['Detailed_Category'])
print("Labels binarized.")
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Vectorize the text data using TF-IDF
print("Step 5: Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(training_data['Original Response'] + " " + training_data['Summary'])

# Split data into training and testing sets
print("Step 6: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Train multi-label classifier
print("Step 7: Training multi-label classifier...")
clf = MultiOutputClassifier(RandomForestClassifier())
clf.fit(X_train, y_train)
print("Multi-label classifier trained.")
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Predict detailed categories for training data (for verification)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Ensure we use the same index for predictions as the original training data
training_data['Predicted_Detailed_Category'] = pd.Series(mlb.inverse_transform(y_train_pred), index=training_data.index[:len(y_train_pred)])
print(training_data[['Original Response', 'Detailed_Category', 'Predicted_Detailed_Category']].head())

# Save detailed category review file
detailed_category_review_file_path = f"{OUTPUT_DATA_FOLDER}/detailed_category_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
training_data[['Original Response', 'Detailed_Category', 'Predicted_Detailed_Category']].to_excel(detailed_category_review_file_path, index=False)
print(f"Detailed category review file saved to {detailed_category_review_file_path}")

# Save the trained models and other components
joblib.dump(clf, os.path.join(MODEL_SAVE_FOLDER, 'multi_label_classifier.pkl'))
joblib.dump(vectorizer, os.path.join(MODEL_SAVE_FOLDER, 'tfidf_vectorizer.pkl'))
joblib.dump(mlb, os.path.join(MODEL_SAVE_FOLDER, 'multi_label_binarizer.pkl'))

print("All models and components saved successfully.")
elapsed_time = time.time() - start_time
print(f"Total elapsed time: {elapsed_time:.2f} seconds")
