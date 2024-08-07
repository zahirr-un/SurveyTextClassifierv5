Here is the entire `README.md` content in a single markdown block for easy copying:

```markdown
# Text Classification Project

This project focuses on summarizing and categorizing refugee responses. It uses a T5 model for text summarization and a multi-label classifier for categorizing the summarized text.

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Setup Instructions](#setup-instructions)
- [Usage Instructions](#usage-instructions)
- [Model Training](#model-training)
- [Applying the Model](#applying-the-model)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to process refugee response data and generate three additional columns: `Summary`, `Detailed_Category`, and `Main_Category`.

1. **Text Summarization**: Summarize the original response into a concise form.
2. **Detailed Category Detection**: Detect detailed categories based on the original text and summary.
3. **Main Category Selection**: Determine the main category based on predefined logic.

## Folder Structure

```
App_TextClassification_V5/
├── data/
│   ├── training_data.xlsx
│   ├── survey_data.xlsx
│   └── Verification_DummatySurveydata.xlsx
├── models/
│   ├── t5-summarization-model/
│   ├── t5-summarization-tokenizer/
│   ├── multi_label_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── multi_label_binarizer.pkl
├── notebooks/
│   └── data_preprocessing_notebook.ipynb
├── output/
│   └── (empty for now, will contain output files)
├── scripts/
│   ├── v5_01_train_model.py
│   ├── v5_02_apply_model.py
│   └── requirements.txt
├── README.md
└── .gitignore
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/TextClassificationProject.git
   cd TextClassificationProject
   ```

2. **Install the Required Packages**:
   ```bash
   pip install -r scripts/requirements.txt
   ```

3. **Ensure Data and Model Files are in Place**:
   - Place `training_data.xlsx`, `survey_data.xlsx`, and `Verification_DummatySurveydata.xlsx` in the `data/` folder.
   - Ensure the trained models are in the `models/` folder.

## Usage Instructions

### Model Training

To train the models, run:
```bash
python scripts/v5_01_train_model.py
```

This script will:
- Load and preprocess the training data.
- Train the T5 model for text summarization.
- Train the multi-label classifier for detailed category detection.
- Save the trained models and other components.

### Applying the Model

To apply the trained models on survey data, run:
```bash
python scripts/v5_02_apply_model.py
```

This script will:
- Load and preprocess the survey data.
- Generate summaries for the survey responses.
- Detect detailed categories for the survey responses.
- Select main categories for the survey responses.
- Save the processed data to the `output/` folder.

## Requirements

The required packages are listed in `scripts/requirements.txt`. Install them using:
```bash
pip install -r scripts/requirements.txt
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
```