# womensorg_sentiment_755
Final project for 755 using data from Facebook collected on local women's organizations.

# Facebook Women Organization Posts Database

## 📌 Project Overview
This repository contains a dataset and code for analyzing Facebook posts from women’s organizations. The primary objective is to **understand the characteristics of posts**—including engagement metrics—through **sentiment analysis** and **neural network modeling**.

We focus on uncovering how different post features (e.g., tone, content type, engagement) relate to audience interaction, with the goal of drawing insights into effective social media communication strategies.

---

## 📂 Repository Structure

```plaintext
.
├── data/                       # Raw and cleaned datasets
├── scripts/
│   ├── data_cleaning.R          # Data cleaning and preprocessing (R)
│   ├── sentiment_message.py     # Sentiment analysis for 'message' column (Python)
│   ├── sentiment_description.py # Sentiment analysis for 'page.description' column (Python)
│   ├── neural_network.R         # Neural network modeling (R)
│   ├── eda_missingness.py       # Exploratory Data Analysis for missingness (Python)
├── docs/
│   ├── data_dictionary.md       # Data dictionary for all columns
│   ├── project_report.qmd       # Quarto file with detailed project explanation
└── README.md                    # This file
```

---

## 📊 Dataset Description

The dataset contains Facebook post information from women’s organizations, with columns such as:

| Column              | Description |
|---------------------|-------------|
| `date`              | Date of the post |
| `type`              | Post type (e.g., photo, link, video, status) |
| `title`             | Title of the post (if applicable) |
| `caption`           | Caption associated with the post |
| `description`       | Short description or summary |
| `message`           | Main text content of the post |
| ...                 | Additional engagement and metadata columns |

See the **[data dictionary](docs/data_dictionary.md)** for full details.

---

## 🔍 Methods & Workflow

1. **Data Cleaning**  
   - Performed in R to handle missing values, format dates, and standardize text fields.

2. **Exploratory Data Analysis (EDA)**  
   - Conducted in Python, focusing on **missingness patterns** and variable distributions.

3. **Sentiment Analysis**  
   - In Python:  
     - `message` column sentiment  
     - `page.description` column sentiment  
   - Used NLP libraries to generate polarity and subjectivity scores.

4. **Neural Network Modeling**  
   - Built in R to predict engagement based on post features and sentiment scores.

5. **Documentation**  
   - Full data dictionary in Markdown.  
   - Detailed project explanation in a Quarto file.

---

## 🛠 Tools & Languages
- **R**: Data cleaning, neural network modeling  
- **Python**: Sentiment analysis, EDA for missingness  
- **Quarto**: Project documentation  
- **Markdown**: Data dictionary

---

## 📜 License
This project is licensed under the MIT License.

---

## ✨ Acknowledgements
We thank the organizations and open data contributors whose content made this research possible.
