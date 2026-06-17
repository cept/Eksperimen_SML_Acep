import pandas as pd
import string
import re
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def run_pipeline(input_file_path, output_file_path):
    # 1. Load Data
    df = pd.read_csv(input_file_path)
    df_clean = df[['Customer Review', 'Emotion']].copy()
    
    # 2. Preprocessing sesuai kriteria
    df_clean = df_clean.dropna().drop_duplicates()
    df_clean['word_count'] = df_clean['Customer Review'].apply(lambda x: len(str(x).split()))
    df_clean = df_clean[df_clean['word_count'] > 1]
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    df_clean['cleaned_text'] = df_clean['Customer Review'].apply(clean_text)
    
    # 3. Encoding
    le = LabelEncoder()
    df_clean['encoded_emotion'] = le.fit_transform(df_clean['Emotion'])
    
    # 4. Save processed data & artifacts
    df_clean.to_csv(output_file_path, index=False)
    
    # Simpan vectorizer untuk digunakan saat deployment/API
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf.fit(df_clean['cleaned_text'])
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    
    print(f"Otomatisasi selesai! Data tersimpan di {output_file_path}")

if __name__ == "__main__":
    # Input mengambil dari folder raw (naik 1 tingkat ke PRDECT-ID_raw)
    input_path = '../PRDECT-ID_raw/PRDECT-ID Dataset.csv'
    # Output disimpan di folder preprocessing ini sendiri
    output_path = 'PRDECT-ID_preprocessing.csv'
    
    run_pipeline(input_path, output_path)