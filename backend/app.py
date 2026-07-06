"""
Fake News Analyzer - 4-Stage NLTK Pipeline

Stage 1: Data Collection
Stage 2: Text Preprocessing
Stage 3: Feature Extraction
Stage 4: Rumours/Fake News Classification
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import numpy as np
import re
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# DuckDuckGo search – optional, degrades gracefully
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False
    print("[WARNING] duckduckgo-search not installed. Run: pip install duckduckgo-search")

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app)

# STAGE 1: DATA COLLECTION
class DataCollector:
    """Stage 1: Collect and validate input data"""
    
    def __init__(self):
        self.collected_data = []
        self.metadata = {}
    
    def collect_text(self, text, source="user_input"):
        """Collect text data from various sources"""
        data_point = {
            'raw_text': text,
            'source': source,
            'timestamp': self._get_timestamp(),
            'length': len(text),
            'word_count': len(text.split())
        }
        self.collected_data.append(data_point)
        return data_point
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_statistics(self):
        """Get collection statistics"""
        return {
            'total_documents': len(self.collected_data),
            'avg_length': np.mean([d['length'] for d in self.collected_data]) if self.collected_data else 0,
            'sources': list(set([d['source'] for d in self.collected_data]))
        }

# STAGE 2: TEXT PREPROCESSING
class TextPreprocessor:
    """Stage 2: Clean and preprocess text using NLTK"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
    
    def preprocess(self, text, steps=None):
        """Full preprocessing pipeline"""
        if steps is None:
            steps = ['lowercase', 'remove_urls', 'remove_punctuation', 
                    'tokenize', 'remove_stopwords', 'lemmatize']
        
        results = {'original': text}
        processed = text
        
        for step in steps:
            if step == 'lowercase':
                processed = self._to_lowercase(processed)
            elif step == 'remove_urls':
                processed = self._remove_urls(processed)
            elif step == 'remove_punctuation':
                processed = self._remove_punctuation(processed)
            elif step == 'tokenize':
                tokens = self._tokenize(processed)
                results['tokens'] = tokens
                processed = ' '.join(tokens)
            elif step == 'remove_stopwords':
                tokens = self._remove_stopwords(results.get('tokens', self._tokenize(processed)))
                results['tokens_no_stop'] = tokens
                processed = ' '.join(tokens)
            elif step == 'lemmatize':
                tokens = self._lemmatize(results.get('tokens_no_stop', self._tokenize(processed)))
                results['lemmatized'] = tokens
                processed = ' '.join(tokens)
            elif step == 'stem':
                tokens = self._stem(results.get('tokens', self._tokenize(processed)))
                results['stemmed'] = tokens
                processed = ' '.join(tokens)
            
            results[step] = processed
        
        # Add sentiment analysis
        results['sentiment'] = self._analyze_sentiment(text)
        
        return results
    
    def _to_lowercase(self, text):
        return text.lower()
    
    def _remove_urls(self, text):
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def _remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def _tokenize(self, text):
        return word_tokenize(text)
    
    def _remove_stopwords(self, tokens):
        return [t for t in tokens if t.lower() not in self.stop_words]
    
    def _lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def _stem(self, tokens):
        return [self.stemmer.stem(t) for t in tokens]
    
    def _analyze_sentiment(self, text):
        scores = self.sia.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'label': 'positive' if scores['compound'] > 0.05 else ('negative' if scores['compound'] < -0.05 else 'neutral')
        }

# STAGE 3: FEATURE EXTRACTION

class FeatureExtractor:
    """Stage 3: Extract features using TF-IDF and linguistic features"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.is_fitted = False
    
    def extract_features(self, texts, fit=False):
        """Extract TF-IDF features"""
        if fit or not self.is_fitted:
            features = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
        else:
            features = self.vectorizer.transform(texts)
        
        return features
    
    def extract_linguistic_features(self, text, tokens):
        """Extract additional linguistic features"""
        sentences = sent_tokenize(text)
        
        features = {
            # Basic counts
            'char_count': len(text),
            'word_count': len(tokens),
            'sentence_count': len(sentences),
            
            # Average metrics
            'avg_word_length': np.mean([len(w) for w in tokens]) if tokens else 0,
            'avg_sentence_length': len(tokens) / len(sentences) if sentences else 0,
            
            # Vocabulary richness
            'unique_words': len(set(tokens)),
            'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
            
            # Punctuation features
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'quote_count': text.count('"') + text.count("'"),
            
            # Capitalization
            'all_caps_words': sum(1 for w in tokens if w.isupper() and len(w) > 1),
            
            # Readability (simple approximation)
            'avg_syllables_per_word': self._estimate_syllables(tokens),
        }
        
        return features
    
    def _estimate_syllables(self, tokens):
        """Estimate syllable count"""
        def count_syllables(word):
            vowels = 'aeiouy'
            count = 0
            prev_was_vowel = False
            for char in word.lower():
                if char in vowels:
                    if not prev_was_vowel:
                        count += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False
            if word.endswith('e'):
                count -= 1
            return max(1, count)
        
        if not tokens:
            return 0
        return np.mean([count_syllables(w) for w in tokens])
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

# STAGE 4: FAKE NEWS CLASSIFICATION

class FakeNewsClassifier:
    """Stage 4: Classify news as real or fake"""
    
    def __init__(self):
        self.models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        self.is_trained = False
    
    def train(self, X, y, model_name='logistic_regression'):
        """Train a classification model"""
        if model_name in self.models:
            self.models[model_name].fit(X, y)
            self.trained_models[model_name] = self.models[model_name]
            self.is_trained = True
            return True
        return False
    
    def predict(self, X, model_name='logistic_regression'):
        """Make predictions"""
        if model_name in self.trained_models:
            model = self.trained_models[model_name]
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            return {
                'prediction': 'FAKE' if prediction == 1 else 'REAL',
                'is_fake': bool(prediction == 1),
                'confidence': float(max(probabilities)),
                'probabilities': {
                    'real': float(probabilities[0]),
                    'fake': float(probabilities[1])
                }
            }
        return None
    
    def get_model_info(self):
        """Get information about available models"""
        return {
            'available_models': list(self.models.keys()),
            'trained_models': list(self.trained_models.keys()),
            'is_trained': self.is_trained
        }

# MAIN PIPELINE
class FakeNewsAnalyzer:
    """Complete 4-Stage Fake News Analysis Pipeline"""
    
    def __init__(self):
        self.collector = DataCollector()
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.classifier = FakeNewsClassifier()
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Load and train on sample data"""
        # Sample training data for demonstration
        sample_real = [
            "The government announced new policies today in a press conference.",
            "Scientists discover new species in the Amazon rainforest.",
            "Local community comes together to support homeless shelter.",
            "New study shows benefits of regular exercise for mental health.",
            "Technology company releases updated software with security patches.",
            "Weather forecast predicts sunny skies for the weekend.",
            "University researchers publish findings on climate change.",
            "Hospital reports decrease in patient wait times.",
            "City council approves funding for public transportation.",
            "Education board introduces new curriculum standards."
        ]
        
        sample_fake = [
            "SHOCKING: Government hiding alien technology from citizens!!!",
            "MIRACLE CURE: This one weird trick cures all diseases forever!",
            "BREAKING: Famous celebrity dies and comes back to life!",
            "CONSPIRACY: Secret society controls the world's banks!",
            "URGENT: Scientists discover chocolate makes you immortal!",
            "EXPOSED: All politicians are actually lizard people!",
            "MUST READ: This natural remedy doctors don't want you to know!",
            "ALERT: Moon landing was completely faked by Hollywood!",
            "INCREDIBLE: Man discovers he can fly after drinking energy drink!",
            "WARNING: 5G towers are mind control devices!!!"
        ]
        
        texts = sample_real + sample_fake
        labels = [0] * len(sample_real) + [1] * len(sample_fake)
        
        # Preprocess all texts
        processed_texts = []
        for text in texts:
            prep = self.preprocessor.preprocess(text)
            processed_texts.append(' '.join(prep.get('lemmatized', prep.get('tokens', []))))
        
        # Extract features and train
        X = self.feature_extractor.extract_features(processed_texts, fit=True)
        self.classifier.train(X, labels, 'logistic_regression')
        self.classifier.train(X, labels, 'naive_bayes')
    
    def analyze(self, text, detailed=False):
        """Run complete analysis pipeline"""
        results = {
            'pipeline_stages': {}
        }
        
        # Stage 1: Data Collection
        collected = self.collector.collect_text(text)
        results['pipeline_stages']['data_collection'] = {
            'status': 'completed',
            'document_length': collected['length'],
            'word_count': collected['word_count']
        }
        
        # Stage 2: Text Preprocessing
        preprocessed = self.preprocessor.preprocess(text)
        results['pipeline_stages']['text_preprocessing'] = {
            'status': 'completed',
            'steps_applied': ['lowercase', 'remove_urls', 'remove_punctuation', 
                            'tokenize', 'remove_stopwords', 'lemmatize'],
            'tokens_count': len(preprocessed.get('tokens', [])),
            'sentiment': preprocessed['sentiment']
        }
        
        # Stage 3: Feature Extraction
        processed_text = ' '.join(preprocessed.get('lemmatized', preprocessed.get('tokens', [])))
        features = self.feature_extractor.extract_features([processed_text])
        linguistic = self.feature_extractor.extract_linguistic_features(
            text, preprocessed.get('tokens', [])
        )
        results['pipeline_stages']['feature_extraction'] = {
            'status': 'completed',
            'tfidf_features': features.shape[1],
            'linguistic_features_extracted': len(linguistic)
        }
        
        # Stage 4: Classification
        prediction = self.classifier.predict(features, 'logistic_regression')
        results['pipeline_stages']['classification'] = {
            'status': 'completed',
            'model_used': 'logistic_regression'
        }
        
        # Final results
        results['prediction'] = prediction
        results['linguistic_features'] = linguistic if detailed else None
        
        return results

# Initialize analyzer
analyzer = FakeNewsAnalyzer()

# API ENDPOINTS

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint"""
    data = request.json
    text = data.get('text', '')
    detailed = data.get('detailed', False)
    
    if not text or len(text.strip()) < 10:
        return jsonify({
            'error': 'Text too short. Please provide at least 10 characters.',
            'success': False
        }), 400
    
    try:
        results = analyzer.analyze(text, detailed=detailed)
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    """Text preprocessing endpoint"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided', 'success': False}), 400
    
    try:
        results = analyzer.preprocessor.preprocess(text)
        return jsonify({
            'success': True,
            'preprocessing_results': results
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/features', methods=['POST'])
def extract_features():
    """Feature extraction endpoint"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided', 'success': False}), 400
    
    try:
        preprocessed = analyzer.preprocessor.preprocess(text)
        tokens = preprocessed.get('tokens', [])
        linguistic = analyzer.feature_extractor.extract_linguistic_features(text, tokens)
        
        return jsonify({
            'success': True,
            'linguistic_features': linguistic,
            'sentiment': preprocessed['sentiment']
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'nltk_version': nltk.__version__,
        'pipeline_stages': [
            'Data Collection',
            'Text Preprocessing',
            'Feature Extraction',
            'Rumours/Fake News Classification'
        ],
        'models_available': analyzer.classifier.get_model_info(),
        'ddg_available': DDG_AVAILABLE
    })


# ── Keyword extractor helper ──────────────────────────────────────────────────
def extract_search_query(text: str, max_keywords: int = 6) -> str:
    """Extract the most meaningful keywords from text for a search query."""
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words('english'))
    # Add common filler words
    stop_words.update({
        'said', 'say', 'says', 'also', 'would', 'could', 'one', 'two',
        'three', 'new', 'get', 'got', 'make', 'made', 'like', 'will',
        'use', 'used', 'using', 'may', 'many', 'much', 'even', 'back',
        'still', 'way', 'well', 'need', 'time', 'year', 'years', 'people',
        'report', 'reports', 'reported', 'news', 'article', 'content',
        'fetched', 'extracted', 'text', 'extracted', 'ocr', 'image',
        'analysis', 'body', 'standard', 'provided'
    })

    tokens = word_tokenize(text.lower())
    # Keep only alphabetic tokens not in stopwords, length > 3
    keywords = [
        t for t in tokens
        if t.isalpha() and t not in stop_words and len(t) > 3
    ]

    # Frequency count and keep top N
    freq = Counter(keywords)
    top_words = [w for w, _ in freq.most_common(max_keywords)]
    return ' '.join(top_words)


@app.route('/api/search-context', methods=['POST'])
def search_context():
    """Search the web for context/coverage related to the analyzed news text."""
    data = request.json
    text = data.get('text', '')
    is_fake = data.get('is_fake', None)

    if not text or len(text.strip()) < 10:
        return jsonify({'error': 'No text provided', 'success': False}), 400

    if not DDG_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Search library not installed. Run: pip install duckduckgo-search',
            'results': []
        }), 503

    try:
        query = extract_search_query(text)
        if not query.strip():
            query = text[:120]  # fallback: use raw text snippet

        results = []
        seen_urls = set()

        with DDGS() as ddgs:
            # 1. News search first — most relevant for fake news checking
            try:
                news_hits = list(ddgs.news(
                    keywords=query,
                    region='wt-wt',
                    safesearch='moderate',
                    max_results=4
                ))
                for h in news_hits:
                    url = h.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append({
                            'type': 'news',
                            'title':   h.get('title', ''),
                            'url':     url,
                            'snippet': h.get('body', ''),
                            'source':  h.get('source', ''),
                            'date':    h.get('date', ''),
                        })
            except Exception:
                pass  # news search may fail, fall through to web

            # 2. Web search for fact-checks / related coverage
            if len(results) < 5:
                try:
                    fact_query = query + ' fact check'
                    web_hits = list(ddgs.text(
                        keywords=fact_query,
                        region='wt-wt',
                        safesearch='moderate',
                        max_results=4
                    ))
                    for h in web_hits:
                        url = h.get('href', '')
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            results.append({
                                'type': 'web',
                                'title':   h.get('title', ''),
                                'url':     url,
                                'snippet': h.get('body', ''),
                                'source':  _extract_domain(url),
                                'date':    '',
                            })
                except Exception:
                    pass

        return jsonify({
            'success': True,
            'query': query,
            'results': results[:6]  # cap at 6 total
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'results': []
        }), 500


def _extract_domain(url: str) -> str:
    """Extract readable domain from a URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.netloc.replace('www.', '')
        return host
    except Exception:
        return url


if __name__ == '__main__':
    app.run(debug=True, port=5000)
