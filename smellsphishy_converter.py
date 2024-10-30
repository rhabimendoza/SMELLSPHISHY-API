import re
import shap
import pickle
import tldextract
import numpy as np
import pandas as pd
import urllib.parse
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

class PhishingDetector:
    
    def __init__(self, model_path, scaler_path, domain_model_path, path_model_path):
        
        # Load the trained model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load word2vec models
        self.domain_model = Word2Vec.load(domain_model_path)
        self.path_model = Word2Vec.load(path_model_path)

        # Initialize shap explainer
        self.explainer = shap.TreeExplainer(self.model.rf)

        # Store feature names 
        self.feature_names = self.scaler.feature_names_in_

        # Tld scoring data
        self.TOTAL_BENIGN = 35445
        self.TOTAL_COMBINED_PHISHING_ADVERSARIAL = 18014

        # Benign tld dictionary
        self.benign_tlds = {
            'com': 25691, 'org': 3476, 'edu': 1310, 'int': 921, 'net': 622,
            'gov': 413, 'ca': 398, 'co.uk': 307, 'google': 299, 'nhs.uk': 271,
            'us': 253, 'ac.uk': 209, 'co.in': 92, 'com.au': 78, 'tv': 71,
            'co': 65, 'fm': 65, 'info': 62, 'org.uk': 41, 'com.ph': 36,
            'de': 34, 'youtube': 33, 'ph': 31, 'qc.ca': 29, 'io': 25,
            'eu': 22, 'gc.ca': 19, 'it': 18, 'co.za': 18, 'link': 17,
            'fr': 15, 'in': 15, 'gl': 15, 'me': 12, 'be': 12,
            'co.kr': 12, 'ch': 11, 'es': 11, 'gov.uk': 11, 'biz': 11,
            'at': 10, 'com.br': 10, 'co.jp': 10, 'dk': 9, 'blog': 8,
            'hn': 8, 'com.tr': 8, 'ru': 8, 'app': 7, 'ie': 7
        }

        # Combined phishing and adversarial tld dictionary
        self.combined_phishing_adversarial_tlds = {
            'io': 6049, 'com': 3492, '': 1491, 'top': 1061, 'dev': 669,
            'me': 587, 'de': 387, 'xyz': 307, 'to': 301, 'app': 293,
            'net': 276, 'ly': 242, 'org': 232, 'cc': 212, 'shop': 190,
            'co': 157, 'info': 149, 'vip': 130, 'live': 113, 'site': 108,
            'fr': 87, 'help': 85, 'cn': 79, 'ru': 65, 'buzz': 60,
            'link': 52, 'ink': 51, 'sbs': 48, 'online': 44, 'asia': 41,
            'cfd': 37, 'icu': 33, 'life': 32, 'it': 31, 'one': 30,
            'com.br': 29, 'club': 29, 'pl': 29, 'my.id': 25, 'bio': 22,
            'lol': 21, 'network': 20, 'gy': 20, 'sh': 18, 'at': 18,
            'id.vn': 18, 'host': 17, 'click': 16, 'es': 15, 'cyou': 15
        }

    # Url preparation
    def clean_url(self, url):
        url = url.strip().lower()
        url = url.split('#')[0]
        url = re.sub(r'(?<!:)/{2,}', '/', url)
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        return url
    def normalize_url(self, url):
        parsed = urllib.parse.urlparse(url)
        path = parsed.path if parsed.path else '/'
        if path != '/' and path.endswith('/'):
            path = path[:-1]
        query = '&'.join(sorted(parsed.query.split('&')))
        return urllib.parse.urlunparse((
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.params,
            query,
            ''
        ))
    def extract_url_components(self, url):
        parsed = urllib.parse.urlparse(url)
        extracted = tldextract.extract(parsed.netloc)
        return {
            'scheme': parsed.scheme,
            'subdomain': extracted.subdomain,
            'domain': extracted.domain,
            'tld': extracted.suffix,
            'path': parsed.path,
            'query': parsed.query
        }
    def preprocess_url(self, url):
        cleaned_url = self.clean_url(url)
        normalized_url = self.normalize_url(cleaned_url)
        components = self.extract_url_components(normalized_url)
        components['url'] = normalized_url
        components['url_length'] = len(normalized_url)
        components['domain_length'] = len(components['domain'])
        components['num_subdomains'] = len(components['subdomain'].split('.')) if components['subdomain'] else 0
        return components
    def calculate_entropy(self, string):
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        entropy = - sum([p * np.log(p) / np.log(2.0) for p in prob if p > 0])
        return entropy
    def calculate_consonant_ratio(self, string):
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        return sum(1 for c in string if c in consonants) / len(string) if string else 0
    def calculate_tld_score(self, tld):
        benign_freq = self.benign_tlds.get(tld, 0) / self.TOTAL_BENIGN
        malicious_freq = self.combined_phishing_adversarial_tlds.get(tld, 0) / self.TOTAL_COMBINED_PHISHING_ADVERSARIAL
        if benign_freq == 0 and malicious_freq == 0:
            return 0
        if benign_freq > 0 and malicious_freq > 0:
            return np.log(benign_freq / malicious_freq)
        return 1 if benign_freq > 0 else -1
    def generate_embeddings(self, tokens, model):
        embeddings = [model.wv[token] for token in tokens if token in model.wv]
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)
    
    # Feature extraction
    def extract_features(self, url):
        preprocessed = self.preprocess_url(url)
        features = {}
        
        # Url features
        features['url_length'] = preprocessed['url_length']
        features['url_entropy'] = self.calculate_entropy(preprocessed['url'])
        features['char_diversity'] = len(set(preprocessed['url'].lower())) / len(preprocessed['url']) if preprocessed['url'] else 0
        features['url_consonant_ratio'] = self.calculate_consonant_ratio(preprocessed['url'])

        # Domain features
        features['domain_length'] = preprocessed['domain_length']
        features['domain_entropy'] = self.calculate_entropy(preprocessed['domain'])
        features['domain_consonant_ratio'] = self.calculate_consonant_ratio(preprocessed['domain'])
        domain_words = re.findall(r'\w+', preprocessed['domain'])
        features['domain_longest_word_length'] = max(len(word) for word in domain_words) if domain_words else 0
        features['domain_digit_ratio'] = sum(c.isdigit() for c in preprocessed['domain']) / len(preprocessed['domain']) if preprocessed['domain'] else 0

        # Path features
        path = preprocessed['path'].strip('/').split('/')
        features['path_length'] = len(preprocessed['path'])
        features['path_entropy'] = self.calculate_entropy(preprocessed['path'])
        features['path_token_count'] = len(path)
        features['max_path_token_length'] = max([len(token) for token in path] + [0])

        # Query features
        features['query_length'] = len(preprocessed['query'])
        features['query_entropy'] = self.calculate_entropy(preprocessed['query'])
        try:
            features['num_query_params'] = len(urllib.parse.parse_qs(preprocessed['query']))
        except Exception:
            features['num_query_params'] = 0

        # Character count features
        features['num_dots'] = preprocessed['url'].count('.')
        features['num_hyphens'] = preprocessed['url'].count('-')
        features['num_at'] = preprocessed['url'].count('@')
        features['num_ampersand'] = preprocessed['url'].count('&')
        features['num_digits'] = sum(c.isdigit() for c in preprocessed['url'])
        features['num_subdirectories'] = len(path)

        # Host-based features
        features['subdomain_count'] = preprocessed['num_subdomains']
        features['tld_length'] = len(preprocessed['tld'])
        features['is_ip_address'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', preprocessed['domain']) else 0
        features['tld_score'] = self.calculate_tld_score(preprocessed['tld'])

        # Word analysis features
        suspicious_keywords = ['secure', 'verify', 'update', 'confirm', 'validate', 'auth', 'unlock', 
                               'suspended', 'blocked', 'renew', 'password', 'credentials', 'id', 'access', 
                               'support', 'signin', 'activate', 'alert', 'warning', 'recovery', 'banking', 
                               'banned', 'upgrade', 'checkout']
        features['has_suspicious_keywords'] = 1 if any(keyword in preprocessed['url'].lower() for keyword in suspicious_keywords) else 0

        # Security indicator features
        features['contains_https'] = 1 if preprocessed['scheme'] == 'https' else 0

        # Generate embeddings
        domain_tokens = simple_preprocess(preprocessed['domain'])
        path_tokens = simple_preprocess(preprocessed['path'])
        domain_embedding = self.generate_embeddings(domain_tokens, self.domain_model)
        path_embedding = self.generate_embeddings(path_tokens, self.path_model)

        # Separate embeddings
        for i, value in enumerate(domain_embedding):
            features[f'domain_emb_{i}'] = value
        for i, value in enumerate(path_embedding):
            features[f'path_emb_{i}'] = value

        # Return features
        return features
    
    # Top features
    def get_top_features(self, feature_values, n=5):
        
        # Calculate shap values for this specific prediction
        shap_values = self.explainer.shap_values(feature_values)
        
        # Handle the shap
        shap_values = np.abs(shap_values[0])  
        
        # Sum the absolute shap values across both classes
        total_shap = np.sum(shap_values, axis=1)
        
        # Create a dictionary of feature names and their total shap values
        feature_importance = dict(zip(self.feature_names, total_shap))

        # Sort features by total shap value
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Get the top n features
        top_features = sorted_features[:n]

        # Create a dictionary with feature names and their values
        top_features_with_values = {
            feature: {
                'importance': float(importance),
                'value': float(feature_values[0][self.feature_names.tolist().index(feature)]),
                'shap_value': float(shap_values[self.feature_names.tolist().index(feature)][0])
            }
            for feature, importance in top_features
        }
        
        # Return feature with values
        return top_features_with_values
    
    # Make prediction
    def predict(self, url):
        features = self.extract_features(url)
        feature_df = pd.DataFrame([features])

        # Ensure all expected columns are present
        expected_columns = self.scaler.feature_names_in_
        for col in expected_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0

        # Reorder columns to match the scaler expected order
        feature_df = feature_df[expected_columns]

        # Scale features
        scaled_features = self.scaler.transform(feature_df)

        # Make prediction
        prediction = self.model.predict(scaled_features)[0]
        probability = self.model.predict_proba(scaled_features)[0]

        # Get top 5 features
        top_features = self.get_top_features(scaled_features)

        return {
            "url": url,
            "is_phishing": int(prediction),
            "phishing_probability": float(probability[1]),
            "benign_probability": float(probability[0]),
            "top_features": top_features
        }

def checkURLInput(url):

    # Invalid checking
    if not validURL(url):
        return 3, 0.0, 0.0, [] 

    # Setup
    detector = PhishingDetector(
        model_path = 'outputs/model.pkl',
        scaler_path = 'outputs/standard_scaler.pkl', 
        domain_model_path = 'outputs/domain.model',
        path_model_path = 'outputs/path.model'
    )

    # Get result
    result = detector.predict(url)
    
    # Get output of model
    benign_prob = result['benign_probability'] 
    phishing_prob = result['phishing_probability']
    top_features = []
    for feature, info in result['top_features'].items():
        top_features.append(f"{feature}: Importance = {info['importance']:.4f}, Value = {info['value']:.4f}")

    # Classify if phishing, warning, or safe
    if 0.71 <= phishing_prob <= 1.00:
        return 1, benign_prob, phishing_prob, top_features
    elif 0.51 <= phishing_prob <= 0.70:
        return 2, benign_prob, phishing_prob, top_features
    else:
        return 0, benign_prob, phishing_prob, top_features

def validURL(url):
    
    try:

        # Use urllib parse to break down the url
        parsed = urllib.parse.urlparse(url)
        
        # Check if scheme and netloc are present
        if not all([parsed.scheme, parsed.netloc]):
            return False
        
        # Validate scheme
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # Validate netloc
        netloc_pattern = r'^([a-zA-Z0-9.-]+(@[a-zA-Z0-9.-]+)?\.)+[a-zA-Z]{2,}$'
        if not re.match(netloc_pattern, parsed.netloc):
            return False
        
        # Ensure there's no double slash after the scheme
        if url.replace(parsed.scheme + '://', '', 1).startswith('//'):
            return False
        
        # Additional check for valid characters in path and query
        valid_path_query_pattern = r'^[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]*$'
        if not re.match(valid_path_query_pattern, parsed.path + '?' + parsed.query):
            return False

        # Return if valid
        return True
    
    except Exception:

        # If any exception occurs during parsing then consider it invalid
        return False