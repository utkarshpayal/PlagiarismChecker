import ast
import astunparse
import difflib
import re
import hashlib
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample code files (replace with your code files)
code_file1 = "index.py"
code_file2 = "index1.py"

# Function to parse code and extract tokens
def parse_code(file_path):
    with open(file_path, 'r') as f:
        code = f.read()
    return ast.parse(code)

# Function to normalize code by removing whitespace and comments
def normalize_code(code):
    code = re.sub(r'#.*', '', code)  # Remove comments
    code = re.sub(r'\s+', ' ', code)  # Remove extra whitespace
    return code.strip()

# Function to calculate code similarity using AST differencing
def calculate_ast_similarity(file1, file2):
    tree1 = parse_code(file1)
    tree2 = parse_code(file2)

    code1 = astunparse.unparse(tree1)
    code2 = astunparse.unparse(tree2)

    norm_code1 = normalize_code(code1)
    norm_code2 = normalize_code(code2)

    return difflib.SequenceMatcher(None, norm_code1, norm_code2).ratio()

# Function to calculate code similarity using TF-IDF and cosine similarity
def calculate_tfidf_similarity(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        code1 = f1.read()
        code2 = f2.read()

    norm_code1 = normalize_code(code1)
    norm_code2 = normalize_code(code2)

    code_list = [norm_code1, norm_code2]

    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(code_list)

    return cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])[0][0]

# Function to calculate hash-based similarity
def calculate_hash_similarity(file1, file2):
    BLOCK_SIZE = 65536
    hash1 = hashlib.sha256()
    hash2 = hashlib.sha256()

    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        while True:
            data1 = f1.read(BLOCK_SIZE)
            data2 = f2.read(BLOCK_SIZE)
            if not data1 or not data2:
                break
            hash1.update(data1)
            hash2.update(data2)

    return 1.0 - (abs(int(hash1.hexdigest(), 16) - int(hash2.hexdigest(), 16)) / 2**256)

# Define a threshold for plagiarism detection
plagiarism_threshold = 0.5 # Adjust as needed

# Check code similarity using multiple techniques
ast_similarity = calculate_ast_similarity(code_file1, code_file2)
tfidf_similarity = calculate_tfidf_similarity(code_file1, code_file2)
hash_similarity = calculate_hash_similarity(code_file1, code_file2)

# Aggregate similarity scores (you can customize this aggregation method)
similarity_scores = [ast_similarity, tfidf_similarity, hash_similarity]
mean_similarity = np.mean(similarity_scores)

# Check if the mean similarity exceeds the threshold
if mean_similarity >= plagiarism_threshold:
    print("Code plagiarism detected.")
else:
    print("No code plagiarism detected.")
