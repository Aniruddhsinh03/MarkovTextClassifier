import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from collections import Counter

# Input files: Add paths to your text files
input_files = [
    'edgar_allan_poe.txt',  # Class 0
    'robert_frost.txt'      # Class 1
]

# -----------------------------
# Data Preparation
# -----------------------------
texts = []       # Store all text lines
labels = []      # Store corresponding labels

print("Data Preparation: Reading files and assigning labels...")
for label, file in enumerate(input_files):
    print(f"Reading '{file}' as label {label}")
    with open(file, 'r') as f:
        for line in f:
            line = line.rstrip().lower()
            if line:  # Only process non-empty lines
                line = line.translate(str.maketrans('', '', string.punctuation))
                texts.append(line)
                labels.append(label)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# -----------------------------
# Vocabulary Creation with N-Grams
# -----------------------------
def generate_ngrams(sentence, n=1):
    """Generate n-grams for a sentence."""
    tokens = sentence.split()
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

print("Building vocabulary with N-grams (unigrams + bigrams)...")
vocab = {'<UNK>': 0}  # Unknown word token
current_idx = 1
ngram_size = 2  # Use unigrams and bigrams
min_freq = 2    # Minimum frequency to include word in vocab

# Build vocabulary from training data
counter = Counter()
for sentence in X_train:
    for n in range(1, ngram_size + 1):
        counter.update(generate_ngrams(sentence, n=n))

# Prune vocabulary based on frequency
for word, freq in counter.items():
    if freq >= min_freq:
        vocab[word] = current_idx
        current_idx += 1

# -----------------------------
# Convert Text to Integer Format
# -----------------------------
def text_to_int(text_data, vocab_dict, ngram_size=1):
    """Convert text data into integers based on the vocabulary."""
    int_data = []
    for sentence in text_data:
        sentence_ngrams = []
        for n in range(1, ngram_size + 1):
            sentence_ngrams += generate_ngrams(sentence, n=n)
        int_sentence = [vocab_dict.get(token, 0) for token in sentence_ngrams]
        int_data.append(int_sentence)
    return int_data

X_train_int = text_to_int(X_train, vocab, ngram_size=ngram_size)
X_test_int = text_to_int(X_test, vocab, ngram_size=ngram_size)

# -----------------------------
# Transition and Initial Probability Matrices with Smoothing
# -----------------------------
vocab_size = len(vocab)
alpha = 1  # Laplace smoothing factor

A_class0 = np.ones((vocab_size, vocab_size)) * alpha
pi_class0 = np.ones(vocab_size) * alpha

A_class1 = np.ones((vocab_size, vocab_size)) * alpha
pi_class1 = np.ones(vocab_size) * alpha

def compute_counts(data, A, pi):
    """Count transitions and initial words."""
    for sentence in data:
        prev_idx = None
        for idx in sentence:
            if prev_idx is None:
                pi[idx] += 1
            else:
                A[prev_idx, idx] += 1
            prev_idx = idx

compute_counts([x for x, y in zip(X_train_int, y_train) if y == 0], A_class0, pi_class0)
compute_counts([x for x, y in zip(X_train_int, y_train) if y == 1], A_class1, pi_class1)

# Normalize Probabilities
A_class0 /= A_class0.sum(axis=1, keepdims=True)
pi_class0 /= pi_class0.sum()

A_class1 /= A_class1.sum(axis=1, keepdims=True)
pi_class1 /= pi_class1.sum()

# Convert to Log Probabilities
log_A0 = np.log(A_class0)
log_pi0 = np.log(pi_class0)

log_A1 = np.log(A_class1)
log_pi1 = np.log(pi_class1)

# -----------------------------
# Class Priors (Balanced)
# -----------------------------
class_prior0 = np.log(sum(y == 0 for y in y_train) / len(y_train))
class_prior1 = np.log(sum(y == 1 for y in y_train) / len(y_train))

# -----------------------------
# Classifier Definition
# -----------------------------
class MarkovTextClassifier:
    def __init__(self, log_transition_probs, log_initial_probs, class_priors):
        self.log_A = log_transition_probs
        self.log_pi = log_initial_probs
        self.class_priors = class_priors

    def _compute_log_likelihood(self, sentence, class_idx):
        log_A = self.log_A[class_idx]
        log_pi = self.log_pi[class_idx]

        likelihood = 0
        prev_idx = None
        for idx in sentence:
            if prev_idx is None:
                likelihood += log_pi[idx]
            else:
                likelihood += log_A[prev_idx, idx]
            prev_idx = idx
        return likelihood

    def predict(self, data):
        predictions = []
        for sentence in data:
            log_likelihood_class0 = self._compute_log_likelihood(sentence, 0) + self.class_priors[0]
            log_likelihood_class1 = self._compute_log_likelihood(sentence, 1) + self.class_priors[1]
            predictions.append(0 if log_likelihood_class0 > log_likelihood_class1 else 1)
        return predictions

# -----------------------------
# Train and Evaluate Classifier
# -----------------------------
classifier = MarkovTextClassifier(
    log_transition_probs=[log_A0, log_A1],
    log_initial_probs=[log_pi0, log_pi1],
    class_priors=[class_prior0, class_prior1]
)

print("\nEvaluating Model...")
train_preds = classifier.predict(X_train_int)
test_preds = classifier.predict(X_test_int)

# Ensure predictions and labels are numpy arrays
train_preds = np.array(train_preds)
y_train = np.array(y_train)
test_preds = np.array(test_preds)
y_test = np.array(y_test)

# Calculate accuracies
train_acc = np.mean(train_preds == y_train)
test_acc = np.mean(test_preds == y_test)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")

# Confusion Matrix and F1 Score
print("\nTrain Confusion Matrix:")
print(confusion_matrix(y_train, train_preds))

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, test_preds))

print("\nTrain F1 Score:", f1_score(y_train, train_preds))
print("Test F1 Score:", f1_score(y_test, test_preds))
