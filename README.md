# MarkovQuillClassifier

**A Markov Chain-based text classification tool to identify authorship of literary works using n-grams and probabilistic modeling.**

---

## Overview
MarkovQuillClassifier leverages **Markov Chains** and **N-grams** (unigrams and bigrams) to analyze word transitions and patterns in literary texts. It identifies the author of a given text by learning the unique writing style of each author based on transition probabilities.

This project includes:
- **N-Gram Features**: Extracts unigrams and bigrams for improved context.
- **Laplace Smoothing**: Handles unseen word transitions.
- **Class Balancing**: Mitigates class imbalance for better accuracy.
- **Performance Metrics**: Reports accuracy, confusion matrix, and F1-score for evaluation.

---

## Key Features
- **Probabilistic Modeling**: Uses Markov Chains to model word transitions.
- **Custom Vocabulary**: Builds vocabulary dynamically with N-grams.
- **Performance Evaluation**: Outputs accuracy, confusion matrix, and F1-score.
- **Lightweight**: Built with Python and minimal dependencies.

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/MarkovQuillClassifier.git
   cd MarkovQuillClassifier
   ```

2. Install required libraries:
   ```bash
   pip install numpy scikit-learn
   ```

3. Add your text files (e.g., `edgar_allan_poe.txt`, `robert_frost.txt`) in the project directory.

---

## Usage
Run the following command to train and evaluate the model:
```bash
python nlp_test1.py
```

### Sample Output:
```
Training Accuracy: 93.73%
Testing Accuracy: 81.90%

Train Confusion Matrix:
[[ 462  108]
 [   0 1153]]

Test Confusion Matrix:
[[ 71  77]
 [  1 282]]

Test F1 Score: 0.8785
```

---

## Applications
- **Authorship Attribution**: Identify the author of anonymous or disputed texts.
- **Text Classification**: Classify poems, articles, or literary works.
- **Plagiarism Detection**: Detect stylistic similarities in text corpora.

---

## Future Improvements
- Add support for **trigrams** to enhance context capture.
- Implement **TF-IDF** for better word weighting.
- Experiment with advanced models like **Naive Bayes** or **Logistic Regression**.
- Expand to classify multiple authors.

---

## License
This project is licensed under the **MIT License**.
