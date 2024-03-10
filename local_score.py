import difflib
from sentence_transformers import SentenceTransformer, util
from detoxify import Detoxify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Necessary installations:
# pip install sentence-transformers detoxify nltk

# Load models
similarity_model = SentenceTransformer('stsb-roberta-base', device="cpu")
context_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
spice_model = Detoxify('unbiased-small')

# Download necessary NLTK data
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Define a function to normalize the compound sentiment score
def get_sentiment(text) -> float:
    score = sia.polarity_scores(text)
    normalized_score = (score['compound'] + 1) / 2  # Normalizing to range [0,1]
    return normalized_score

sentiment_weight = 2  # Increase this value to weigh sentiment delta more heavily

# Define original text and its variations
original_text = "We live in an advertising hellscape now. The biggest crock of crap is being forced to watch an advertisement at the freaking gas station as I pump my gas. I'll never voluntarily use those pumps ever again."
worst_terms =  [
    "hellscape",
    "crock of crap",
    "freaking"
]
variations = {
    "worst_fix": "We live in an advertising-heavy environment now. The most bothersome thing is being forced to watch an advertisement at the gas station as I pump my gas. I'll avoid using those pumps in the future.",
    "nvc": "I feel overwhelmed by the amount of advertising in our environment now. It bothers me to have to watch an advertisement while pumping gas at the gas station. I prefer to use pumps without advertisements going forward.",
    "constructive": "The prevalence of advertising in our environment can feel overwhelming at times. Having to watch ads while pumping gas is particularly bothersome to me. I would appreciate less adverts at gas stations for a less distracting, peaceful experience.",
    "hybrid": "We live in an advertising-heavy environment now. Having to watch an advertisement at the gas station while pumping gas is quite bothersome. I'll avoid using those pumps when possible going forward.",
    "beta": "We live in an advertising-heavy landscape now. The biggest frustration is being forced to watch an advertisement at the annoying gas station as I pump my gas. I'll never voluntarily use those pumps ever again.",
    "identity":"We live in an advertising hellscape now. The biggest crock of crap is being forced to watch an advertisement at the freaking gas station as I pump my gas. I'll never voluntarily use those pumps ever again."
}

# Identify replacements using difflib for 'worst_fix' variation
matcher = difflib.SequenceMatcher(None, original_text.split(), variations['worst_fix'].split())
replacements = {}
for opcode in matcher.get_opcodes():
    if opcode[0] == 'replace':
        original_phrase = ' '.join(original_text.split()[opcode[1]:opcode[2]])
        new_phrase = ' '.join(variations['worst_fix'].split()[opcode[3]:opcode[4]])
        replacements[original_phrase] = new_phrase

print("Replacements found:", replacements)

# Function to calculate an aggregate score for each variant
def calculate_aggregate_score(overall_similarity, negativity_score, sentiment_delta, edit_distance):
    # Here we can define how we want to combine the scores
    # For simplicity, we'll just sum them up
    # Smaller edit distances are better, so we subtract the normalized edit distance from 1 to invert it
    # Adjust the weight of the edit distance to favor it less
    edit_distance_weight = 0.5  # Reduce this value to weigh edit distance less heavily
    normalized_edit_distance = (1 - (edit_distance / max(len(original_text.split()), 1))) * edit_distance_weight
    return overall_similarity - negativity_score + sentiment_delta + (normalized_edit_distance * edit_distance_weight)

# Dictionary to hold variant names and their aggregate scores
variant_scores = {}

def calculate_negativity_score(predictions):
    weights = {
        'toxicity': 0.5,
        'severe_toxicity': 0.2,
        'obscene': 0.1,
        'threat': 0.1,
        'insult': 0.05,
        'identity_attack': 0.05
    }
    score = sum(predictions[key] * weight for key, weight in weights.items())
    return score

# Calculate similarity for replacements and overall similarity for all variations
for name, text in sorted(variations.items(), key=lambda item: calculate_aggregate_score(
    util.pytorch_cos_sim(context_model.encode(original_text, convert_to_tensor=True),
                         context_model.encode(item[1], convert_to_tensor=True)).item(),
    calculate_negativity_score(spice_model.predict(item[1])),
    (get_sentiment(item[1]) - get_sentiment(original_text)) * sentiment_weight,
    nltk.edit_distance(original_text, item[1])), reverse=True):
    # Compute overall semantic similarity
    original_embedding = context_model.encode(original_text, convert_to_tensor=True)
    variation_embedding = context_model.encode(text, convert_to_tensor=True)
    overall_similarity = util.pytorch_cos_sim(original_embedding, variation_embedding).item()

    # Calculate negativity score using Detoxify
    spice_scores = spice_model.predict(text)
    negativity_score = calculate_negativity_score(spice_scores)

    # Calculate sentiment score delta
    sentiment_delta = (get_sentiment(text) - get_sentiment(original_text)) * sentiment_weight

    print(f"\nVariation: {name}")
    print(f"Overall similarity: {overall_similarity:.4f}")
    print(f"Negativity score: {negativity_score:.4f}")
    print(f"Weighted sentiment delta: {sentiment_delta:.4f}")

    # Calculate and store the aggregate score
    edit_distance = nltk.edit_distance(original_text, text)
    aggregate_score = calculate_aggregate_score(overall_similarity, negativity_score, sentiment_delta, edit_distance)
    variant_scores[name] = aggregate_score

# Print the sorted variants and their scores
print("\nSorted Variants by Aggregate Score:")
for name in sorted(variant_scores, key=variant_scores.get, reverse=True):
    print(f"{name}: {variant_scores[name]:.4f}")

"""
Example output:

Replacements found: {'advertising hellscape': 'advertising-heavy environment', 'biggest crock of crap': 'most bothersome thing', 'never voluntarily use': 'avoid using', 'ever again.': 'in the future.'}

Variation: beta
Overall similarity: 0.8454
Negativity score: 0.0004
Weighted sentiment delta: -0.0190

Variation: constructive
Overall similarity: 0.7353
Negativity score: 0.0002
Weighted sentiment delta: 1.0908

Variation: identity
Overall similarity: 1.0000
Negativity score: 0.5117
Weighted sentiment delta: 0.0000

Variation: nvc
Overall similarity: 0.7693
Negativity score: 0.0002
Weighted sentiment delta: 0.6595

Variation: worst_fix
Overall similarity: 0.8584
Negativity score: 0.0004
Weighted sentiment delta: 0.0348

Variation: hybrid
Overall similarity: 0.7776
Negativity score: 0.0002
Weighted sentiment delta: 0.1931

Sorted Variants by Aggregate Score:
beta: 0.9071
constructive: 0.9002
identity: 0.7383
nvc: 0.7192
worst_fix: 0.6563
hybrid: 0.4637

"""
