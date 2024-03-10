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
    "beta": "We live in an advertising-heavy landscape now. The biggest frustration is being forced to watch an advertisement at the gas station as I pump my gas. I'll never voluntarily use those pumps ever again.",
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
def calculate_aggregate_score(overall_similarity, negativity_score, sentiment_delta, edit_distance, max_length, name):
    negativity_weight=1
    sentiment_weight=1
    edit_distance_weight=0.7
    normalized_edit_distance = (edit_distance / max(max_length, 1))

    # Combine the scores, taking into account the weights for each component
    return overall_similarity - (negativity_score * negativity_weight) + (sentiment_delta*sentiment_weight) - (normalized_edit_distance*edit_distance_weight)

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
original_embedding = context_model.encode(original_text, convert_to_tensor=True)
original_text_negativity_score = calculate_negativity_score(spice_model.predict(original_text))
for name, text in variations.items():
    # Compute overall semantic similarity
    variation_embedding = context_model.encode(text, convert_to_tensor=True)
    overall_similarity = util.pytorch_cos_sim(original_embedding, variation_embedding).item()

    # Calculate negativity score using Detoxify
    negativity_score = calculate_negativity_score(spice_model.predict(text))
    negativity_score_delta = original_text_negativity_score - negativity_score # unused

    # Calculate sentiment score delta
    sentiment_delta = get_sentiment(text) - get_sentiment(original_text)

    # Calculate the maximum length between the original and variation texts for normalization
    max_length = max(len(original_text), len(text))

    # Calculate and store the aggregate score
    edit_distance = nltk.edit_distance(original_text, text)
    aggregate_score = calculate_aggregate_score(overall_similarity, negativity_score, sentiment_delta, edit_distance, max_length,name=name)
    variant_scores[name] = {
        "overall_similarity": overall_similarity,
        "negativity_score": negativity_score,
        "sentiment_delta": sentiment_delta,
        "edit_distance": edit_distance,
        "max_length": max_length,
        "aggregate_score": aggregate_score,
        "variant_text": text
    }

# Sort the variants by aggregate score
sorted_variants = sorted(variant_scores.items(), key=lambda x: x[1]['aggregate_score'], reverse=True)

for name, score in sorted_variants:
    print(f"\nVariation: {name}")
    print(f"Aggregate score: {variant_scores[name]['aggregate_score']:.4f}")
    print(f"Negativity score: {variant_scores[name]['negativity_score']:.4f}")
    print(f"Sentiment delta: {variant_scores[name]['sentiment_delta']:.4f}")
    print(f"Edit distance: {variant_scores[name]['edit_distance']}")
    print(f"Variant text: `{variant_scores[name]['variant_text']}`\n")

"""
Example output:

Replacements found: {'advertising hellscape': 'advertising-heavy environment', 'biggest crock of crap': 'most bothersome thing', 'never voluntarily use': 'avoid using', 'ever again.': 'in the future.'}

Variation: beta
Aggregate score: 0.7957
Negativity score: 0.0003
Sentiment delta: 0.0428
Edit distance: 29
Variant text: `We live in an advertising-heavy landscape now. The biggest frustration is being forced to watch an advertisement at the gas station as I pump my gas. I'll never voluntarily use those pumps ever again.`


Variation: constructive
Aggregate score: 0.7833
Negativity score: 0.0002
Sentiment delta: 0.5454
Edit distance: 174
Variant text: `The prevalence of advertising in our environment can feel overwhelming at times. Having to watch ads while pumping gas is particularly bothersome to me. I would appreciate less adverts at gas stations for a less distracting, peaceful experience.`


Variation: nvc
Aggregate score: 0.6408
Negativity score: 0.0002
Sentiment delta: 0.3297
Edit distance: 142
Variant text: `I feel overwhelmed by the amount of advertising in our environment now. It bothers me to have to watch an advertisement while pumping gas at the gas station. I prefer to use pumps without advertisements going forward.`


Variation: worst_fix
Aggregate score: 0.6295
Negativity score: 0.0004
Sentiment delta: 0.0174
Edit distance: 72
Variant text: `We live in an advertising-heavy environment now. The most bothersome thing is being forced to watch an advertisement at the gas station as I pump my gas. I'll avoid using those pumps in the future.`


Variation: hybrid
Aggregate score: 0.4914
Negativity score: 0.0002
Sentiment delta: 0.0965
Edit distance: 112
Variant text: `We live in an advertising-heavy environment now. Having to watch an advertisement at the gas station while pumping gas is quite bothersome. I'll avoid using those pumps when possible going forward.`


Variation: identity
Aggregate score: 0.4883
Negativity score: 0.5117
Sentiment delta: 0.0000
Edit distance: 0
Variant text: `We live in an advertising hellscape now. The biggest crock of crap is being forced to watch an advertisement at the freaking gas station as I pump my gas. I'll never voluntarily use those pumps ever again.`

"""
