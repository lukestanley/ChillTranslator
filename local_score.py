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
def calculate_aggregate_score(overall_similarity, negativity_score, sentiment_delta, edit_distance, max_length, name, sentiment):
    negativity_weight=3
    sentiment_weight=1
    edit_distance_weight=1
    similarity_weight=0.8
    
    normalized_edit_distance = edit_distance / max(max_length, 1)
    weighted_similarity = overall_similarity * similarity_weight
    weighted_negativity = negativity_score * negativity_weight
    weighted_happy_sentiment = (sentiment * sentiment_weight)
    weighted_edit_distance = normalized_edit_distance * edit_distance_weight
    result =  weighted_similarity - weighted_edit_distance - weighted_negativity + weighted_happy_sentiment
    return result

# Dictionary to hold variant names and their aggregate scores
variant_scores = {}

def calculate_negativity_score(predictions):
    score = sum(predictions.values()) / len(predictions)
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
    variant_sentiment = get_sentiment(text)
    sentiment_delta = variant_sentiment - get_sentiment(original_text)

    # Calculate the maximum length between the original and variation texts for normalization
    max_length = max(len(original_text), len(text))

    # Calculate and store the aggregate score
    edit_distance = nltk.edit_distance(original_text, text)
    aggregate_score = calculate_aggregate_score(overall_similarity, negativity_score, sentiment_delta, edit_distance, max_length,name=name, sentiment=variant_sentiment)
    variant_scores[name] = {
        "overall_similarity": overall_similarity,
        "negativity_score": negativity_score,
        "sentiment_delta": sentiment_delta,
        "sentiment": variant_sentiment,
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
    print(f"Sentiment: {variant_scores[name]['sentiment']:.4f}")
    print(f"Sentiment delta: {variant_scores[name]['sentiment_delta']:.4f}")
    print(f"Edit distance: {variant_scores[name]['edit_distance']}")
    print(f"Variant text: `{variant_scores[name]['variant_text']}`\n")

"""
Example output:

Replacements found: {'advertising hellscape': 'advertising-heavy environment', 'biggest crock of crap': 'most bothersome thing', 'never voluntarily use': 'avoid using', 'ever again.': 'in the future.'}

Variation: beta
Aggregate score: 0.6764
Negativity score: 0.0001
Sentiment: 0.1366
Sentiment delta: 0.0428
Edit distance: 29
Variant text: `We live in an advertising-heavy landscape now. The biggest frustration is being forced to watch an advertisement at the gas station as I pump my gas. I'll never voluntarily use those pumps ever again.`


Variation: constructive
Aggregate score: 0.5169
Negativity score: 0.0001
Sentiment: 0.6391
Sentiment delta: 0.5454
Edit distance: 174
Variant text: `The prevalence of advertising in our environment can feel overwhelming at times. Having to watch ads while pumping gas is particularly bothersome to me. I would appreciate less adverts at gas stations for a less distracting, peaceful experience.`


Variation: worst_fix
Aggregate score: 0.4461
Negativity score: 0.0002
Sentiment: 0.1111
Sentiment delta: 0.0174
Edit distance: 72
Variant text: `We live in an advertising-heavy environment now. The most bothersome thing is being forced to watch an advertisement at the gas station as I pump my gas. I'll avoid using those pumps in the future.`


Variation: nvc
Aggregate score: 0.3843
Negativity score: 0.0001
Sentiment: 0.4234
Sentiment delta: 0.3297
Edit distance: 142
Variant text: `I feel overwhelmed by the amount of advertising in our environment now. It bothers me to have to watch an advertisement while pumping gas at the gas station. I prefer to use pumps without advertisements going forward.`


Variation: hybrid
Aggregate score: 0.2656
Negativity score: 0.0001
Sentiment: 0.1902
Sentiment delta: 0.0965
Edit distance: 112
Variant text: `We live in an advertising-heavy environment now. Having to watch an advertisement at the gas station while pumping gas is quite bothersome. I'll avoid using those pumps when possible going forward.`


Variation: identity
Aggregate score: 0.1348
Negativity score: 0.2530
Sentiment: 0.0937
Sentiment delta: 0.0000
Edit distance: 0
Variant text: `We live in an advertising hellscape now. The biggest crock of crap is being forced to watch an advertisement at the freaking gas station as I pump my gas. I'll never voluntarily use those pumps ever again.`
"""
