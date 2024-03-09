import difflib
from sentence_transformers import SentenceTransformer, util
from detoxify import Detoxify

# Load models
similarity_model = SentenceTransformer('stsb-roberta-base', device="cpu")
context_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
spice_model= Detoxify('unbiased-small')

# Example: spice_model.predict("You suck, you did it wrong!")
# Returns:
{'toxicity': 0.9,
 'severe_toxicity': 0.08,
 'obscene': 0.9,
 'threat': 0.003,
 'insult': 0.7,
 'identity_attack': 0.007} 

# Define the original text and variations
original_text = "We live in an advertising hellscape now. The biggest crock of crap is being forced to watch an advertisement at the freaking gas station as I pump my gas. I'll never voluntarily use those pumps ever again."
worst_terms =  [
    "hellscape",
    "crock of crap",
    "freaking"
]
variations = {
    "worst_fix": "We live in an advertising-heavy environment now. The most bothersome thing is being forced to watch an advertisement at the gas station as I pump my gas. I'll avoid using those pumps in the future.",
    "nvc": "I feel overwhelmed by the amount of advertising in our environment now. It bothers me to have to watch an advertisement while pumping gas at the gas station. I prefer to use pumps without advertisements going forward.",
    "constructive": "The prevalence of advertising in our environment can feel overwhelming at times. Having to watch ads while pumping gas is particularly bothersome to me. I would appreciate more ad-free options at gas stations for a less distracting experience.",
    "hybrid": "We live in an advertising-heavy environment now. Having to watch an advertisement at the gas station while pumping gas is quite bothersome. I'll avoid using those pumps when possible going forward."
}

# Identify replacements using difflib
matcher = difflib.SequenceMatcher(None, original_text.split(), variations['worst_fix'].split())
replacements = {}
for opcode in matcher.get_opcodes():
    if opcode[0] == 'replace':
        original_phrase = ' '.join(original_text.split()[opcode[1]:opcode[2]])
        new_phrase = ' '.join(variations['worst_fix'].split()[opcode[3]:opcode[4]])
        replacements[original_phrase] = new_phrase

print("Replacements found:", replacements)

# Calculate similarity for replacements
for original, replaced in replacements.items():
    original_embedding = similarity_model.encode(original, convert_to_tensor=True)
    replaced_embedding = similarity_model.encode(replaced, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(original_embedding, replaced_embedding)
    print(f"Similarity between '{original}' and its replacement '{replaced}': {similarity.item():.4f}")

# Calculate overall similarity between original and variations
for name, text in variations.items():
    original_embedding = context_model.encode(original_text, convert_to_tensor=True)
    variation_embedding = context_model.encode(text, convert_to_tensor=True)
    overall_similarity = util.pytorch_cos_sim(original_embedding, variation_embedding)
    print(f"Overall similarity ({name}): {overall_similarity.item():.4f}")

# Calculate edit distance for each variation
for name, text in variations.items():
    edit_distance = difflib.SequenceMatcher(None, original_text, text).ratio()
    print(f"Edit distance ({name}): {edit_distance:.4f}")

# Calculate spice score for each variation
original_spice_score = sum(spice_model.predict(original_text).values())
normalized_original_spice_score = original_spice_score / 6
print(f"Normalized spice score (original): {normalized_original_spice_score:.4f}")
weights = {'similarity_weight': 0.5, 'spice_change_weight': 0.5}
best_variation = None
best_variation_score = -float('inf')
for name, text in variations.items():
    spice_score = sum(spice_model.predict(text).values())
    normalized_spice_score = spice_score / 6
    spice_score_change = spice_score - original_spice_score
    normalized_spice_score_change = normalized_spice_score - normalized_original_spice_score
    print(f"Normalized spice score ({name}): {normalized_spice_score:.4f} (change: {normalized_spice_score_change:.2f})")
    # Calculate the weighted sum for each variation
    overall_similarity = util.pytorch_cos_sim(original_embedding, variation_embedding)
    weighted_sum = (weights['similarity_weight'] * overall_similarity.item() +
                    weights['spice_change_weight'] * (1 - normalized_spice_score_change))
    print(f"Weighted sum ({name}): {weighted_sum:.4f}")
    # Determine the best variation
    if weighted_sum > best_variation_score:
        best_variation_score = weighted_sum
        best_variation = name

# Output the best variation
print(f"The best variation is '{best_variation}' with a weighted sum of {best_variation_score:.4f}")

"""
Output example:


Replacements found: {'advertising hellscape': 'advertising-heavy environment', 'biggest crock of crap': 'most bothersome thing', 'never voluntarily use': 'avoid using', 'ever again.': 'in the future.'}
Similarity between 'advertising hellscape' and its replacement 'advertising-heavy environment': 0.4981
Similarity between 'biggest crock of crap' and its replacement 'most bothersome thing': 0.6168
Similarity between 'never voluntarily use' and its replacement 'avoid using': 0.6128
Similarity between 'ever again.' and its replacement 'in the future.': 0.3310
Overall similarity (worst_fix): 0.8584
Overall similarity (nvc): 0.7693
Overall similarity (constructive): 0.6835
Overall similarity (hybrid): 0.7776
Edit distance (worst_fix): 0.7562
Edit distance (nvc): 0.0521
Edit distance (constructive): 0.0357
Edit distance (hybrid): 0.6318
Normalized spice score (original): 0.2951
Normalized spice score (worst_fix): 0.0002 (change: -0.29)
Weighted sum (worst_fix): 1.0362
Normalized spice score (nvc): 0.0001 (change: -0.30)
Weighted sum (nvc): 1.0363
Normalized spice score (constructive): 0.0001 (change: -0.30)
Weighted sum (constructive): 1.0363
Normalized spice score (hybrid): 0.0001 (change: -0.29)
Weighted sum (hybrid): 1.0363
The best variation is 'nvc' with a weighted sum of 1.0363
"""