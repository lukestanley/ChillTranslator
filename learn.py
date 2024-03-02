# %%
import pandas as pd
from datasets import load_dataset
from detoxify import Detoxify
predict_model = Detoxify('original-small')
dataset = load_dataset("tasksource/jigsaw")

train_data = dataset['train'] 
print('length',len(train_data)) # length 159571
print(train_data[0]) # {'id': '0000997932d777bf', 'comment_text': "Explanation\nWhy the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27", 'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 0}

small_subset = train_data[:2000]

predict_model.predict("You suck, that is not Markdown!") # Also accepts an array of strings, returning an single dict of arrays of predictions.
# Returns:
{'toxicity': 0.98870254,
 'severe_toxicity': 0.087154716,
 'obscene': 0.93440753,
 'threat': 0.0032278204,
 'insult': 0.7787105,
 'identity_attack': 0.007936229}

# %%
import asyncio
import json
import time
import os
import hashlib
from functools import wraps


_in_memory_cache = {}

def handle_cache(prefix, func, *args, _result=None, **kwargs):
    # Generate a key based on function name and arguments
    key = f"{func.__name__}_{args}_{kwargs}"
    hashed_key = hashlib.sha1(key.encode()).hexdigest()
    cache_filename = f"{prefix}_{hashed_key}.json"

    # Check the in-memory cache first
    if key in _in_memory_cache:
        return _in_memory_cache[key]

    # Check if cache file exists and read data
    if os.path.exists(cache_filename):
        with open(cache_filename, 'r') as file:
            #print("Reading from cache file with prefix", prefix)
            _in_memory_cache[key] = json.load(file)
            return _in_memory_cache[key]

    # If result is not provided (for sync functions), compute it
    if _result is None:
        _result = func(*args, **kwargs)

    # Update the in-memory cache and write it to the file
    _in_memory_cache[key] = _result
    with open(cache_filename, 'w') as file:
        json.dump(_result, file)

    return _result


def acache(prefix):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate a key based on function name and arguments
            key = f"{func.__name__}_{args}_{kwargs}"
            hashed_key = hashlib.sha1(key.encode()).hexdigest()
            cache_filename = f"{prefix}_{hashed_key}.json"

            # Check the in-memory cache first
            if key in _in_memory_cache:
                return _in_memory_cache[key]

            # Check if cache file exists and read data
            if os.path.exists(cache_filename):
                with open(cache_filename, 'r') as file:
                    _in_memory_cache[key] = json.load(file)
                    return _in_memory_cache[key]

            # Await the function call and get the result
            print("Computing result for async function")
            result = await func(*args, **kwargs)

            # Update the in-memory cache and write it to the file
            _in_memory_cache[key] = result
            with open(cache_filename, 'w') as file:
                json.dump(result, file)

            return result

        return wrapper
    return decorator


def cache(prefix):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Direct call to the shared cache handling function
            return handle_cache(prefix, func, *args, **kwargs)
        return wrapper
    return decorator

def timeit(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)  # Awaiting the async function
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.1f} seconds to run.")
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Calling the sync function
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.1f} seconds to run.")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper



# %%

@cache("toxicity")
def cached_toxicity_prediction(comments):
    data = predict_model.predict(comments)
    return data

def predict_toxicity(comments, batch_size=4):
    """
    Predicts toxicity scores for a list of comments.
    
    Args:
    - comments: List of comment texts.
    - batch_size: Size of batches for prediction to manage memory usage.
    
    Returns:
    A DataFrame with the original comments and their predicted toxicity scores.
    """
    results = {'comment_text': [], 'toxicity': [], 'severe_toxicity': [], 'obscene': [], 'threat': [], 'insult': [], 'identity_attack': []}
    for i in range(0, len(comments), batch_size):
        batch_comments = comments[i:i+batch_size]
        predictions = cached_toxicity_prediction(batch_comments)
        # We convert the JSON serializable data back to a DataFrame:
        results['comment_text'].extend(batch_comments)
        for key in predictions.keys():
            results[key].extend(predictions[key])
    return pd.DataFrame(results)

# Predict toxicity scores for the small subset of comments:
#small_subset_predictions = predict_toxicity(small_subset['comment_text'][4])
# Let's just try out 4 comments with cached_toxicity_prediction:
small_subset['comment_text'][0:1]

# %%
small_subset_predictions=predict_toxicity(small_subset['comment_text'][0:200])

# %%
small_subset_predictions

# %%
def filter_comments(dataframe, toxicity_threshold=0.2, severe_toxicity_threshold=0.4):
    """
    Filters comments based on specified thresholds for toxicity, severe toxicity.
    
    Args:
    - dataframe: DataFrame containing comments and their toxicity scores.
    - toxicity_threshold: Toxicity score threshold.
    - severe_toxicity_threshold: Severe toxicity score threshold.
    - identity_attack_threshold: Identity attack score threshold.
    
    Returns:
    DataFrame filtered based on the specified thresholds.
    """
    identity_attack_threshold = 0.5
    insult_threshold = 0.3
    obscene_threshold = 0.6
    threat_threshold = 0.3
    filtered_df = dataframe[
        (dataframe['toxicity'] >= toxicity_threshold) &
        #(dataframe['toxicity'] < 1.0) &  # Ensure comments are spicy but not 100% toxic
        (dataframe['severe_toxicity'] < severe_toxicity_threshold) &
        (dataframe['identity_attack'] < identity_attack_threshold) &
        (dataframe['insult'] < insult_threshold) &
        (dataframe['obscene'] < obscene_threshold) &
        (dataframe['threat'] < threat_threshold)

    ]
    return filtered_df

spicy_comments = filter_comments(small_subset_predictions)


# Lets sort spicy comments by combined toxicity score:
spicy_comments.sort_values(by=['toxicity', 'severe_toxicity'], ascending=True, inplace=True)

# Print the spicy comments comment_text and their toxicity scores as a formatted string:
for index, row in spicy_comments.iterrows():
    print(f"Comment: `{row['comment_text']}` \n Toxiciy: {(row['toxicity'] + row['severe_toxicity']) / 2 * 100:.0f}% \n")
