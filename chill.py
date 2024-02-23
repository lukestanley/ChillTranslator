import json
import time
from pydantic import BaseModel, Field
from utils import query_ai_prompt

# This script uses the llama_cpp server to improve a text.
# To run this script, you need to do something like this:
# Download the model: https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf?download=true
# Rename it as needed.
# Install the server and start it:
# pip install llama-cpp-python[server] --upgrade
# python3 -m llama_cpp.server --model mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf  --port 5834 --n_ctx 4096 --use_mlock false
# Run this script:
# python3 chill.py
# This will then try and improve the text below:

original_text = """Stop chasing dreams instead. Life is not a Hollywood movie. Not everyone is going to get a famous billionaire. Adjust your expectations to reality, and stop thinking so highly of yourself, stop judging others. Assume the responsibility for the things that happen in your life. It is kind of annoying to read your text, it is always some external thing that "happened" to you, and it is always other people who are not up to your standards. At some moment you even declare with despair. And guess what? This is true and false at the same time, in a fundamental level most people are not remarkable, and you probably aren't too. But at the same time, nobody is the same, you have worth just by being, and other people have too. The impression I get is that you must be someone incredibly annoying to work with, and that your performance is not even nearly close to what you think it is, and that you really need to come down to earth. Stop looking outside, work on yourself instead. You'll never be satisfied just by changing jobs. Do therapy if you wish, become acquainted with stoicism, be a volunteer in some poor country, whatever, but do something to regain control of your life, to get some perspective, and to adjust your expectations to reality."""
# From elzbardico on https://news.ycombinator.com/item?id=36119858

# TODO: See README.md for the more plans.
# TODO: Segment the text into sentences
"""
import pysbd
sentences = pysbd.Segmenter(language="en", clean=False).segment(paragraph)
"""

global suggestions
suggestions = []

start_time = time.time()

class ImprovedText(BaseModel):
    text: str = Field(str, description="The improved text.")

class SpicyScore(BaseModel):
    spicy_score: float = Field(float, description="The spiciness score of the text.")

class Critique(BaseModel):
    critique: str = Field(str, description="The critique of the text.")

class FaithfulnessScore(BaseModel):
    faithfulness_score: float = Field(float, description="The faithfulness score of the text.")

improve_prompt = """
Your task is to rephrase inflammatory text, so it is more calm and constructive, without changing the intended meaning.
The improved text should have a softened tone, avoiding judgemental and extreme words.
Make sure the refined text is a good reflection of the original text, without adding new ideas.

1. Rather than accusations, share perspective.
2. Remove or soften judgemental language.
3. Focus on specific actions rather than character.
4. Rephrase extreme words like "always", "never" or "everyone" to be more moderate.
5. Focus on softening the tone, rather than changing the substance or meaning.
6. Use gentler alternatives to express similar points.
7. Don't add completely new ideas, ONLY build upon what's already there.
8 For example, you might reframe an existing point to be more balanced. Never introduce unrelated concepts.
9. Make everyone happy! Make them INFORMED and not *offended*. Make the original author to *content* that their points where *honoured* by your edit, by refining their text without loosing the original intent.

Example:
Example input text: "You're always annoying me. You never listen to me."
Example improved text output: {"text":"I am frustrated by your behaviour. Could you listen to me better?"}

End of example.
Here is the real input text to improve:
`{original_text}`

Previous rephrasing attempts:
{previous_suggestions}

Provide your improved version in this format:
{"text":"STRING"}
To get a good answer, make the original text non-inflamitory, while being as faithful to the ideas in the original text as much as possible. Use valid JSON then stop, do not add any remarks before or after the JSON.
"""

critique_prompt = """
Critique the text. We prefer the edit prevent inflaming discussions!
We also prefer concise text, and a similar semantic intent to the original.

Here is the original text:
`{original_text}`

Here is the text to critique:
`{last_edit}`

Output your response as valid JSON in this format:
{
    "critique":"STRING",
}

E.g:
{
    "critique":"This is too fluffy and different from the original intent."
}
Please critique the text."""


spicy_scorer_prompt = """
Score the text.

A calm spicy_score of 0 is ideal. A spicy_score of 1 is the worst, very inflammatory text that makes the reader feel attacked.

Here is the original text:
`{original_text}`

Here is the text to score:
`{last_edit}`
The float variable is scored from 0 to 1.

Output your response as valid JSON in this format, then stop:
{
    "spicy_score":FLOAT
}
Please score the text.
"""


faith_scorer_prompt = """
Score the text.

A score of 1 would have the same semantic intent as the original text. A score of 0 would mean the text has lost all semantic similarity.

Here is the original text:
`{original_text}`

Here is the new text to score:
`{last_edit}`

The float variable is scored from 0 to 1.

Output your response as valid JSON in this format, then stop:
{
    "faithfulness_score":FLOAT
}
Please score the text.
"""



def improve_text():
    global suggestions
    replacements = {
        "original_text": json.dumps(original_text),
        "previous_suggestions": json.dumps(suggestions, indent=2),
    }
    resp_json = query_ai_prompt(improve_prompt, replacements, ImprovedText)
    #print('resp_json', resp_json)
    return resp_json["text"]


def critique_text(last_edit):
    replacements = {"original_text": original_text, "last_edit": last_edit}

    # Query the AI for each of the new prompts separately

    critique_resp = query_ai_prompt(
        critique_prompt, replacements, Critique 
    )
    faithfulness_resp = query_ai_prompt(
        faith_scorer_prompt, replacements, FaithfulnessScore
    )
    spiciness_resp = query_ai_prompt(
        spicy_scorer_prompt, replacements, SpicyScore
    )

    # Combine the results from the three queries into a single dictionary
    combined_resp = {
        "critique": critique_resp["critique"],
        "faithfulness_score": faithfulness_resp["faithfulness_score"],
        "spicy_score": spiciness_resp["spicy_score"],
    }

    return combined_resp


def calculate_overall_score(faithfulness, spiciness):
    baseline_weight = 0.8
    overall = faithfulness + (1 - baseline_weight) * spiciness * faithfulness
    return overall


def should_stop(
    iteration,
    overall_score,
    time_used,
    min_iterations=2,
    min_overall_score=0.85,
    max_seconds=60,
):
    good_attempt = iteration >= min_iterations and overall_score >= min_overall_score
    too_long = time_used > max_seconds and overall_score >= 0.7
    return good_attempt or too_long


def update_suggestions(critique_dict):
    global suggestions
    critique_dict["overall_score"] = round(
        calculate_overall_score(
            critique_dict["faithfulness_score"], critique_dict["spicy_score"]
        ),
        2,
    )
    critique_dict["edit"] = last_edit
    suggestions.append(critique_dict)
    suggestions = sorted(suggestions, key=lambda x: x["overall_score"], reverse=True)[
        :2
    ]


def print_iteration_result(iteration, overall_score, time_used):
    global suggestions
    print(
        f"Iteration {iteration}: overall_score={overall_score:.2f}, time_used={time_used:.2f} seconds."
    )
    print("suggestions:")
    print(json.dumps(suggestions, indent=2))


max_iterations = 20


for iteration in range(1, max_iterations + 1):
    try:
        if iteration % 2 == 1:
            last_edit = improve_text()
        else:
            critique_dict = critique_text(last_edit)
            update_suggestions(critique_dict)
            overall_score = critique_dict["overall_score"]
            time_used = time.time() - start_time

            print_iteration_result(iteration, overall_score, time_used)

            if should_stop(iteration, overall_score, time_used):
                print(
                    "Stopping\nTop suggestion:\n", json.dumps(suggestions[0], indent=4)
                )
                break
    except ValueError as e:
        print("ValueError:", e)
        continue

"""
Outputs something like this:
 {
    "critique": "The revised text effectively conveys the same message as the original but in a more constructive and diplomatic tone, maintaining the original's intention while promoting a more positive discussion.",
    "faithfulness_score": 0.85,
    "spicy_score": 0.25,
    "overall_score": 0.89,
    "edit": "Consider shifting your focus from chasing dreams to finding fulfillment in reality. Life isn't a Hollywood movie, and becoming a famous billionaire isn't a realistic goal for everyone. It might be helpful to recalibrate your expectations to better align with what's possible. Instead of judging others, try to understand them better. Take responsibility for the events in your life, rather than attributing them to external factors or other people. I understand that it can be frustrating when things don't go as planned, but keep in mind that most people, including yourself, are not inherently exceptional or unremarkable. However, everyone has unique worth that doesn't depend on their achievements or status. It's essential to recognize that you may come across as demanding to work with and that your self-perception might not match others' opinions of your performance. To gain a fresh perspective and adjust your expectations, you could explore personal growth opportunities such as therapy, practicing stoicism, volunteering in underserved communities, or any other activity that helps you develop self-awareness and emotional intelligence."
}
"""
