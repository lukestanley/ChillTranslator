# chill.py
import argparse
import json
import time
import uuid
from data import log_to_jsonl
from datetime import datetime
from utils import calculate_overall_score, query_ai_prompt
from promptObjects import (
    improve_prompt,
    critique_prompt,
    faith_scorer_prompt,
    spicy_scorer_prompt,
    ImprovedText,
    Critique,
    FaithfulnessScore,
    SpicyScore,
)

# This script uses the large language model to improve a text, it depends on a llama_cpp server being setup with a model loaded.
# There are several different interfaces it can use, see utils.py for more details.
# Here is a bit of a local setup example:
# pip install llama-cpp-python[server] --upgrade
# python3 -m llama_cpp.server --model mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf  --port 5834 --n_ctx 4096 --use_mlock false
# Run this script:
# python3 chill.py
# This should then try and improve the original text below.
# Or you could import the improvement_loop function with a string as an argument to improve a specific text,
# or use it as a command line tool with the -t flag to improve a specific text.

original_text = """Stop chasing dreams instead. Life is not a Hollywood movie. Not everyone is going to get a famous billionaire. Adjust your expectations to reality, and stop thinking so highly of yourself, stop judging others. Assume the responsibility for the things that happen in your life. It is kind of annoying to read your text, it is always some external thing that "happened" to you, and it is always other people who are not up to your standards. At some moment you even declare with despair. And guess what? This is true and false at the same time, in a fundamental level most people are not remarkable, and you probably aren't too. But at the same time, nobody is the same, you have worth just by being, and other people have too. The impression I get is that you must be someone incredibly annoying to work with, and that your performance is not even nearly close to what you think it is, and that you really need to come down to earth. Stop looking outside, work on yourself instead. You'll never be satisfied just by changing jobs. Do therapy if you wish, become acquainted with stoicism, be a volunteer in some poor country, whatever, but do something to regain control of your life, to get some perspective, and to adjust your expectations to reality."""
# From elzbardico on https://news.ycombinator.com/item?id=36119858

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

global suggestions
suggestions = []
last_edit = ""
request_count = 0
start_time = None


def improve_text_attempt():
    global suggestions
    global request_count
    replacements = {
        "original_text": json.dumps(original_text),
        "previous_suggestions": json.dumps(suggestions, indent=2),
    }
    request_count += 1
    resp_json = query_ai_prompt(improve_prompt, replacements, ImprovedText)
    return resp_json["text"]


def critique_text(last_edit):
    global suggestions
    global request_count
    replacements = {"original_text": original_text, "last_edit": last_edit}

    # Query the AI for each of the new prompts separately

    request_count += 3
    critique_resp = query_ai_prompt(critique_prompt, replacements, Critique)
    faithfulness_resp = query_ai_prompt(
        faith_scorer_prompt, replacements, FaithfulnessScore
    )
    spiciness_resp = query_ai_prompt(spicy_scorer_prompt, replacements, SpicyScore)

    # Combine the results from the three queries into a single dictionary
    combined_resp = {
        "critique": critique_resp["critique"],
        "faithfulness_score": faithfulness_resp["faithfulness_score"],
        "spicy_score": spiciness_resp["spicy_score"],
    }

    return combined_resp


def update_suggestions(critique_dict, iteration):
    """
    Gets weighted score for new suggestion, 
    adds new suggestion,
    sorts suggestions by score,
    updates request_count, time_used,
    log progress and return highest score
    """
    global suggestions
    global time_used
    global iteration_count
    iteration_count = iteration
    time_used = time.time() - start_time
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
    critique_dict["request_count"] = request_count
    print_iteration_result(iteration_count, critique_dict["overall_score"], time_used)
    return critique_dict["overall_score"]


def print_iteration_result(iteration, overall_score, time_used):
    global suggestions
    print(
        f"Iteration {iteration}: overall_score={overall_score:.2f}, time_used={time_used:.2f} seconds."
    )
    print("suggestions:")
    print(json.dumps(suggestions, indent=2))


def improvement_loop(input_text, max_iterations=3, good_score=0.85, min_iterations=2, good_score_if_late=0.7, deadline_seconds=60):
    global original_text
    global last_edit
    global suggestions
    global request_count
    global start_time
    suggestions = []
    last_edit = ""
    request_count = 0
    start_time = time.time()
    original_text = input_text

    for iteration in range(1, max_iterations + 1):
        last_edit = improve_text_attempt()
        critique_dict = critique_text(last_edit)
        overall_score = update_suggestions(critique_dict, iteration)
        good_attempt = iteration >= min_iterations and overall_score >= good_score
        too_long = time_used > deadline_seconds and overall_score >= good_score_if_late
        if good_attempt or too_long:
            break
    
    assert len(suggestions) > 0
    print("Stopping\nTop suggestion:\n", json.dumps(suggestions[0], indent=4))
    suggestions[0].update({"iteration_count": iteration, "max_allowed_iterations": max_iterations, "time_used": time_used})
    log_entry = {
        "uuid": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "input": original_text,
        "output": suggestions[0],
    }
    log_to_jsonl("inputs_and_outputs.jsonl", log_entry)
    return suggestions[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and improve text.")
    parser.add_argument(
        "-t", "--text", type=str, help="Text to be improved", default=original_text
    )
    args = parser.parse_args()

    improvement_loop(args.text)
# TODO: Segment the text into sentences for parallel processing, and isolate the most problematic parts for improvement
"""
# import pysbd
# sentences = pysbd.Segmenter(language="en", clean=False).segment(paragraph)
"""
