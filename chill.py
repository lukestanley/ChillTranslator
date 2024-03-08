# chill.py
from argparse import ArgumentParser
import json
from time import time
from uuid import uuid4
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



class ImprovementContext:
    def __init__(self, original_text=None):
        self.suggestions = []
        self.last_edit = ""
        self.request_count = 0
        self.start_time = time()
        self.original_text = original_text
        self.improvement_result  = dict()

def query_ai_prompt_with_count(prompt, replacements, model_class, context):
    context.request_count += 1
    return query_ai_prompt(prompt, replacements, model_class)




def improve_text_attempt(context):
    replacements = {
        "original_text": json.dumps(context.original_text),
        "previous_suggestions": json.dumps(context.suggestions, indent=2),
    }
    return query_ai_prompt_with_count(improve_prompt, replacements, ImprovedText, context)


def critique_text(context):
    replacements = {"original_text": context.original_text, "last_edit": context.last_edit}

    # Query the AI for each of the new prompts separately

    critique_resp = query_ai_prompt_with_count(critique_prompt, replacements, Critique, context)
    faithfulness_resp = query_ai_prompt_with_count(
        faith_scorer_prompt, replacements, FaithfulnessScore, context
    )
    spiciness_resp = query_ai_prompt_with_count(
        spicy_scorer_prompt, replacements, SpicyScore, context
    )

    # Combine the results from the three queries into a single dictionary
    combined_resp = {
        "critique": critique_resp["critique"],
        "faithfulness_score": faithfulness_resp["faithfulness_score"],
        "spicy_score": spiciness_resp["spicy_score"],
    }

    return combined_resp


def update_suggestions(critique_dict, iteration, context):
    """
    Gets weighted score for new suggestion,
    adds new suggestion,
    sorts suggestions by score,
    updates request_count, time_used,
    log progress and return highest score
    """
    context.iteration = iteration
    time_used = time() - context.start_time
    critique_dict["overall_score"] = round(
        calculate_overall_score(
            critique_dict["faithfulness_score"], critique_dict["spicy_score"]
        ),
        2,
    )
    critique_dict["edit"] = context.last_edit
    if "worst_fix" in context.improvement_result:
        critique_dict["worst_fix"] = context.improvement_result["worst_fix"]
    if "nvc" in context.improvement_result:
        critique_dict["perspective"] = context.improvement_result["nvc"]
    if "constructive" in context.improvement_result:
        critique_dict["constructive"] = context.improvement_result["constructive"]
    context.suggestions.append(critique_dict)
    context.suggestions = sorted(context.suggestions, key=lambda x: x["overall_score"], reverse=True)[
        :2
    ]
    critique_dict["request_count"] = context.request_count
    if context.verbose:
        print_iteration_result(context.iteration, critique_dict["overall_score"], time_used, context.suggestions)
    return critique_dict["overall_score"]


def print_iteration_result(iteration, overall_score, time_used, suggestions):
    print(
        f"Iteration {iteration}: overall_score={overall_score:.2f}, time_used={time_used:.2f} seconds."
    )
    print("suggestions:")
    print(json.dumps(suggestions, indent=2))


def done_log(context):
    log_entry = {
        "uuid": str(uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "input": context.original_text,
        "output": context.suggestions[0],
    }
    log_to_jsonl("inputs_and_outputs.jsonl", log_entry)


def improvement_loop(
    input_text,
    max_iterations=3,
    good_score=0.85,
    min_iterations=2,
    good_score_if_late=0.7,
    deadline_seconds=60,
    verbose=True,
):
    context = ImprovementContext()
    context.original_text = input_text
    context.verbose = verbose
    time_used = 0

    for iteration in range(1, max_iterations + 1):
        context.improvement_result = improve_text_attempt(context)
        context.last_edit = context.improvement_result["hybrid"]
        critique_dict = critique_text(context)
        overall_score = update_suggestions(critique_dict, iteration, context)
        good_attempt = iteration >= min_iterations and overall_score >= good_score
        time_used = time() - context.start_time
        too_long = time_used > deadline_seconds and overall_score >= good_score_if_late
        if good_attempt or too_long:
            break

    assert len(context.suggestions) > 0
    if verbose: print("Stopping\nTop suggestion:\n", json.dumps(context.suggestions[0], indent=4))
    context.suggestions[0].update({
        "input": context.original_text,
        "iteration_count": iteration, 
        "max_allowed_iterations": max_iterations, 
        "time_used": time_used,
        "worst_terms": context.improvement_result.get("worst_terms", ""),
        "worst_fix": context.improvement_result.get("worst_fix", ""),
        "perspective": context.improvement_result.get("nvc", ""),
        "constructive": context.improvement_result.get("constructive", "")
    })
    done_log(context)
    return context.suggestions[0]


if __name__ == "__main__":
    parser = ArgumentParser(description="Process and improve text.")
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
