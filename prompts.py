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
