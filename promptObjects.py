from pydantic import BaseModel, Field

improve_prompt = """
Given some inflammatory text, make minimal changes to the text to make it less inflammatory, while keeping the original meaning as much as possible.
Make the new version more calm and constructive, without changing the intended meaning, with only minimal changes to the existing text.
Make sure the refined text is a good reflection of the original text, without adding new ideas.
Make the changes as minimal as possible. Some optional strategies to make the text less inflammatory include:
-Soften harsh tone, replace or omit judgemental or extreme words.
-Rather than accusations, share perspective.
-Consider focusing on specific actions rather than character.
-Rephrasing exaggerated expressions like "always", "never" or "everyone" to be more moderate.
-Using gentler alternatives to express similar points where needed.

Avoid adding new ideas, ONLY build upon what's already there, for example, you might reframe an existing point to be more balanced but never introduce unrelated concepts.
Make both parties more happy where possible:
The reader should be INFORMED and not *offended*, and the original author should be *content* that their points where *honoured* by your edit, by minimally refining their text without loosing the original intent.

Example:
Example input text: "You're always annoying me. You never listen to me."
Example improved text output: {"text":"You're often frustrating me. It feels like you often don't listen to me."}

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
Please critique the text.
You must output the JSON in the required format only, with no remarks or prefacing remarks - JUST JSON!"""


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
You must output the JSON in the required format only, with no remarks or prefacing remarks - JUST JSON!
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
You must output the JSON in the required format only, with no remarks or prefacing remarks - JUST JSON!
"""


class ImprovedText(BaseModel):
    text: str = Field(str, description="The improved text.")


class SpicyScore(BaseModel):
    spicy_score: float = Field(float, description="The spiciness score of the text.")


class Critique(BaseModel):
    critique: str = Field(str, description="The critique of the text.")


class FaithfulnessScore(BaseModel):
    faithfulness_score: float = Field(
        float, description="The faithfulness score of the text."
    )
