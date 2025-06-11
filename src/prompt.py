prompt_template = ("""
You are an expert question generator with a deep understanding of Machine Learning documentation.
Your role is to help coders and programmers prepare for their exams, coding interviews, and technical assessments.

Given the text below:
------------
{text}
------------

Please generate a thoughtful and comprehensive list of questions that thoroughly test the coder’s understanding of the material. Ensure that:
- The questions cover all key concepts and technical details in the text.
- The questions challenge both foundational and advanced understanding.
- The questions are clear and concise, with no ambiguity.
- The questions vary in difficulty, including some that encourage critical thinking.

QUESTIONS:
""")






refine_template = """
You are an expert at creating and refining practice questions based on coding material and documentation.
Your goal is to help coders and programmers prepare for coding tests.

We have already prepared some initial questions:
------------
{existing_answer}
------------

Below is additional context that may help refine or improve these questions:
------------
{text}
------------

Given this new context:
- Refine the existing questions where appropriate to improve clarity, coverage, or difficulty.
- If the new context is not helpful, simply return the original questions unchanged.

Please provide the improved questions in English.

QUESTIONS:
"""
