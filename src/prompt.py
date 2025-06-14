prompt_template = ("""
You are an expert question generator with a deep understanding of Machine Learning documentation.
Your role is to help coders and programmers prepare for their exams, coding interviews, and technical assessments.

Given the text below:
------------
{text}
------------

Please generate a list of questions that test understanding of the material. Each question should:
1. Start with a number followed by a period (e.g., "1.")
2. End with a question mark
3. Be clear and concise
4. Test both basic and advanced understanding

Format your response exactly like this example:
1. What is the main concept discussed in the text?
2. How does this concept apply in real-world scenarios?
3. What are the key components of this system?

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
