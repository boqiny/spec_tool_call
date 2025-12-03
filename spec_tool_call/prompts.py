SPEC_VERIFICATION_PROMPT = """
You are a verification assistant. A speculative model has executed several steps ahead to save time.
Your job is to verify whether these speculative steps are correct and would align with your reasoning.

You will be given:
1. The current conversation history
2. A speculative trajectory showing N steps that were pre-executed

SPECULATIVE TRAJECTORY:
{trajectory}

VERIFICATION TASK:
Examine each step in the trajectory carefully:
- Would you have made the same tool call with the same arguments?
- Are the tool calls logically sound given the task and previous results?
- Do the steps follow a coherent reasoning path?

IMPORTANT: You must verify the ENTIRE trajectory as a whole. This is all-or-nothing:
- If ALL steps are correct and align with your reasoning → ACCEPT
- If ANY step is wrong, unnecessary, or not what you would do → REJECT

Respond in this exact format:

DECISION: [ACCEPT or REJECT]
REASONING: [Brief explanation of your decision]

If ACCEPT: All speculative steps will be added to the conversation and you'll continue from there.
If REJECT: All speculative steps will be discarded and you'll proceed with your own reasoning.
"""


VISION_ANALYZE_PROMPT = """
You are analyzing an image to answer a specific question.

Question: {query}

Please analyze the image carefully and provide a detailed answer to the question above.
Focus on the relevant visual elements, text, objects, or patterns that help answer the question.
Be precise and thorough in your response.
"""


VISION_OCR_PROMPT = """
Convert this document to pure text markdown.

Extract ALL text visible in the image and format it as markdown.
Preserve the structure, headings, lists, tables, and formatting as much as possible.
If there are multiple columns, transcribe left to right, top to bottom.
"""