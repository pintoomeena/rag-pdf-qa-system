from transformers import pipeline

generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-large",
    framework="pt"
)


def generate_answer(question, context):
    if not context or not context.strip():
        return "I couldn't find enough information in the document."

    prompt = f"""
Answer the question directly using only the provided context.

Instructions:
- Start immediately with the answer.
- Do not repeat the question.
- Do not add an introduction or unnecessary opening sentence.
- Keep the answer clear and concise.
- Use 2 to 4 sentences when appropriate.
- Do not invent facts or use information outside the context.
- If the context does not contain the answer, say:
  I couldn't find enough information in the document.

Context:
{context[:2000]}

Question:
{question}

Direct answer:
"""

    try:
        result = generator(
            prompt,
            max_new_tokens=120,
            do_sample=False,
            truncation=True
        )

        answer = result[0]["generated_text"].strip()

        if not answer:
            return "I couldn't generate an answer from the provided context."

        return answer

    except Exception as e:
        print(f"Answer generation error: {e}")
        return "Sorry, an error occurred while generating the answer."