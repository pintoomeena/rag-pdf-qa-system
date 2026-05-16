from transformers import pipeline

generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    framework="pt"
)


def generate_answer(
    question,
    context
):

    prompt = f"""
    Context:
    {context}

    Question:
    {question}

    Give a short answer.
    """

    result = generator(
        prompt,
        max_new_tokens=50
    )

    return result[0]["generated_text"]