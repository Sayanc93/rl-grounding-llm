import re
from typing import List, Tuple, Optional

def advanced_verify_claim(user_prompt: str, evidence_pointer: str, verifier_tokenizer, verifier_model):
    """
    Use the verifier critic to check if the response is supported by the snippet.
    Returns a reward: +1 for 'Yes', -1 for 'No', -0.5 for 'Uncertain'.
    If the verifier model outputs a confidence score, use it for finer reward granularity.
    """
    prompt = (
        f"User Prompt: {user_prompt}\n"
        f"Response Evidence Pointer: {evidence_pointer}\n"
        "Is the response evidence directly and verifiably supported by the provided user prompt?"
        "Respond with 'Yes', 'No', or 'Uncertain'."
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = verifier_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = verifier_tokenizer([text], return_tensors="pt").to(verifier_model.device)
    generated_ids = verifier_model.generate(
        **model_inputs,
        max_new_tokens=10
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    answer = verifier_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].lower().strip()

    if "yes" in answer:
        return (text, answer, 2.0)
    else:
        return (text, answer, 0.0)

def parse_response_and_evidence(output: str) -> List[Tuple[str, str]]:
    """
    Parses the output for (<response>, <evidence_pointer>) tuple.
    Returns a list of tuples.
    """
    responses = re.findall(r"<response>(.*?)</response>", output, re.DOTALL)
    evidences = re.findall(r"<evidence_pointer>(.*?)</evidence_pointer>", output, re.DOTALL)
    response_and_evidence = []
    for i in range(min(len(responses), len(evidences))):
        response_and_evidence.append((responses[i].strip(), evidences[i].strip()))
    return response_and_evidence
