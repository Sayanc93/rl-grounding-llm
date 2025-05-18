import re
from typing import List, Tuple, Optional

def extract_snippet(document: str, evidence_pointer: str) -> str:
    """
    Extracts a snippet from the document based on the evidence pointer.
    Evidence pointer can be a sentence number, paragraph number, or a verbatim quote.
    Returns the most relevant snippet or an empty string if not found.
    """
    # Try to match sentence or paragraph number
    sent_match = re.search(r"Sent(?:ence)?\s*(\d+)", evidence_pointer, re.IGNORECASE)
    para_match = re.search(r"Para(?:graph)?\s*(\d+)", evidence_pointer, re.IGNORECASE)
    quote_match = re.search(r'"(.+?)"', evidence_pointer)
    
    # Split document into paragraphs and sentences
    paragraphs = [p.strip() for p in document.split("\n") if p.strip()]
    sentences = [s.strip() for p in paragraphs for s in re.split(r'(?<=[.!?]) +', p)]
    
    if sent_match:
        idx = int(sent_match.group(1)) - 1
        if 0 <= idx < len(sentences):
            return sentences[idx]
    if para_match:
        idx = int(para_match.group(1)) - 1
        if 0 <= idx < len(paragraphs):
            return paragraphs[idx]
    if quote_match:
        quote = quote_match.group(1)
        for sent in sentences:
            if quote in sent:
                return sent
        for para in paragraphs:
            if quote in para:
                return para
    # Fallback: return evidence_pointer if it's a substring in the document
    if evidence_pointer in document:
        return evidence_pointer
    return ""

def advanced_verify_claim(claim: str, snippet: str, verifier_tokenizer, verifier_model, threshold: float = 0.5) -> float:
    """
    Use the verifier critic to check if the claim is supported by the snippet.
    Returns a reward: +1 for 'Yes', -1 for 'No', -0.5 for 'Uncertain'.
    If the verifier model outputs a confidence score, use it for finer reward granularity.
    """
    prompt = (
        f"Document Snippet: {snippet}\n"
        f"Claim: {claim}\n"
        "Is the claim directly and verifiably supported by ONLY the provided document snippet? "
        "Respond with 'Yes', 'No', or 'Uncertain'. Optionally, provide a confidence score between 0 and 1."
    )
    inputs = verifier_tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(verifier_model.device) for k, v in inputs.items()}
    outputs = verifier_model.generate(**inputs, max_new_tokens=20)
    answer = verifier_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    # Try to extract confidence score
    conf_match = re.search(r"([01](?:\.\d+)?)", answer)
    if "yes" in answer:
        if conf_match:
            conf = float(conf_match.group(1))
            return min(1.0, max(0.0, conf))
        return 1.0
    elif "no" in answer:
        if conf_match:
            conf = float(conf_match.group(1))
            return -min(1.0, max(0.0, conf))
        return -1.0
    else:
        if conf_match:
            conf = float(conf_match.group(1))
            return -0.5 * min(1.0, max(0.0, conf))
        return -0.5

def parse_claims_and_evidence(output: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Parses the output for (<claim>, <evidence_pointer>, <confidence>) triplets.
    Returns a list of tuples.
    """
    claims = re.findall(r"<claim>(.*?)</claim>", output, re.DOTALL)
    evidences = re.findall(r"<evidence_pointer>(.*?)</evidence_pointer>", output, re.DOTALL)
    confidences = re.findall(r"<confidence>(.*?)</confidence>", output, re.DOTALL)
    triplets = []
    for i in range(min(len(claims), len(evidences))):
        conf = confidences[i] if i < len(confidences) else None
        triplets.append((claims[i].strip(), evidences[i].strip(), conf.strip() if conf else None))
    return triplets
