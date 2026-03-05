import hashlib
import json

def make_qid(subject: str, question: str, choices: list) -> str:
    payload = {"subject": subject, "question": question, "choices": choices}
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()