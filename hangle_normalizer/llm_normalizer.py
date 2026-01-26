# llm_normalizer.py
# LLM-only normalizer (Ollama)

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """너는 한국어 문장 교정기이다.
띄어쓰기, 오타, 깨진 문장을 자연스러운 한국어 문장으로 복원하라.
의미를 바꾸지 말고 한 문장만 출력하라.
"""

def normalize_sentence(sentence: str) -> str:
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": SYSTEM_PROMPT + "\n입력: " + sentence + "\n출력:",
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()


def normalize_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    outputs = []
    for line in lines:
        outputs.append(normalize_sentence(line.strip()))

    with open(output_path, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(o + "\n")


if __name__ == "__main__":
    normalize_file("input.txt", "output_LLM.txt")
