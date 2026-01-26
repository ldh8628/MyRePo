# morph_gec_llm_normalizer.py
# Morph + GEC + LLM Hybrid

from konlpy.tag import Okt
import requests
import re

okt = Okt()

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """너는 한국어 문장 최종 교정기이다.
초안 문장을 자연스럽고 올바른 한국어 문장으로 수정하라.
불필요한 설명 없이 결과 문장만 출력하라.
"""

def morph_gec(sentence: str) -> str:
    morphs = okt.morphs(sentence)
    cleaned = [m for m in morphs if re.match(r"[가-힣]", m)]
    result = "".join(cleaned)
    if not result.endswith("다"):
        result += "다"
    return result


def llm_refine(sentence: str) -> str:
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

    results = []
    for line in lines:
        gec_out = morph_gec(line.strip())
        final_out = llm_refine(gec_out)
        results.append(final_out)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r + "\n")


if __name__ == "__main__":
    normalize_file("input.txt", "output_GEC_LLM.txt")
