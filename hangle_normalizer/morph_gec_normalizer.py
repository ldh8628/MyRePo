# morph_gec_normalizer.py
# Morph + Rule-based GEC

from konlpy.tag import Okt
import re

okt = Okt()

def normalize_sentence(sentence: str) -> str:
    """
    1. 형태소 분석
    2. 의미 없는 단위 제거
    3. 단순 결합 기반 문장 복원
    """
    sentence = sentence.strip()
    morphs = okt.morphs(sentence)

    cleaned = []
    for m in morphs:
        if re.match(r"[가-힣]", m):
            cleaned.append(m)

    result = "".join(cleaned)

    # 간단한 종결 처리
    if not result.endswith("다"):
        result += "다"

    return result


def normalize_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    for line in lines:
        results.append(normalize_sentence(line))

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r + "\n")


if __name__ == "__main__":
    normalize_file("input.txt", "output_GEC.txt")
