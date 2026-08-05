# hangle_normalizer

띄어쓰기와 오타가 포함된 한국어 문장을 규칙 기반 형태소 처리, 로컬 Ollama LLM 또는 두 방법의 조합으로 정규화하고 CER/WER를 계산하는 비교 실험이다.

## 실행 방법

모든 명령은 `input.txt`를 읽어 방식별 출력 파일을 만든다. 형태소 방식은 KoNLPy/Okt와 Java runtime, LLM 방식은 `requests`와 로컬 Ollama의 `llama3.1:8b` 모델이 필요하다.

```bash
cd hangle_normalizer

# KoNLPy 형태소 + 단순 규칙
python3 morph_gec_normalizer.py

# Ollama LLM만 사용
python3 llm_normalizer.py

# 형태소 초안 후 Ollama 보정
python3 morph_gec_llm_normalizer.py
```

LLM 방식은 `http://localhost:11434/api/generate`에 접속하므로 먼저 Ollama service와 `llama3.1:8b` 모델을 준비해야 한다.

```bash
ollama pull llama3.1:8b
ollama serve  # 별도 terminal에서 계속 실행
```

## 입력, 출력과 평가

저장소의 `input.txt`와 `ground.txt`에는 서로 대응하는 예제 13문장이 있다. 출력은 각각 `output_GEC.txt`, `output_LLM.txt`, `output_GEC_LLM.txt`로 저장한다.

```bash
python3 evaluate.py --ground ground.txt --output output_GEC_LLM.txt
```

평가는 exact match, 평균·중앙 CER/WER와 최댓값을 표준 출력한다.

## 제한

규칙 기반 방식은 한글 형태소만 이어 붙이고 종결 문자 `다`를 추가하므로 공백·숫자·문장부호를 보존하지 않는다. Ollama 결과는 모델과 실행 시점에 따라 달라질 수 있다. `evaluate.py`는 reference와 output 길이가 다른지 검사하지 않고 `zip()`으로 짝을 만들기 때문에 줄 수를 동일하게 유지해야 한다.
