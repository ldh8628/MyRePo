# CadPrompt

CAD 모델 하나를 주면 YOLO-World와 YOLOE에 넣을 프롬프트를 추천하는 프로그램이다. 모델을 여러 시점에서 렌더링하고, 후보 문구를 만든 뒤, 각 후보를 검출기에 직접 넣어 렌더 이미지에서 물체를 얼마나 찾는지 재서 순위를 매긴다. 실제 물체 사진은 필요 없다. 브라우저 화면(로컬 웹 서버)과 명령줄 두 가지로 실행할 수 있다.

## 동작 방식

1. **렌더링:** 구 위에 고르게 퍼진 시점(기본 24개)에서 CAD를 렌더링한다. Open3D 광선 추적을 CPU로 돌리므로 OpenGL이나 화면이 없어도 된다. 텍스처(.obj+.mtl, BOP .ply)나 정점 색이 있으면 그 색을 쓴다.
2. **후보 문구:** CAD 자체에서 얻을 수 있는 것만 쓴다.
   - 파일 이름의 단어 (`BBQSauce` → `bbq sauce`, `obj_000005`처럼 숫자뿐이면 쓰지 않음)
   - 렌더에서 면적이 큰 색 (`yellow`, `dark red` …)
   - 경계 상자 비율로 정한 대략적인 모양 (`box`, `cylinder` …)
   - CLIP(ViT-B/32)이 렌더와 가장 가깝다고 본 명사. 명사 목록은 LVIS와 Objects365 클래스 이름(약 1,360개)이며, 특정 물체를 염두에 두고 고른 목록이 아니다.
   - 위의 조합 (`색 + 명사`, `색 + 모양`)과 `object`, 사용자가 직접 넣은 문구
3. **시험 이미지:** 렌더를 배경 위에 크기·위치·밝기를 바꿔 붙인다(기본 48장, 상자 위치를 앎). 물체 없는 배경만의 이미지(24장)도 따로 만든다. 배경은 사용자가 준 사진 폴더를 쓰고, 없으면 무늬를 생성해 쓴다.
4. **채점:** 후보 하나씩 단독 클래스로 검출기에 넣는다.
   - 찾음: 신뢰도 ≥ 임계값(기본 0.1)이고 IoU ≥ 0.5인 상자가 있는 시험 이미지 비율
   - 오검출: 배경만의 이미지에서 임계값 넘는 상자가 하나라도 나온 비율
   - 점수 = 찾음 − 0.5 × 오검출
5. **YOLOE 시각 프롬프트:** 렌더 9장을 모은 모자이크를 참조 이미지로 쓴다. 모자이크만 쓴 경우와, 가장 좋은 텍스트 문구를 더한 경우를 함께 채점한다.

검출기별로 텍스트 점수가 가장 높은 문구를 추천한다. 점수가 0을 넘는 후보가 하나도 없으면 "추천 없음"으로 표시한다.

## 입력

- **CAD 형식:** `.obj`(같은 폴더에 `.mtl`과 텍스처 이미지), `.ply`(정점 색 또는 BOP처럼 텍스처를 쓴 것), `.glb`/`.gltf`, `.stl`, `.off`
- **단위:** 상관없다. 크기에 맞춰 카메라 거리를 정한다.
- **색:** 색 정보가 없으면 회색으로 렌더하고 화면에 경고를 띄운다. 이때 색 단어 후보는 믿기 어렵다.
- **배경 사진 (선택):** 검출기를 실제로 쓸 장소의 사진을 폴더로 주면, 시험 이미지와 오검출 측정이 그 장소에 가까워진다. 사진에 대상 물체가 찍혀 있으면 안 된다.

## 설치

Ubuntu 22.04, Python 3.11, NVIDIA GPU에서 확인했다. GPU가 없어도 돌지만 많이 느리다.

```bash
cd CadPrompt
conda env create -f environment.yml      # 또는: python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
conda activate cadprompt
```

GPU에 맞는 PyTorch가 따로 필요하면 https://pytorch.org 의 안내대로 먼저 설치한 뒤 `pip install -r requirements.txt`를 실행한다.

처음 실행할 때 검출기 가중치(`yolov8s-worldv2.pt`, `yoloe-11s-seg.pt`)와 텍스트 인코더를 인터넷에서 내려받는다. 가중치는 `CadPrompt/weights/`에, CLIP은 `~/.cache/clip/`에 저장된다.

## 실행

### 브라우저 화면

```bash
cd CadPrompt
python -m cadprompt.server      # http://localhost:8797
```

1. CAD 파일을 끌어다 놓는다. `.obj`이면 `.mtl`과 텍스처 이미지도 함께 놓는다. 서버 PC에 있는 파일이면 경로를 직접 입력해도 된다.
2. 필요하면 물체 이름, 검출기, 시점 수, 배경 사진 폴더, 추가 문구를 정한다.
3. "추천 시작"을 누른다.

결과 화면에는 다음이 나온다.
- 검출기별 추천 문구와 점수
- 모든 후보의 순위표
- 후보를 만든 근거(이름·색·모양·CLIP 명사)
- 렌더 이미지, 시각 프롬프트 모자이크, 시험 이미지 예시

`prompts.json`과 `visual_prompt_mosaic.png`는 `results/`에 저장되고 화면에서 내려받을 수도 있다. 다른 PC에서 접속하려면 `--host 0.0.0.0`을 준다.

### 명령줄

```bash
python -m cadprompt models/BBQSauce.obj
python -m cadprompt models/obj_000005.ply --name "mustard bottle" --backgrounds photos/lab --extra "yellow bottle"
python -m cadprompt models/cup.glb --detectors yoloe --views 36 --tests 96 --out results/cup
```

`python -m cadprompt -h`로 모든 옵션을 볼 수 있다.

### 추천 결과 쓰는 법

```python
from ultralytics import YOLOWorld, YOLOE
m = YOLOWorld("yolov8s-worldv2.pt"); m.set_classes(["yellow carton"])            # YOLO-World 텍스트
m = YOLOE("yoloe-11s-seg.pt"); m.set_classes(["dark red cigar"], m.get_text_pe(["dark red cigar"]))   # YOLOE 텍스트
```

YOLOE 시각 프롬프트를 쓰려면 `visual_prompt_mosaic.png`를 참조 이미지로 주고, `prompts.json`의 `visual_prompt.boxes`를 상자로 준다(클래스는 모두 0).

## 결과 예시

공개 CAD 세 개로 확인했다. 기본 설정(시점 24, 시험 48장, 배경 생성)을 썼고, RTX 5090 노트북 GPU에서 셋을 동시에 돌려 각각 7분 정도 걸렸다. 하나만 돌리면 2~3분이다.

| CAD | 이름 단어 | YOLO-World 추천 (점수) | YOLOE 텍스트 추천 (점수) | YOLOE 시각 프롬프트 (점수) |
|---|---|---|---|---|
| YCB-V `obj_000005` (머스터드 병) | 없음 | `yellow carton` (0.65) | `yellow freshener` (1.00) | 모자이크+텍스트 0.79 |
| HOPE `BBQSauce` | `bbq sauce` | 추천할 만한 것 없음 (최고 0.04) | `dark red cigar` (0.69) | 모자이크+텍스트 0.92 |
| YCB-V `obj_000009` (캔) | 없음 | `canned` (0.73) | — | 모자이크 0.79 |

- 물체 이름(`bbq sauce`)은 두 검출기 모두에서 0점이었다. 이름이 사람에게 자연스러워도 검출기는 다른 단어에 반응할 수 있다. 이 프로그램이 후보를 직접 재 보는 이유다.
- 추천 문구가 사람 눈에는 이상할 수 있다(`freshener`, `cigar`). 검출기의 반응만 보고 골랐기 때문이다.
- 시각 프롬프트는 찾는 비율이 높았지만 생성 배경에서 오검출도 17~46 %로 높았다.

## 한계

- **점수는 합성 이미지에서 잰 값이다.** 렌더를 배경에 붙인 이미지이므로 실제 카메라 영상의 성능과 다르다. 특히 반사·가림·조명·흐림은 반영되지 않는다. 순위를 정하는 데 쓰고, 최종 문구는 실제 영상에서 확인해야 한다.
- **비슷한 다른 물체와의 혼동은 재지 않는다.** 오검출은 대상 물체가 없는 배경에서만 잰다. 실제 장면에 비슷한 물체가 있으면 실제 오검출은 더 많을 수 있다.
- **생성 배경은 실제 장소와 다르다.** 시각 프롬프트의 오검출률은 배경에 크게 좌우된다. 쓸 장소의 사진을 `--backgrounds`로 주는 것이 좋다.
- **후보는 정해진 규칙으로 만든다.** 모양 판단이 거칠어서(병을 `box`로 부르기도 함) 좋은 문구가 후보에 없을 수 있다. `--extra`로 직접 넣을 수 있다.
- **텍스처 품질에 좌우된다.** 색 없는 CAD나 저해상도 텍스처에서는 렌더가 실제 물체와 달라 추천이 약해진다.
- 후보마다 검출기를 새로 불러오므로 후보 수에 비례해 시간이 든다.

## 결과 파일

`prompts.json`의 주요 항목은 다음과 같다.

| 항목 | 내용 |
|---|---|
| `recommended` | 검출기별 추천 문구 (`null`은 추천 없음) |
| `ranking` | 후보별 `prompt`, `source`, `detector`, `mode`(text / visual / visual+text), `hit_rate`, `false_alarm_rate`, `mean_conf`, `score` |
| `evidence` | 이름 단어, 주요 색과 면적 비율, 모양, CLIP 명사와 확률 |
| `visual_prompt.boxes` | `visual_prompt_mosaic.png` 안의 상자 9개 |
| `settings` | 시점 수, 시험 이미지 수, 임계값, 배경, 가중치 |

## 파일 구성

```
cadprompt/
  render.py      CAD 읽기, CPU 광선 추적 렌더링
  candidates.py  후보 문구 (이름·색·모양·CLIP 명사)
  compose.py     시험 이미지와 배경만의 이미지
  detect.py      YOLO-World / YOLOE 채점, 시각 프롬프트 모자이크
  pipeline.py    전체 흐름, 결과 저장
  server.py      로컬 웹 서버
  web/index.html 화면
  __main__.py    명령줄
```
