# 분실물 인증 시스템 — 전체 파이프라인

자연어 입력을 받아 분실물의 실제 주인인지 판단하는 모델
텍스트(NLU) + 이미지(CV) 파이프라인을 결합한 구조

---

## 전체 아키텍처

```
[사용자 자연어 입력]
        │
        ├─→ ① 텍스트 파이프라인 (NLU)
        │
[Delhi Metro DB] ──→ ② 이미지 파이프라인 (CV)
        │
        └─→ ③ 융합(Fusion) → 최종 판단(주인 맞음/아님)
                        │
                        └─→ ④ 평가(Evaluation)
```

---

## ① 텍스트 파이프라인 (NLU 파트)

### 1-1. 전처리
- 한국어라면 KSS(문장 토큰화) + KoSpacing(띄어쓰기 보정)
- 정수 인코딩(Tokenizer) → OOV 처리

### 1-2. 슬롯 추출 (핵심 기법: NER / BIO 태깅)

사용자 입력에서 4가지 슬롯을 추출

```
입력: "5월 중순쯤 라지브 초크역에서 파란색 아이폰을 잃어버렸어요"

물건종류: 아이폰       (ITEM)
색상/외형: 파란색      (ATTR)
장소:     라지브 초크역  (LOC)
시간:     5월 중순     (DATE)
```

- **모델**: BiLSTM-CRF 또는 BiLSTM-CNN-CRF (문자 단위 CNN 추가로 OOV 대응력 강화)
- **레이블 체계**: BIO 태깅 (`B-ITEM`, `I-ITEM`, `B-LOC`, `B-DATE`, `O` 등)
- **학습 데이터**: Delhi Metro 데이터의 `item_name`, `description`, `station_name`, `receiving_date`를 활용해 슬롯 태깅용 학습 문장을 (규칙 기반 템플릿으로) 자동 생성 가능

### 1-3. 텍스트 임베딩 & 매칭

```
추출된 슬롯 텍스트 → 임베딩 → DB 레코드의 텍스트와 유사도 비교
```

- **임베딩 방식**: TF-IDF(베이스라인) → Word2Vec/GloVe → BiLSTM 기반 문장 벡터 (난이도별 단계적 적용)
- **유사도**: 코사인 유사도 (물건종류/색상 등 텍스트 속성 비교)
- **장소/날짜**: 텍스트 유사도보다 규칙 기반 매칭이 적합
  - 장소: 정확 일치 또는 인근 역(그래프 거리) 가중치
  - 날짜: 사용자가 말한 기간과 `receiving_date`의 차이(일 수)를 점수화

---

## ② 이미지 파이프라인 (CV 파트)

### 2-1. 데이터
- Open Images V7에서 Delhi Metro `item_name`과 겹치는 카테고리 필터링
- (Phone, Bag, Umbrella, Laptop 등)

### 2-2. 물건 종류 분류
- **모델**: 사전학습 CNN(ResNet, EfficientNet 등) 파인튜닝 — 전이학습 활용
- **출력**: 물건 카테고리 확률 분포

### 2-3. (선택, 심화) 색상/속성 추출
- OpenCV로 주요 색상(dominant color) 추출 → 텍스트 슬롯의 "색상"과 매칭
- 또는 CLIP처럼 이미지-텍스트를 같은 임베딩 공간에 놓는 모델 활용 시 속성까지 통합 비교 가능

---

## ③ 융합(Fusion) — 최종 판단

### 방식 A: 스코어 결합 (구현 난이도 낮음, 추천 시작점)

```
Score = w1·S_item + w2·S_attr + w3·S_loc + w4·S_date
```

| 구성 요소 | 계산 방법 |
|---|---|
| S_item | 이미지 분류기 예측 vs 텍스트 슬롯 "물건종류" 일치도 |
| S_attr | 텍스트 임베딩 코사인 유사도 (색상/외형 등) |
| S_loc | 장소 일치 점수 (정확 일치=1, 인근=0.5, 불일치=0 등 규칙) |
| S_date | 날짜 차이 기반 감쇠 함수 (가까울수록 1에 근접) |

- 가중치 $w_i$는 검증셋으로 그리드서치 또는 **로지스틱 회귀**로 학습 (스코어들을 feature로, "주인 맞음/아님" 정답 레이블로 학습)
- Threshold 넘으면 "주인 확률 높음" 판정

### 방식 B: 통합 신경망 (심화, 시간 여유 있을 때)

```
텍스트 임베딩 + 이미지 임베딩 → concat → Dense layer → 이진분류 출력
```

- BiLSTM 텍스트 인코더의 최종 hidden state + CNN 이미지 인코더의 feature vector를 concat
- concat 후 Linear layer로 차원 정리 → sigmoid로 "주인 맞음" 확률 출력
- 손실함수: 크로스엔트로피

---

## ④ 평가

| 지표 | 용도 |
|---|---|
| Precision / Recall / F1 | 이진분류(주인 맞음/아님) 성능 평가 |
| Macro avg vs Weighted avg | 물건 종류별 성능 편차 확인 |
| Confusion Matrix | 오탐(FP)/누락(FN) 정성 분석 |

**Negative pair 생성**: Delhi Metro 데이터에서 서로 다른 레코드끼리 무작위로 짝지어 "주인 아님" 학습 데이터 생성 (positive:negative 비율 조정으로 클래스 불균형 고려)

---

## 단계별 구현 로드맵 (난이도순)

| 단계 | 내용 | 난이도 |
|---|---|---|
| 1단계 | 텍스트만으로 TF-IDF + 코사인 유사도 매칭 (베이스라인) | ★ |
| 2단계 | BiLSTM-CRF로 슬롯 추출(NER) 파이프라인 구축 | ★★ |
| 3단계 | 이미지 분류기(전이학습) 추가, 스코어 결합(방식 A) | ★★★ |
| 4단계 | (심화) 텍스트+이미지 통합 신경망(방식 B), 하이퍼파라미터 튜닝 | ★★★★ |

---

## 이번 학기 배운 개념과의 연결 (보고서용)

| 파이프라인 단계 | 활용 개념 |
|---|---|
| 슬롯 추출 | BIO 태깅, BiLSTM-CRF, (문자 단위 CNN) |
| 텍스트 임베딩 | TF-IDF, Word2Vec/GloVe, 코사인/자카드 유사도 |
| 융합 | 로지스틱 회귀, 크로스엔트로피, concat |
| 평가 | Precision/Recall/F1, macro/weighted avg |

---

## 참고 데이터셋

- **텍스트**: Delhi Metro Lost and Found Dataset (item_name, description, station_name, receiving_date 등 13,000+ 건)
- **이미지**: Open Images V7 (Google) — Phone, Bag, Umbrella, Laptop 등 카테고리 필터링하여 사용
