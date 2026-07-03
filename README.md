# Lostandfound-Helper

분실물 자연어 검색 시스템

## 프로젝트 개요
분실물센터 접수 담당자가 수행하던 "설명 청취 → 카테고리 분류 → 기존 습득물과 매칭" 업무를
자연어 처리 기술로 대체/보조하는 시스템입니다.

사용자가 자연어로 분실 상황을 설명하면("어제 2호선에서 검은색 백팩을 잃어버렸어요, 안에 노트북이 있어요"),
시스템이 자동으로 핵심 정보를 추출하고 기존 습득물 데이터베이스에서 유사한 항목을 찾아 매칭해줍니다.

## 대체하는 human role
지하철/공공기관 분실물센터 접수 담당자가 수행하던:
1. 분실물 설명을 듣고 카테고리(품목/색상/특징) 정리
2. 기존 습득물 기록과 수작업으로 대조하여 매칭

## 사용 데이터셋
[Delhi Metro Lost and Found Dataset](https://www.kaggle.com/datasets/forgetabhi/delhi-metro-lost-and-found-dataset)
- 2021.12 ~ 2024.03 델리 지하철 분실/습득물 13,000+ 건
- 필드: item_name, description, item_quantity, station_name, receiving_date, receiving_time

## 적용 NLP 기법
1. **정보 추출 (Information Extraction)**: 자연어 질의에서 품목/색상/장소/날짜 등 핵심 슬롯 추출
2. **유사도 매칭 (Embedding-based Similarity / RAG)**: 질의를 임베딩하여 기존 습득물 description들과
   cosine similarity 기반 top-k 매칭, 또는 LLM API 기반 RAG로 자연어 응답 생성

## 프로젝트 구조
```
├── data/           # 데이터셋
├── notebooks/       # 탐색적 분석
├── src/            # 소스 코드
│   ├── preprocessing.py
│   ├── extraction.py
│   ├── matching.py
│   └── app.py
├── requirements.txt
└── README.md
```

## 실행 방법
```bash
pip install -r requirements.txt
python src/app.py
```

## 팀원 / 역할
- (작성 예정)

## 참고 자료
- Delhi Metro Lost and Found Dataset (Kaggle)
