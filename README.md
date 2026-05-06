# 📊 KOSPI 저평가 우량주 발굴 모델

> 선형 회귀 기반 KOSPI 저평가 종목 식별 시스템  
> FinanceDataReader × statsmodels × Tableau

---

## 🔍 개요

KOSPI 보통주의 재무 데이터를 수집하고, **log(PBR)를 타겟**으로 한 선형 회귀 모델을 통해
시장에서 구조적으로 저평가된 종목을 발굴합니다.

단순 PBR 순위 기반 선정이 아닌, **ROE·배당수익률 대비 PBR이 얼마나 낮은지**를 잔차로 측정해
"우량한데 저평가된" 종목을 정량적으로 식별합니다.

---

## 📁 파일 구성

```
├── kospi_regression.py        # 전체 KOSPI 저평가 우량주 발굴 모델
├── single_stock_diagnosis.py  # 단일 종목 심층 진단 모델
└── README.md
```

### 모델 1 — 전체 KOSPI 분석 (`kospi_regression.py`)
- KOSPI 보통주 전체 대상 재무 데이터 수집
- OLS 회귀로 저평가 종목 25개 식별
- 시장 전체의 구조적 저평가 패턴 분석

### 모델 2 — 단일 종목 심층 진단 (`single_stock_diagnosis.py`)
- 원하는 종목 코드 1개 입력 → 자동 진단 리포트 생성
- KOSPI 비교군 대비 백분위 순위 시각화
- 동아리 밸류에이션 분석과 병행 활용

---

## 🧠 방법론

### Target & Features

| 구분 | 변수 | 설명 |
|------|------|------|
| **Target** | `log(PBR)` | 오른쪽 치우침 보정, 정규성 개선 |
| Feature 1 | `ROE` | β = +0.281, p < 0.001 |
| Feature 2 | `배당수익률` | β = −0.342, p < 0.001 |
| Feature 3 | `자사주소각` | DART 공시 기반 더미 변수 |

### 분석 파이프라인

```
데이터 수집          정제 & 검증            모델링              식별
FinanceDataReader → 결측치 제거 → VIF 검사 → OLS 회귀 → 잔차 < −0.20
DART API           극단값 필터   StandardScaler  log(PBR)   저평가 종목
```

### 주요 결과

- **R² = 0.315** — ROE와 배당수익률이 PBR의 핵심 드라이버
- **저평가 종목 25개 식별** (SK스퀘어, SK하이닉스, 삼성전자 등)
- **인사이트**: 한국 고배당주의 구조적 저평가 현상 발견 (배당 β = −0.342)

---

## 📈 시각화

> 분석 결과는 **Tableau 대시보드**로 인터랙티브하게 시각화

- (A) Feature Importance — 표준화 회귀 계수 β
- (B) ROE vs PBR 산점도 — 저평가 종목 위치
- (C) 잔차 분포 & 정규성 검정
- (D) 실제 PBR vs 예측 PBR

---

## ⚙️ 실행 방법

```bash
pip install finance-datareader opendartreader statsmodels scikit-learn
```

```python
# kospi_regression.py 상단 설정값만 변경
MAX_CNT = 150          # 수집 종목 수
DART_API_KEY = "..."   # https://opendart.fss.or.kr 무료 발급

# 단일 종목 진단
TARGET_CODE = '005930'  # 삼성전자
```

---

## 🔧 한계 및 개선 방향

| 한계 | 개선 방향 |
|------|---------|
| 유효 샘플 106개 (결측치·필터링 후) | 다년도 패널 데이터로 확장 |
| 선형 관계만 포착 | **XGBoost 추가 예정** — 비선형 관계 검증 |
| Shapiro-Wilk 정규성 위반 | 한국 증시 구조적 특성 반영, 로버스트 회귀 검토 |

---

## 🛠️ 사용 기술

![Python](https://img.shields.io/badge/Python-3.10-blue)
![statsmodels](https://img.shields.io/badge/statsmodels-OLS-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-StandardScaler-orange)
![Tableau](https://img.shields.io/badge/Tableau-대시보드-blue)
![FinanceDataReader](https://img.shields.io/badge/FinanceDataReader-KOSPI-green)

---

## 👩‍💻 만든 사람

서울여자대학교 데이터사이언스학과  
금융투자(밸류에이션) 동아리 활동 병행
