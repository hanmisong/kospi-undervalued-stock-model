"""
=============================================================
 저평가 우량주 발굴 — 선형 회귀 분석 모델 (최종본)

 데이터 소스: FinanceDataReader (네이버 재무제표)
 분석 대상: KOSPI 보통주
 Target: log(PBR)
 Features: ROE, 배당수익률, 자사주소각, 부채비율

 설치: pip install finance-datareader opendartreader
=============================================================
"""

# ── 라이브러리 ────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import time, warnings

warnings.filterwarnings('ignore')

# ── 한글 폰트 ─────────────────────────────────────────────────────
plt.rcParams['axes.unicode_minus'] = False
for font in ['AppleGothic', 'NanumGothic', 'Malgun Gothic']:
    try:
        plt.rcParams['font.family'] = font
        break
    except Exception:
        continue

import FinanceDataReader as fdr

# ══════════════════════════════════════════════════════════════════
# 설정값 (여기만 바꾸면 됩니다)
# ══════════════════════════════════════════════════════════════════
MAX_CNT = 150  # 수집할 종목 수 (테스트: 30 / 실전: 150~300)
DART_API_KEY = "8129c503af47d9d19fa5ebdb37a22f1812139cb1"  # DART API 키 (https://opendart.fss.or.kr 무료 발급)
import datetime
today = datetime.date.today()
week_ago = today - datetime.timedelta(days=7)
PRICE_DATE_START = week_ago.strftime('%Y-%m-%d')
PRICE_DATE_END   = today.strftime('%Y-%m-%d')

# ══════════════════════════════════════════════════════════════════
# STEP 1. KOSPI 종목 리스트
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1. KOSPI 종목 리스트")
print("=" * 60)

df_list = fdr.StockListing('KOSPI')
df_list = df_list.reset_index(drop=True)
df_list = df_list.loc[:, ~df_list.columns.duplicated()]
df_list['Code'] = df_list['Code'].astype(str).str.zfill(6)
df_list = df_list[~df_list['Code'].str.endswith('5')].copy()  # 우선주 제거
df_list = df_list[['Code', 'Name']].dropna().reset_index(drop=True)

print(f"  KOSPI 보통주: {len(df_list)}개\n")

# ══════════════════════════════════════════════════════════════════
# STEP 2. 재무 데이터 수집 (네이버 재무제표)
#
#  SnapDataReader 구조: 행=날짜, 열=계정명
#  → 마지막 행(최신 연도)에서 값 추출
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 2. 재무 데이터 수집")
print("=" * 60)


def get_finstate_value(fs, keywords):
    """재무제표 DataFrame에서 키워드에 해당하는 최신 값 반환"""
    cols = fs.columns.astype(str)
    for kw in keywords:
        matched = [c for c in cols if kw in c]
        if matched:
            for row_idx in reversed(range(len(fs))):
                val_str = (str(fs.iloc[row_idx][matched[0]])
                           .replace(',', '').replace(' ', '')
                           .replace('−', '-').replace('–', '-')
                           .replace('\xa0', ''))
                try:
                    result = float(val_str)
                    if result != 0:
                        return result
                except Exception:
                    continue
    return None


records = []

for i, row in df_list.head(MAX_CNT).iterrows():
    code, name = row['Code'], row['Name']
    try:
        fs = fdr.SnapDataReader(f'NAVER/FINSTATE-2Y/{code}')
        if fs is None or fs.empty:
            continue

        net_income = get_finstate_value(fs, ['당기순이익', '순이익'])
        equity = get_finstate_value(fs, ['자본총계', '자기자본'])
        debt = get_finstate_value(fs, ['부채총계', '총부채'])
        bps = get_finstate_value(fs, ['BPS', '주당순자산'])
        dps = get_finstate_value(fs, ['DPS', '주당배당금', '현금DPS'])

        if not (net_income and equity and equity != 0):
            continue

        roe = net_income / equity * 100
        debt_ratio = (debt / equity * 100) if (debt and equity != 0) else None

        # 주가 (연말 기준)
        last_price = None
        try:
            price_df = fdr.DataReader(code, PRICE_DATE_START, PRICE_DATE_END)
            if price_df is not None and not price_df.empty:
                last_price = float(price_df['Close'].iloc[-1])
        except Exception:
            pass

        if not last_price:
            continue

        pbr = (last_price / bps) if (bps and bps > 0) else None
        div_yield = (dps / last_price * 100) if (dps and dps > 0) else 0.0

        if pbr is None:
            continue

        records.append({
            'Code': code,
            '기업명': name,
            'ROE': round(roe, 2),
            '부채비율': round(debt_ratio, 1) if debt_ratio else None,
            '배당수익률': round(div_yield, 2),
            'PBR': round(pbr, 2),
        })

        if (i + 1) % 10 == 0:
            print(f"  진행: {i + 1}/{MAX_CNT} — 수집: {len(records)}개")

        time.sleep(0.4)

    except Exception:
        continue

df = pd.DataFrame(records)
print(f"\n  재무 수집 완료: {len(df)}개 종목")

if df.empty:
    print(" 데이터 없음. MAX_CNT 확인 후 재실행하세요.")
    import sys;

    sys.exit()

print(df[['기업명', 'PBR', 'ROE', '배당수익률', '부채비율']].head(5).to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# STEP 3. 자사주 소각 공시 탐지 (DART)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 3. 자사주 소각 공시 탐지")
print("=" * 60)

buyback_set = set()

if DART_API_KEY:
    try:
        import OpenDartReader

        dart = OpenDartReader(DART_API_KEY)
        for code in df['Code'].tolist()[:50]:
            try:
                disc = dart.list(code, start='2023-01-01', end='2024-12-31')
                if disc is not None and not disc.empty:
                    if disc['report_nm'].str.contains('자기주식소각', na=False).any():
                        buyback_set.add(code)
                time.sleep(0.1)
            except Exception:
                pass
        print(f"  자사주 소각 탐지: {len(buyback_set)}개 종목")
    except ImportError:
        print("  opendartreader 미설치 → pip install opendartreader")
else:
    print("  ℹ DART 키 미입력 → 자사주소각 전체 0으로 처리")

df['자사주소각'] = df['Code'].isin(buyback_set).astype(int)

# ══════════════════════════════════════════════════════════════════
# STEP 4. 데이터 정제
# ══════════════════════════════════════════════════════════════════
df = df.dropna(subset=['PBR', 'ROE', '배당수익률', '부채비율'])
df = df[
    df['PBR'].between(0.05, 3.0) &  # 바이오/성장주 극단값 제거
    df['ROE'].between(-50, 100) &
    df['부채비율'].between(0, 500)
    ].reset_index(drop=True)

# log(PBR) 변환 — 오른쪽 치우침 보정, 정규성 개선
df['logPBR'] = np.log(df['PBR'])

print(f"\n  최종 분석 종목: {len(df)}개")
if len(df) < 10:
    print("  종목 수 부족. MAX_CNT를 늘려주세요.")
    import sys;

    sys.exit()

# ══════════════════════════════════════════════════════════════════
# STEP 5. VIF 검사 + OLS 회귀
# ══════════════════════════════════════════════════════════════════
feature_cols = ['ROE', '배당수익률', '자사주소각']
feature_cols = [c for c in feature_cols if df[c].std() > 0]  # 분산 0 제거

# ── VIF ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5a. 다중공선성 검사 (VIF)")
print("=" * 60)

X_raw = df[feature_cols].values
vif_data = pd.DataFrame({
    'Feature': feature_cols,
    'VIF': [variance_inflation_factor(X_raw, i) for i in range(len(feature_cols))]
})
print(vif_data.round(2))
print("  ※ VIF > 10: 심각한 다중공선성 → 자동 제거")

high_vif = vif_data[vif_data['VIF'] > 10]['Feature'].tolist()
if high_vif:
    print(f"  ⚠ 제거: {high_vif}")
    feature_cols = [f for f in feature_cols if f not in high_vif]

# ── OLS 회귀 ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5b. OLS 회귀 분석 (Target: log PBR)")
print("=" * 60)

scaler = StandardScaler()
X_sc = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)
y = df['logPBR'].values
X_sm = sm.add_constant(X_sc)
model = sm.OLS(y, X_sm).fit()
print(model.summary())

# ── 잔차 & 저평가 식별 ───────────────────────────────────────────
df['logPBR_예측'] = model.predict(X_sm)
df['PBR_예측'] = np.exp(df['logPBR_예측'])  # 원래 단위로 역변환
df['잔차'] = df['PBR'] - df['PBR_예측']
df['저평가여부'] = df['잔차'] < -0.2
undervalued = df[df['저평가여부']].sort_values('잔차')

print(f"\n  저평가 종목: {len(undervalued)}개 (잔차 < -0.20)")
if not undervalued.empty:
    print(undervalued[['기업명', 'PBR', 'PBR_예측', '잔차', 'ROE', '배당수익률']]
          .head(10).round(3).to_string(index=False))

# ── 정규성 검정 (log 잔차 기준) ──────────────────────────────────
log_residuals = df['logPBR'] - df['logPBR_예측']
stat, p_val = stats.shapiro(log_residuals)
print(f"\n  Shapiro-Wilk (log 잔차): W={stat:.4f}, p={p_val:.4f}")
if p_val > 0.05:
    print("  → 정규성 만족 ✓")
else:
    print("  → 정규성 위반 — 한국 증시 구조적 저평가 경향 반영")

# ══════════════════════════════════════════════════════════════════
# STEP 6. 시각화
# ══════════════════════════════════════════════════════════════════
BLUE = '#2563EB'
RED = '#DC2626'
ORANGE = '#F97316'
GRAY = '#9CA3AF'

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('저평가 우량주 발굴 — 선형 회귀 분석 (실제 KOSPI 데이터)',
             fontsize=16, fontweight='bold')
for ax in axes.flatten():
    ax.set_facecolor('#F8FAFC')
fig.patch.set_facecolor('white')

# (A) Feature Importance ──────────────────────────────────────────
ax1 = axes[0, 0]
n_feat = len(feature_cols)
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.params[1:n_feat + 1].values,
    'p_value': model.pvalues[1:n_feat + 1].values,
}).sort_values('Coefficient', ascending=True)

colors = [RED if c > 0 else BLUE for c in coef_df['Coefficient']]
bars = ax1.barh(coef_df['Feature'], coef_df['Coefficient'],
                color=colors, alpha=0.85, edgecolor='white')
for bar, pv in zip(bars, coef_df['p_value']):
    sig = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else ''))
    ax1.text(bar.get_width() + 0.003 * np.sign(bar.get_width()),
             bar.get_y() + bar.get_height() / 2, sig, va='center', fontsize=10)
ax1.axvline(0, color='#374151', lw=1.2, linestyle='--')
ax1.set_title('(A) Feature Importance (표준화 β)', fontweight='bold')
ax1.set_xlabel('표준화 회귀 계수 β')
ax1.text(0.98, 0.02, f'R²={model.rsquared:.3f}', transform=ax1.transAxes,
         ha='right', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRAY))

# (B) ROE vs PBR ─────────────────────────────────────────────────
ax2 = axes[0, 1]
mask = ~df['저평가여부']
ax2.scatter(df.loc[mask, 'ROE'], df.loc[mask, 'PBR'],
            color=BLUE, alpha=0.4, s=40, label='일반')
ax2.scatter(df.loc[df['저평가여부'], 'ROE'], df.loc[df['저평가여부'], 'PBR'],
            color=RED, s=80, marker='*', zorder=5, label='저평가')
sl, ic, *_ = stats.linregress(df['ROE'], df['PBR'])
xl = np.linspace(df['ROE'].min(), df['ROE'].max(), 200)
ax2.plot(xl, ic + sl * xl, color=ORANGE, lw=2, linestyle='--', label='회귀선')
for _, r in undervalued.head(5).iterrows():
    ax2.annotate(r['기업명'], xy=(r['ROE'], r['PBR']),
                 xytext=(r['ROE'] + 0.5, r['PBR'] + 0.05),
                 fontsize=7, color=RED, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))
ax2.set_title('(B) ROE vs PBR — 저평가 식별', fontweight='bold')
ax2.set_xlabel('ROE (%)');
ax2.set_ylabel('PBR (배)')
ax2.legend(fontsize=9)

# (C) 잔차 분포 (log 잔차 기준) ──────────────────────────────────
ax3 = axes[1, 0]
ax3.hist(log_residuals, bins=min(25, len(df) // 2 + 1),
         color=BLUE, alpha=0.7, edgecolor='white', density=True)
xn = np.linspace(log_residuals.min(), log_residuals.max(), 200)
ax3.plot(xn, stats.norm.pdf(xn, log_residuals.mean(), log_residuals.std()),
         color=RED, lw=2, label='정규분포')
ax3.axvline(0, color=GRAY, lw=1, linestyle=':')
ax3.set_title('(C) 잔차 분포 & 정규성 (log 기준)', fontweight='bold')
ax3.set_xlabel('log 잔차');
ax3.set_ylabel('밀도')
ax3.legend(fontsize=9)
ax3.text(0.02, 0.95, f'Shapiro p={p_val:.3f}', transform=ax3.transAxes,
         fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRAY))

# (D) 실제 PBR vs 예측 PBR ───────────────────────────────────────
ax4 = axes[1, 1]
ax4.scatter(df['PBR_예측'], df['PBR'], color=BLUE, alpha=0.4, s=40)
ax4.scatter(df.loc[df['저평가여부'], 'PBR_예측'],
            df.loc[df['저평가여부'], 'PBR'],
            color=RED, s=80, marker='*', zorder=5, label='저평가')
lim = [min(df['PBR_예측'].min(), df['PBR'].min()) - 0.1,
       max(df['PBR_예측'].max(), df['PBR'].max()) + 0.1]
ax4.plot(lim, lim, color=ORANGE, lw=2, linestyle='--', label='Perfect Fit')
ax4.fill_between(lim, [l - 0.2 for l in lim], lim,
                 color=RED, alpha=0.05, label='저평가 영역')
ax4.set_xlim(lim);
ax4.set_ylim(lim)
ax4.set_title('(D) 실제 PBR vs 예측 PBR', fontweight='bold')
ax4.set_xlabel('예측 PBR');
ax4.set_ylabel('실제 PBR')
ax4.legend(fontsize=9)
ax4.text(0.02, 0.95, f'R²={model.rsquared:.3f}', transform=ax4.transAxes,
         fontsize=10, fontweight='bold', va='top',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRAY))

plt.tight_layout()
plt.savefig('regression_result_realdata.png', dpi=150, bbox_inches='tight')
plt.show()
