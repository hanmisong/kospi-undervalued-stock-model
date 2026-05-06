"""
=============================================================
 저평가 우량주 발굴 — 선형 회귀 + XGBoost 비교 모델

 데이터 소스: FinanceDataReader (네이버 재무제표)
 분석 대상: KOSPI 보통주
 Target: log(PBR)
 Features: ROE, 배당수익률, 자사주소각

 설치: pip install finance-datareader opendartreader xgboost scikit-learn
=============================================================
"""

# ── 라이브러리 ────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from xgboost import XGBRegressor
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

# ── 색상 변수 ─────────────────────────────────────────────────────
BLUE   = '#2563EB'
RED    = '#DC2626'
ORANGE = '#F97316'
GRAY   = '#9CA3AF'
GREEN  = '#10B981'

import FinanceDataReader as fdr

# ══════════════════════════════════════════════════════════════════
# 설정값 (여기만 바꾸면 됩니다)
# ══════════════════════════════════════════════════════════════════
MAX_CNT      = 300
DART_API_KEY = "8129c503af47d9d19fa5ebdb37a22f1812139cb1"

import datetime
today            = datetime.date.today()
week_ago         = today - datetime.timedelta(days=7)
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
df_list = df_list[~df_list['Code'].str.endswith('5')].copy()
df_list = df_list[['Code', 'Name']].dropna().reset_index(drop=True)

print(f"  KOSPI 보통주: {len(df_list)}개\n")

# ══════════════════════════════════════════════════════════════════
# STEP 2. 재무 데이터 수집
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 2. 재무 데이터 수집")
print("=" * 60)


def get_finstate_value(fs, keywords):
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
        equity     = get_finstate_value(fs, ['자본총계', '자기자본'])
        debt       = get_finstate_value(fs, ['부채총계', '총부채'])
        bps        = get_finstate_value(fs, ['BPS', '주당순자산'])
        dps        = get_finstate_value(fs, ['DPS', '주당배당금', '현금DPS'])

        if not (net_income and equity and equity != 0):
            continue

        roe        = net_income / equity * 100
        debt_ratio = (debt / equity * 100) if (debt and equity != 0) else None

        last_price = None
        try:
            price_df = fdr.DataReader(code, PRICE_DATE_START, PRICE_DATE_END)
            if price_df is not None and not price_df.empty:
                last_price = float(price_df['Close'].iloc[-1])
        except Exception:
            pass

        if not last_price:
            continue

        pbr       = (last_price / bps) if (bps and bps > 0) else None
        div_yield = (dps / last_price * 100) if (dps and dps > 0) else 0.0

        if pbr is None:
            continue

        records.append({
            'Code':    code,
            '기업명':  name,
            'ROE':     round(roe, 2),
            '부채비율': round(debt_ratio, 1) if debt_ratio else None,
            '배당수익률': round(div_yield, 2),
            'PBR':     round(pbr, 2),
        })

        if (i + 1) % 10 == 0:
            print(f"  진행: {i + 1}/{MAX_CNT} — 수집: {len(records)}개")

        time.sleep(0.4)

    except Exception:
        continue

df = pd.DataFrame(records)
print(f"\n  재무 수집 완료: {len(df)}개 종목")

if df.empty:
    print("  데이터 없음. MAX_CNT 확인 후 재실행하세요.")
    import sys; sys.exit()

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
    print("  DART 키 미입력 → 자사주소각 전체 0으로 처리")

df['자사주소각'] = df['Code'].isin(buyback_set).astype(int)

# ══════════════════════════════════════════════════════════════════
# STEP 4. 데이터 정제
# ══════════════════════════════════════════════════════════════════
df = df.dropna(subset=['PBR', 'ROE', '배당수익률', '부채비율'])
df = df[
    df['PBR'].between(0.05, 3.0) &
    df['ROE'].between(-50, 100)  &
    df['부채비율'].between(0, 500)
].reset_index(drop=True)

df['logPBR'] = np.log(df['PBR'])

print(f"\n  최종 분석 종목: {len(df)}개")
if len(df) < 10:
    print("  종목 수 부족. MAX_CNT를 늘려주세요.")
    import sys; sys.exit()

# ══════════════════════════════════════════════════════════════════
# STEP 5. VIF 검사 + OLS 회귀
# ══════════════════════════════════════════════════════════════════
feature_cols = ['ROE', '배당수익률', '자사주소각']
feature_cols = [c for c in feature_cols if df[c].std() > 0]

print("\n" + "=" * 60)
print("  STEP 5a. 다중공선성 검사 (VIF)")
print("=" * 60)

X_raw    = df[feature_cols].values
vif_data = pd.DataFrame({
    'Feature': feature_cols,
    'VIF':     [variance_inflation_factor(X_raw, i) for i in range(len(feature_cols))]
})
print(vif_data.round(2))
print("  ※ VIF > 10: 심각한 다중공선성 → 자동 제거")

high_vif = vif_data[vif_data['VIF'] > 10]['Feature'].tolist()
if high_vif:
    print(f"  ⚠ 제거: {high_vif}")
    feature_cols = [f for f in feature_cols if f not in high_vif]

print("\n" + "=" * 60)
print("  STEP 5b. OLS 회귀 분석 (Target: log PBR)")
print("=" * 60)

scaler = StandardScaler()
X_sc   = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)
y      = df['logPBR'].values
X_sm   = sm.add_constant(X_sc)
model  = sm.OLS(y, X_sm).fit()
print(model.summary())

df['logPBR_예측'] = model.predict(X_sm)
df['PBR_예측']    = np.exp(df['logPBR_예측'])
df['잔차']        = df['PBR'] - df['PBR_예측']
df['저평가여부']   = df['잔차'] < -0.2
undervalued       = df[df['저평가여부']].sort_values('잔차')

print(f"\n  저평가 종목: {len(undervalued)}개 (잔차 < -0.20)")
if not undervalued.empty:
    print(undervalued[['기업명', 'PBR', 'PBR_예측', '잔차', 'ROE', '배당수익률']]
          .head(10).round(3).to_string(index=False))

log_residuals  = df['logPBR'] - df['logPBR_예측']
stat, p_val    = stats.shapiro(log_residuals)
print(f"\n  Shapiro-Wilk (log 잔차): W={stat:.4f}, p={p_val:.4f}")
if p_val > 0.05:
    print("  → 정규성 만족 ✓")
else:
    print("  → 정규성 위반 — 한국 증시 구조적 저평가 경향 반영")

# ══════════════════════════════════════════════════════════════════
# STEP 6. XGBoost 모델 — OLS와 비교
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 6. XGBoost 모델 학습 및 OLS 비교")
print("=" * 60)

X     = df[feature_cols].values
y_raw = df['logPBR'].values

xgb = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    verbosity=0
)
xgb.fit(X, y_raw)

ols_r2 = model.rsquared
xgb_cv = cross_val_score(xgb, X, y_raw, cv=5, scoring='r2').mean()
xgb_r2 = r2_score(y_raw, xgb.predict(X))

print(f"\n  OLS R²:              {ols_r2:.3f}")
print(f"  XGBoost R² (train):  {xgb_r2:.3f}")
print(f"  XGBoost R² (5-CV):   {xgb_cv:.3f}")

if xgb_cv > ols_r2:
    print("  → XGBoost가 비선형 관계를 추가로 포착함 ✓")
else:
    print("  → OLS와 유사 — 선형 관계가 지배적")

# ══════════════════════════════════════════════════════════════════
# STEP 7. 시각화
# ══════════════════════════════════════════════════════════════════

# ── (A~D) OLS 결과 ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('저평가 우량주 발굴 — 선형 회귀 분석 (실제 KOSPI 데이터)',
             fontsize=16, fontweight='bold')
for ax in axes.flatten():
    ax.set_facecolor('#F8FAFC')
fig.patch.set_facecolor('white')

ax1    = axes[0, 0]
n_feat = len(feature_cols)
coef_df = pd.DataFrame({
    'Feature':     feature_cols,
    'Coefficient': model.params[1:n_feat + 1].values,
    'p_value':     model.pvalues[1:n_feat + 1].values,
}).sort_values('Coefficient', ascending=True)

colors = [RED if c > 0 else BLUE for c in coef_df['Coefficient']]
bars   = ax1.barh(coef_df['Feature'], coef_df['Coefficient'],
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

ax2  = axes[0, 1]
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
ax2.set_xlabel('ROE (%)'); ax2.set_ylabel('PBR (배)')
ax2.legend(fontsize=9)

ax3 = axes[1, 0]
ax3.hist(log_residuals, bins=min(25, len(df) // 2 + 1),
         color=BLUE, alpha=0.7, edgecolor='white', density=True)
xn = np.linspace(log_residuals.min(), log_residuals.max(), 200)
ax3.plot(xn, stats.norm.pdf(xn, log_residuals.mean(), log_residuals.std()),
         color=RED, lw=2, label='정규분포')
ax3.axvline(0, color=GRAY, lw=1, linestyle=':')
ax3.set_title('(C) 잔차 분포 & 정규성 (log 기준)', fontweight='bold')
ax3.set_xlabel('log 잔차'); ax3.set_ylabel('밀도')
ax3.legend(fontsize=9)
ax3.text(0.02, 0.95, f'Shapiro p={p_val:.3f}', transform=ax3.transAxes,
         fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRAY))

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
ax4.set_xlim(lim); ax4.set_ylim(lim)
ax4.set_title('(D) 실제 PBR vs 예측 PBR', fontweight='bold')
ax4.set_xlabel('예측 PBR'); ax4.set_ylabel('실제 PBR')
ax4.legend(fontsize=9)
ax4.text(0.02, 0.95, f'R²={model.rsquared:.3f}', transform=ax4.transAxes,
         fontsize=10, fontweight='bold', va='top',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRAY))

plt.tight_layout()
plt.savefig('regression_result_realdata.png', dpi=150, bbox_inches='tight')
plt.show()

# ── (E~F) XGBoost 비교 ───────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('OLS vs XGBoost 비교', fontsize=14, fontweight='bold')

ax5  = axes2[0]
ax5.set_facecolor('#F8FAFC')
bars = ax5.bar(
    ['OLS\nR²', 'XGBoost\nTrain R²', 'XGBoost\n5-CV R²'],
    [ols_r2, xgb_r2, xgb_cv],
    color=[BLUE, ORANGE, GREEN if xgb_cv > ols_r2 else RED],
    alpha=0.85, edgecolor='white', width=0.5
)
for bar, val in zip(bars, [ols_r2, xgb_r2, xgb_cv]):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
ax5.set_ylim(0, max(ols_r2, xgb_r2, xgb_cv) + 0.1)
ax5.set_title('(E) 모델 성능 비교 (R²)', fontweight='bold')
ax5.set_ylabel('R²')

ax6 = axes2[1]
ax6.set_facecolor('#F8FAFC')
importance_df = pd.DataFrame({
    'Feature':    feature_cols,
    'Importance': xgb.feature_importances_
}).sort_values('Importance', ascending=True)

ax6.barh(importance_df['Feature'], importance_df['Importance'],
         color=ORANGE, alpha=0.85, edgecolor='white')
ax6.set_title('(F) XGBoost Feature Importance', fontweight='bold')
ax6.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('xgboost_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n  완료!")
print(f"  저장 파일: regression_result_realdata.png, xgboost_comparison.png")
