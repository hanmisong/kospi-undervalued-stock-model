"""
=============================================================
 단일 종목 심층 분석 — 저평가 진단 모델

 원하는 기업 코드 하나만 넣으면 자동으로:
 1. 해당 종목 재무 데이터 수집
 2. KOSPI 전체 대비 상대적 위치 분석
 3. 저평가 여부 진단
 4. 시각화 리포트 생성

 설치: pip install finance-datareader opendartreader
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import datetime, time, warnings

warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False
for font in ['AppleGothic', 'NanumGothic', 'Malgun Gothic']:
    try:
        plt.rcParams['font.family'] = font
        break
    except Exception:
        continue

import FinanceDataReader as fdr


# ══════════════════════════════════════════════════════════════════
# ★ 설정값 — 여기만 바꾸면 됩니다
# ══════════════════════════════════════════════════════════════════
TARGET_CODE   = '042700'   # 분석할 종목 코드 (한미반도체: 042700)
BENCHMARK_CNT = 150        # 비교군 종목 수
DART_API_KEY  = ""         # DART API 키 (없어도 실행 가능)

# 주가 기준: 최근 1주일 자동 설정
today            = datetime.date.today()
PRICE_DATE_START = (today - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
PRICE_DATE_END   = today.strftime('%Y-%m-%d')


# ══════════════════════════════════════════════════════════════════
# 재무 데이터 추출 함수
# ══════════════════════════════════════════════════════════════════
def get_finstate_value(fs, keywords):
    """재무제표 DataFrame에서 키워드 컬럼의 최신 값 반환"""
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


def fetch_stock_data(code, name=""):
    """단일 종목 재무 + 주가 데이터 수집"""
    try:
        fs = fdr.SnapDataReader(f'NAVER/FINSTATE-2Y/{code}')
        if fs is None or fs.empty:
            return None

        # ── 미래 예측 행 제거 (NaN 전체 행 + 2024년 이후 행) ────
        fs = fs.dropna(how='all')
        fs = fs[fs.index <= pd.Timestamp('2024-12-31')]
        if fs.empty:
            return None

        # ── 재무 항목 추출 ────────────────────────────────────────
        net_income = get_finstate_value(fs, ['당기순이익', '순이익'])
        equity     = get_finstate_value(fs, ['자본총계', '자기자본'])
        debt       = get_finstate_value(fs, ['부채총계', '총부채'])
        bps        = get_finstate_value(fs, ['BPS(원)', 'BPS', '주당순자산'])
        dps        = get_finstate_value(fs, ['현금DPS(원)', 'DPS(원)', 'DPS', '주당배당금'])
        roe_direct = get_finstate_value(fs, ['ROE(%)', 'ROE'])
        pbr_direct = get_finstate_value(fs, ['PBR(배)', 'PBR'])   # ★ 네이버 제공 PBR

        if not (equity and equity != 0):
            return None

        roe        = roe_direct if (roe_direct and roe_direct != 0) \
                     else (net_income / equity * 100 if net_income else None)
        debt_ratio = (debt / equity * 100) if (debt and equity != 0) else None

        # ── 주가 수집 ─────────────────────────────────────────────
        price_df   = fdr.DataReader(code, PRICE_DATE_START, PRICE_DATE_END)
        if price_df is None or price_df.empty:
            return None
        last_price = float(price_df['Close'].iloc[-1])

        # ── PBR: 네이버 직접값 우선, 없으면 주가/BPS 계산 ─────────
        # pbr_direct를 덮어쓰지 않도록 분리
        if pbr_direct and 0 < pbr_direct < 200:
            pbr = pbr_direct
        elif bps and bps > 0:
            pbr = last_price / bps
        else:
            pbr = None

        div_yield = (dps / last_price * 100) if (dps and dps > 0) else 0.0

        if not (roe and pbr):
            return None

        return {
            'Code':       code,
            '기업명':     name,
            'ROE':        round(roe, 2),
            '부채비율':   round(debt_ratio, 1) if debt_ratio else None,
            '배당수익률': round(div_yield, 2),
            'PBR':        round(pbr, 2),
            '주가':       last_price,
            'BPS':        bps,
        }

    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════════
# STEP 1. 타깃 종목 데이터 수집
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1. 타깃 종목 분석")
print("=" * 60)

df_list = fdr.StockListing('KOSPI')
df_list = df_list.reset_index(drop=True)
df_list = df_list.loc[:, ~df_list.columns.duplicated()]
df_list['Code'] = df_list['Code'].astype(str).str.zfill(6)

target_name_row = df_list[df_list['Code'] == TARGET_CODE]
TARGET_NAME     = target_name_row['Name'].iloc[0] if not target_name_row.empty else TARGET_CODE

print(f"  종목: {TARGET_NAME} ({TARGET_CODE})")

target_data = fetch_stock_data(TARGET_CODE, TARGET_NAME)
if not target_data:
    print("타깃 종목 데이터 수집 실패")
    import sys;
    sys.exit()

print(f"  PBR:       {target_data['PBR']}배")
print(f"  ROE:       {target_data['ROE']}%")
print(f"  배당수익률: {target_data['배당수익률']}%")
print(f"  부채비율:   {target_data['부채비율']}%")
print(f"  주가:       {int(target_data['주가']):,}원")


# ══════════════════════════════════════════════════════════════════
# STEP 2. 비교군(KOSPI) 데이터 수집
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"  STEP 2. 비교군 수집 (상위 {BENCHMARK_CNT}개 종목)")
print("=" * 60)

df_list_filtered = df_list[~df_list['Code'].str.endswith('5')].copy()
df_list_filtered = df_list_filtered[['Code', 'Name']].dropna().reset_index(drop=True)

records = []
for i, row in df_list_filtered.head(BENCHMARK_CNT).iterrows():
    code, name = row['Code'], row['Name']
    if code == TARGET_CODE:
        continue
    data = fetch_stock_data(code, name)
    if data:
        records.append(data)
    if (i + 1) % 30 == 0:
        print(f"  진행: {i+1}/{BENCHMARK_CNT} — 수집: {len(records)}개")
    time.sleep(0.35)

# 타깃 종목 강제 포함
if TARGET_CODE not in [r['Code'] for r in records]:
    records.append(target_data)

df = pd.DataFrame(records)
print(f"\n  비교군 수집 완료: {len(df)}개 종목")


# ══════════════════════════════════════════════════════════════════
# STEP 3. 자사주 소각 탐지 (DART)
# ══════════════════════════════════════════════════════════════════
buyback_set = set()
if DART_API_KEY:
    try:
        import OpenDartReader
        dart = OpenDartReader(DART_API_KEY)
        for code in [TARGET_CODE] + df['Code'].tolist()[:30]:
            try:
                disc = dart.list(code, start='2023-01-01', end='2024-12-31')
                if disc is not None and not disc.empty:
                    if disc['report_nm'].str.contains('자기주식소각', na=False).any():
                        buyback_set.add(code)
                time.sleep(0.1)
            except Exception:
                pass
        print(f"\n  자사주 소각 탐지: {len(buyback_set)}개")
    except ImportError:
        pass

df['자사주소각'] = df['Code'].isin(buyback_set).astype(int)


# ══════════════════════════════════════════════════════════════════
# STEP 4. 데이터 정제 + 회귀 모델
# ══════════════════════════════════════════════════════════════════
df = df.dropna(subset=['PBR', 'ROE', '배당수익률', '부채비율'])

# 타깃 종목은 필터 제외, 비교군만 정제
target_temp = df[df['Code'] == TARGET_CODE]
others      = df[df['Code'] != TARGET_CODE]
others      = others[
    others['PBR'].between(0.05, 3.0) &
    others['ROE'].between(-50, 100)  &
    others['부채비율'].between(0, 500)
]

df = pd.concat([target_temp, others]).reset_index(drop=True)


df['logPBR'] = np.log(df['PBR'])

feature_cols = ['ROE', '배당수익률', '자사주소각']
feature_cols = [c for c in feature_cols if df[c].std() > 0]

scaler = StandardScaler()
X_sc   = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)
y      = df['logPBR'].values
X_sm   = sm.add_constant(X_sc)
model  = sm.OLS(y, X_sm).fit()

df['logPBR_예측'] = model.predict(X_sm)
df['PBR_예측']    = np.exp(df['logPBR_예측'])
df['잔차']        = df['PBR'] - df['PBR_예측']
df['저평가여부']   = df['잔차'] < -0.2

# 타깃 종목 결과 추출
target_row      = df[df['Code'] == TARGET_CODE].iloc[0]
target_residual = target_row['잔차']
target_pbr      = target_row['PBR']
target_pbr_pred = target_row['PBR_예측']
target_roe      = target_row['ROE']
target_div      = target_row['배당수익률']
is_undervalued  = target_row['저평가여부']

# 백분위
pbr_percentile   = (df['PBR']  < target_pbr).mean()      * 100
roe_percentile   = (df['ROE']  < target_roe).mean()      * 100
resid_percentile = (df['잔차'] < target_residual).mean() * 100

print(f"\n{'=' * 60}")
print(f"  [{TARGET_NAME}] 분석 결과")
print("=" * 60)
print(f"  실제 PBR:    {target_pbr:.2f}배")
print(f"  예측 PBR:    {target_pbr_pred:.2f}배")
print(f"  잔차:        {target_residual:.3f}")
print(f"  저평가 여부: {'✅ 저평가' if is_undervalued else '⚠ 적정/고평가'}")
print(f"  PBR 하위:    {pbr_percentile:.1f}%ile")
print(f"  ROE 상위:    {100-roe_percentile:.1f}%ile")
print(f"  모델 R²:     {model.rsquared:.3f}")


# ══════════════════════════════════════════════════════════════════
# STEP 5. 시각화
# ══════════════════════════════════════════════════════════════════
BLUE         = '#2563EB'
RED          = '#DC2626'
ORANGE       = '#F97316'
GRAY         = '#9CA3AF'
GREEN        = '#10B981'
TARGET_COLOR = RED if is_undervalued else ORANGE

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

verdict     = '저평가 종목으로 판단됨' if is_undervalued else '⚠ 적정/고평가 구간'
title_color = RED if is_undervalued else '#374151'
fig.suptitle(
    f'{TARGET_NAME} ({TARGET_CODE}) 저평가 진단 리포트\n{verdict}',
    fontsize=15, fontweight='bold', color=title_color, y=0.98
)

bg = '#F8FAFC'

# (A) KPI 3개 ──────────────────────────────────────────────────
kpi_list = [
    ('실제 PBR',  f'{target_pbr:.2f}배',      f'KOSPI 하위 {pbr_percentile:.0f}%', BLUE),
    ('예측 PBR',  f'{target_pbr_pred:.2f}배',  '모델 적정가',                        GRAY),
    ('잔차',      f'{target_residual:+.3f}',   '음수 = 저평가',                      TARGET_COLOR),
]
for idx, (label, val, sub, color) in enumerate(kpi_list):
    ax_k = fig.add_subplot(gs[0, idx])
    ax_k.set_facecolor(bg)
    ax_k.set_xlim(0, 1); ax_k.set_ylim(0, 1)
    ax_k.axis('off')
    ax_k.text(0.5, 0.72, val,   ha='center', va='center', fontsize=26,
              fontweight='bold', color=color, transform=ax_k.transAxes)
    ax_k.text(0.5, 0.42, label, ha='center', va='center', fontsize=12,
              color='#374151',   transform=ax_k.transAxes)
    ax_k.text(0.5, 0.18, sub,   ha='center', va='center', fontsize=9,
              color=GRAY,        transform=ax_k.transAxes)
    for spine in ax_k.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(2)

# (B) ROE vs PBR 산점도 ────────────────────────────────────────
ax_b   = fig.add_subplot(gs[1, :2])
ax_b.set_facecolor(bg)
other  = df[df['Code'] != TARGET_CODE]
under  = other[other['저평가여부']]
normal = other[~other['저평가여부']]

ax_b.scatter(normal['ROE'], normal['PBR'], color=BLUE,  alpha=0.35, s=35, label='일반 종목')
ax_b.scatter(under['ROE'],  under['PBR'],  color=GRAY,  alpha=0.5,  s=35, marker='*', label='기타 저평가')
ax_b.scatter(target_roe,   target_pbr,    color=TARGET_COLOR, s=220,
             marker='*', zorder=10, label=f'★ {TARGET_NAME}',
             edgecolors='white', linewidths=1)

sl, ic, *_ = stats.linregress(df['ROE'], df['PBR'])
xl = np.linspace(df['ROE'].min(), df['ROE'].max(), 200)
ax_b.plot(xl, ic+sl*xl, color=ORANGE, lw=2, linestyle='--', alpha=0.8, label='회귀선')

ax_b.annotate(
    f'{TARGET_NAME}\nPBR={target_pbr:.2f} (예측={target_pbr_pred:.2f})',
    xy=(target_roe, target_pbr),
    xytext=(target_roe + max(df['ROE'].std()*0.3, 1), target_pbr + 0.1),
    fontsize=9, color=TARGET_COLOR, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=TARGET_COLOR, lw=1.2)
)
ax_b.set_title('(B) ROE vs PBR — 비교군 내 위치', fontweight='bold')
ax_b.set_xlabel('ROE (%)'); ax_b.set_ylabel('PBR (배)')
ax_b.legend(fontsize=8, loc='upper left')

# (C) 잔차 분포 ────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 2])
ax_c.set_facecolor(bg)
log_res        = df['logPBR'] - df['logPBR_예측']
target_log_res = np.log(target_pbr) - np.log(target_pbr_pred)

ax_c.hist(log_res, bins=20, color=BLUE, alpha=0.6, edgecolor='white', density=True)
xn = np.linspace(log_res.min(), log_res.max(), 200)
ax_c.plot(xn, stats.norm.pdf(xn, log_res.mean(), log_res.std()),
          color=GRAY, lw=1.5, linestyle='--', label='정규분포')
ax_c.axvline(target_log_res, color=TARGET_COLOR, lw=2.5, label=f'{TARGET_NAME} 잔차')
ax_c.axvline(0, color=GRAY, lw=1, linestyle=':')
ax_c.set_title('(C) 잔차 분포 내 위치', fontweight='bold')
ax_c.set_xlabel('log 잔차'); ax_c.set_ylabel('밀도')
ax_c.legend(fontsize=8)

# (D) 백분위 바 차트 ───────────────────────────────────────────
ax_d = fig.add_subplot(gs[2, :])
ax_d.set_facecolor(bg)

metrics = {
    'PBR 저평가도\n(낮을수록 Good)':  pbr_percentile,
    'ROE 우수성\n(높을수록 Good)':    100 - roe_percentile,
    '잔차 저평가도\n(낮을수록 Good)': resid_percentile,
    '배당수익률\n백분위':             (df['배당수익률'] < target_div).mean() * 100,
}
labels = list(metrics.keys())
values = list(metrics.values())

good_if_high = ['ROE 우수성\n(높을수록 Good)', '배당수익률\n백분위']
bar_colors = [
    GREEN if (lbl in good_if_high and val > 50) or
             (lbl not in good_if_high and val < 50)
    else RED
    for lbl, val in zip(labels, values)
]
bars = ax_d.barh(labels, values, color=bar_colors, alpha=0.75,
                 edgecolor='white', linewidth=0.8, height=0.5)
ax_d.axvline(50, color=GRAY, lw=1.2, linestyle='--', alpha=0.7)

for bar, val in zip(bars, values):
    ax_d.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
              f'{val:.1f}%ile', va='center', fontsize=10,
              fontweight='bold', color='#374151')

ax_d.set_xlim(0, 110)
ax_d.set_title(f'(D) {TARGET_NAME} — KOSPI 비교군 내 백분위 순위', fontweight='bold')
ax_d.set_xlabel('백분위 (%ile)')
ax_d.text(51, -0.7, '← 저평가  |  고평가 →', fontsize=8, color=GRAY, ha='center')

plt.savefig(f'{TARGET_NAME}_진단리포트.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n 완료: {TARGET_NAME}_진단리포트.png")