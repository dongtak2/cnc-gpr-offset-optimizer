from pathlib import Path
from io import BytesIO
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "gpr_v8_model_bundle.joblib"
CONFIG_FILE = BASE_DIR / "demo_config.json"

st.set_page_config(
    page_title="CNC 설비 옵셋 GPR 모델",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
    .main-title {font-size: 2.15rem; font-weight: 800; margin-bottom: 0.15rem;}
    .sub-title {color: #64748b; margin-bottom: 1.3rem;}
    .section-note {color:#64748b; font-size:0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

bundle = load_model()
cfg = load_config()
holdout = cfg.get("holdout_metrics", {})

PARAMS = cfg["parameters"]
FEATURE_MIN = cfg["feature_min"]
FEATURE_MAX = cfg["feature_max"]
FEATURE_DEFAULT = cfg["feature_default"]

# 최종 GPR의 학습 입력을 원래 Parameter 스케일로 복원한다.
_ccw_model = bundle["models"]["CCW"]
TRAIN_X = _ccw_model["x_scaler"].inverse_transform(_ccw_model["gp"].X_train_)
TRAIN_X = np.rint(TRAIN_X).astype(float)
OBSERVED_VALUES = [np.unique(TRAIN_X[:, j]) for j in range(TRAIN_X.shape[1])]

MINS = np.asarray([FEATURE_MIN[p] for p in PARAMS], dtype=float)
MAXS = np.asarray([FEATURE_MAX[p] for p in PARAMS], dtype=float)
RANGES = np.where(MAXS > MINS, MAXS - MINS, 1.0)


def predict_matrix(X):
    X = np.asarray(X, dtype=float)
    outputs = {}
    for name in ("CCW", "CW"):
        model = bundle["models"][name]
        xs = model["x_scaler"].transform(X)
        pred_s, std_s = model["gp"].predict(xs, return_std=True)
        yscale = float(model["y_scaler"].scale_[0])
        ymean = float(model["y_scaler"].mean_[0])
        outputs[name] = (pred_s * yscale + ymean, std_s * yscale)
    return outputs


@st.cache_data
def training_predictions():
    out = predict_matrix(TRAIN_X)
    ccw, ccw_std = out["CCW"]
    cw, cw_std = out["CW"]
    return ccw, ccw_std, cw, cw_std


def count_outside_training_range(X):
    X = np.asarray(X, dtype=float)
    mask = (X < MINS) | (X > MAXS)
    return mask.sum(axis=1), mask


def make_excel_download(df):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="결과")
    return bio.getvalue()


def pair_layout():
    pairs = []
    for i in range(0, len(PARAMS), 2):
        px, pz = PARAMS[i], PARAMS[i + 1]
        base = px.replace("param_", "").rsplit("_", 1)[0]
        pairs.append((base, px, pz))
    return pairs


def render_parameter_inputs(prefix, default_vector=None):
    if default_vector is None:
        defaults = {p: FEATURE_DEFAULT[p] for p in PARAMS}
    else:
        defaults = {p: float(v) for p, v in zip(PARAMS, default_vector)}

    values = {}
    pairs = pair_layout()
    for start in range(0, len(pairs), 3):
        cols = st.columns(3)
        for j, (base, px, pz) in enumerate(pairs[start:start + 3]):
            with cols[j]:
                st.markdown(f"**Parameter {base}**")
                for p in (px, pz):
                    label = p.rsplit("_", 1)[1]
                    lo = FEATURE_MIN[p]
                    hi = FEATURE_MAX[p]
                    default = defaults[p]
                    help_text = f"학습 범위: {lo:g} ~ {hi:g}"

                    if lo >= 0 and hi <= 1 and float(lo).is_integer() and float(hi).is_integer():
                        values[p] = st.selectbox(
                            label,
                            options=[0, 1],
                            index=int(round(np.clip(default, 0, 1))),
                            key=f"{prefix}_{p}",
                            help=help_text,
                        )
                    else:
                        values[p] = st.number_input(
                            label,
                            value=int(round(default)),
                            step=1,
                            key=f"{prefix}_{p}",
                            help=help_text,
                        )
    return np.asarray([values[p] for p in PARAMS], dtype=float)


def deduplicate_rows(X):
    seen = set()
    out = []
    for row in np.asarray(X, dtype=float):
        key = tuple(np.rint(row).astype(int).tolist())
        if key not in seen:
            seen.add(key)
            out.append(np.asarray(key, dtype=float))
    return np.asarray(out, dtype=float)


def generate_local_candidates(current, n=3500, seed=20260908):
    rng = np.random.default_rng(seed)
    current = np.asarray(current, dtype=float)
    base = np.clip(np.rint(current), MINS, MAXS)
    candidates = np.tile(base, (n, 1))

    for j in range(len(PARAMS)):
        observed = OBSERVED_VALUES[j]
        # 이진/희소 변수는 실제 학습에서 관찰된 값 위주로 탐색한다.
        if len(observed) <= 12:
            mutate = rng.random(n) < 0.16
            if mutate.any():
                candidates[mutate, j] = rng.choice(observed, size=int(mutate.sum()))
        else:
            mutate = rng.random(n) < 0.42
            if mutate.any():
                sigma = max(RANGES[j] * 0.07, 1.0)
                proposed = base[j] + rng.normal(0, sigma, size=int(mutate.sum()))
                candidates[mutate, j] = np.rint(np.clip(proposed, MINS[j], MAXS[j]))

    # 학습된 공간에서 GPR이 양호하게 평가한 기존 조합도 후보에 포함한다.
    tccw, tccw_s, tcw, tcw_s = training_predictions()
    train_score = (tccw + tcw) / 2 + 0.35 * np.maximum(tccw_s, tcw_s)
    seed_idx = np.argsort(train_score)[:120]
    extra = TRAIN_X[seed_idx]

    return deduplicate_rows(np.vstack([base, candidates, extra]))


def generate_global_candidates(n=5000, seed=20260908):
    rng = np.random.default_rng(seed)

    tccw, tccw_s, tcw, tcw_s = training_predictions()
    train_score = (tccw + tcw) / 2 + 0.35 * np.maximum(tccw_s, tcw_s)
    seed_idx = np.argsort(train_score)[:150]
    seeds = TRAIN_X[seed_idx]

    rows = []
    # 좋은 학습 조합 주변을 세밀하게 탐색한다.
    for _ in range(int(n * 0.72)):
        base = seeds[rng.integers(0, len(seeds))].copy()
        for j in range(len(PARAMS)):
            observed = OBSERVED_VALUES[j]
            if len(observed) <= 12:
                if rng.random() < 0.12:
                    base[j] = rng.choice(observed)
            else:
                if rng.random() < 0.32:
                    sigma = max(RANGES[j] * 0.09, 1.0)
                    base[j] = np.rint(np.clip(base[j] + rng.normal(0, sigma), MINS[j], MAXS[j]))
        rows.append(base)

    # 일부는 전체 학습 범위를 넓게 탐색한다.
    for _ in range(n - len(rows)):
        row = np.empty(len(PARAMS), dtype=float)
        for j in range(len(PARAMS)):
            observed = OBSERVED_VALUES[j]
            if len(observed) <= 12:
                row[j] = rng.choice(observed)
            else:
                row[j] = rng.integers(int(np.floor(MINS[j])), int(np.ceil(MAXS[j])) + 1)
        rows.append(row)

    return deduplicate_rows(np.vstack([seeds, np.asarray(rows)]))


def score_candidates(X, current, mode):
    out = predict_matrix(X)
    ccw, ccw_std = out["CCW"]
    cw, cw_std = out["CW"]
    avg = (ccw + cw) / 2.0
    max_std = np.maximum(ccw_std, cw_std)

    current = np.asarray(current, dtype=float)
    change_ratio = np.mean(np.abs((X - current) / RANGES), axis=1)
    changed_count = np.sum(np.abs(X - current) > 1e-9, axis=1)

    if mode == "성능 우선":
        uncertainty_weight = 0.20
        change_weight = 0.45
    elif mode == "안정성 우선":
        uncertainty_weight = 0.90
        change_weight = 1.60
    else:
        uncertainty_weight = 0.50
        change_weight = 1.00

    score = avg + uncertainty_weight * max_std + change_weight * change_ratio

    return {
        "ccw": ccw,
        "ccw_std": ccw_std,
        "cw": cw,
        "cw_std": cw_std,
        "avg": avg,
        "max_std": max_std,
        "change_ratio": change_ratio,
        "changed_count": changed_count,
        "score": score,
    }


def diverse_top_indices(X, score, top_n=5, min_distance=0.08):
    order = np.argsort(score)
    chosen = []
    normalized = (X - MINS) / RANGES

    for idx in order:
        if not chosen:
            chosen.append(int(idx))
        else:
            distances = np.linalg.norm(normalized[idx] - normalized[chosen], axis=1) / np.sqrt(X.shape[1])
            if np.min(distances) >= min_distance:
                chosen.append(int(idx))
        if len(chosen) >= top_n:
            break

    if len(chosen) < top_n:
        for idx in order:
            if int(idx) not in chosen:
                chosen.append(int(idx))
            if len(chosen) >= top_n:
                break
    return chosen


def recommend(current, search_scope, objective_mode, top_n):
    current = np.asarray(current, dtype=float)
    if search_scope == "현재 조건 주변":
        candidates = generate_local_candidates(current)
    else:
        candidates = generate_global_candidates()

    candidates = np.clip(np.rint(candidates), MINS, MAXS)
    candidates = deduplicate_rows(candidates)

    metrics = score_candidates(candidates, current, objective_mode)
    selected = diverse_top_indices(candidates, metrics["score"], top_n=top_n)

    current_out = predict_matrix(current.reshape(1, -1))
    current_ccw = float(current_out["CCW"][0][0])
    current_cw = float(current_out["CW"][0][0])
    current_avg = (current_ccw + current_cw) / 2.0

    records = []
    for rank, i in enumerate(selected, 1):
        improvement = current_avg - float(metrics["avg"][i])
        pct = (improvement / current_avg * 100.0) if abs(current_avg) > 1e-12 else 0.0
        record = {
            "순위": rank,
            "추천ID": f"REC{rank:02d}",
            "GPR_CCW_예측_um": float(metrics["ccw"][i]),
            "CCW_예측σ_um": float(metrics["ccw_std"][i]),
            "GPR_CW_예측_um": float(metrics["cw"][i]),
            "CW_예측σ_um": float(metrics["cw_std"][i]),
            "예측_양방향평균_um": float(metrics["avg"][i]),
            "최대_예측σ_um": float(metrics["max_std"][i]),
            "현재대비_예상개선_um": improvement,
            "현재대비_예상개선율_pct": pct,
            "변경_Parameter수": int(metrics["changed_count"][i]),
            "추천점수": float(metrics["score"][i]),
        }
        for p, v in zip(PARAMS, candidates[i]):
            record[p] = int(round(v))
        records.append(record)

    return pd.DataFrame(records), current_ccw, current_cw, current_avg


def compact_recommendation_table(df):
    cols = [
        "순위", "추천ID",
        "GPR_CCW_예측_um", "GPR_CW_예측_um", "예측_양방향평균_um",
        "최대_예측σ_um", "현재대비_예상개선_um", "현재대비_예상개선율_pct",
        "변경_Parameter수",
    ]
    return df[cols].copy()


st.markdown('<div class="main-title">CNC 설비 옵셋 최적화 GPR 모델</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">설비 Parameter에 따른 CCW/CW 원형편차를 예측하고, 더 낮은 원형편차가 예상되는 Parameter 조합을 추천합니다.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "단일 조합 예측",
    "최적 Parameter 추천",
    "Excel 일괄 예측",
    "모델 설명",
])

with tab1:
    st.markdown("### 설비 Parameter 입력")
    st.caption("초기값은 v8 학습 데이터의 중앙값입니다. 각 입력란에서 학습 Min~Max를 확인할 수 있습니다.")
    X_single = render_parameter_inputs("single")

    if st.button("GPR 예측 실행", type="primary", use_container_width=True, key="single_predict"):
        out = predict_matrix(X_single.reshape(1, -1))
        outside_count, outside_mask = count_outside_training_range(X_single.reshape(1, -1))

        ccw_pred, ccw_std = float(out["CCW"][0][0]), float(out["CCW"][1][0])
        cw_pred, cw_std = float(out["CW"][0][0]), float(out["CW"][1][0])
        avg_pred = (ccw_pred + cw_pred) / 2.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CCW 예측 원형편차", f"{ccw_pred:.3f} μm")
        c2.metric("CW 예측 원형편차", f"{cw_pred:.3f} μm")
        c3.metric("양방향 평균", f"{avg_pred:.3f} μm")
        c4.metric("최대 예측 σ", f"{max(ccw_std, cw_std):.3f} μm")

        result_df = pd.DataFrame(
            [
                ["CCW", ccw_pred, ccw_std, ccw_pred - 1.96 * ccw_std, ccw_pred + 1.96 * ccw_std],
                ["CW", cw_pred, cw_std, cw_pred - 1.96 * cw_std, cw_pred + 1.96 * cw_std],
            ],
            columns=["방향", "예측값 (μm)", "σ (μm)", "95% 하한 (μm)", "95% 상한 (μm)"],
        )
        st.dataframe(
            result_df.style.format({
                "예측값 (μm)": "{:.3f}",
                "σ (μm)": "{:.3f}",
                "95% 하한 (μm)": "{:.3f}",
                "95% 상한 (μm)": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        if outside_count[0] > 0:
            bad = [p for p, flag in zip(PARAMS, outside_mask[0]) if flag]
            st.markdown("**학습 범위 확인:** " + ", ".join(bad) + " 항목이 학습 Min~Max 범위를 벗어났습니다.")
        else:
            st.markdown("**학습 범위 확인:** 26개 Parameter가 모두 현재 v8 학습 Min~Max 범위 안에 있습니다.")

with tab2:
    st.markdown("### 현재 설비 조건")
    st.caption("현재 사용 중인 26개 Parameter를 입력하면 GPR이 후보 조합을 탐색하고 더 낮은 원형편차가 예상되는 조합을 제시합니다.")
    X_current = render_parameter_inputs("recommend")

    opt1, opt2, opt3 = st.columns(3)
    with opt1:
        search_scope = st.selectbox(
            "탐색 범위",
            ["현재 조건 주변", "전체 학습 범위"],
            help="현재 조건 주변은 변경량을 줄인 개선안을 찾고, 전체 학습 범위는 더 넓은 후보 공간을 탐색합니다.",
        )
    with opt2:
        objective_mode = st.selectbox(
            "추천 기준",
            ["균형형", "성능 우선", "안정성 우선"],
            help="균형형은 예상 원형편차, 불확실성, 현재값 대비 변경량을 함께 고려합니다.",
        )
    with opt3:
        top_n = st.selectbox("추천 개수", [5, 10], index=0)

    if st.button("최적 Parameter 추천 실행", type="primary", use_container_width=True, key="recommend_button"):
        with st.spinner("GPR로 후보 조합을 평가하고 있습니다."):
            rec_df, curr_ccw, curr_cw, curr_avg = recommend(
                X_current, search_scope, objective_mode, int(top_n)
            )

        best = rec_df.iloc[0]

        st.markdown("### 추천 결과")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재 예상 평균", f"{curr_avg:.3f} μm")
        c2.metric("추천 1 예상 평균", f'{best["예측_양방향평균_um"]:.3f} μm')
        c3.metric("예상 개선", f'{best["현재대비_예상개선_um"]:.3f} μm')
        c4.metric("예상 개선율", f'{best["현재대비_예상개선율_pct"]:.1f}%')

        compact = compact_recommendation_table(rec_df)
        st.dataframe(
            compact.style.format({
                "GPR_CCW_예측_um": "{:.3f}",
                "GPR_CW_예측_um": "{:.3f}",
                "예측_양방향평균_um": "{:.3f}",
                "최대_예측σ_um": "{:.3f}",
                "현재대비_예상개선_um": "{:.3f}",
                "현재대비_예상개선율_pct": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### 추천 1 Parameter 변경 내용")
        best_vector = np.asarray([best[p] for p in PARAMS], dtype=float)
        change_rows = []
        for p, cur, new in zip(PARAMS, X_current, best_vector):
            if abs(cur - new) > 1e-9:
                change_rows.append([p, int(round(cur)), int(round(new)), int(round(new - cur))])

        if change_rows:
            change_df = pd.DataFrame(change_rows, columns=["Parameter", "현재값", "추천값", "변화량"])
            st.dataframe(change_df, use_container_width=True, hide_index=True)
        else:
            st.write("추천 1은 현재 Parameter와 동일합니다.")

        export_cols = ["순위", "추천ID"] + PARAMS + [
            "GPR_CCW_예측_um", "CCW_예측σ_um",
            "GPR_CW_예측_um", "CW_예측σ_um",
            "예측_양방향평균_um", "최대_예측σ_um",
            "현재대비_예상개선_um", "현재대비_예상개선율_pct",
            "변경_Parameter수", "추천점수",
        ]
        export_df = rec_df[export_cols].copy()
        st.download_button(
            "추천 결과 Excel 다운로드",
            make_excel_download(export_df),
            file_name="GPR_v8_최적Parameter추천.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        st.markdown(
            '<div class="section-note">추천 후보는 v8 학습 Min~Max 범위 안에서 생성하며, '
            '추천 점수는 예상 원형편차, GPR 불확실성, 현재 Parameter 대비 변경량을 함께 반영합니다.</div>',
            unsafe_allow_html=True,
        )

with tab3:
    st.markdown("### 여러 Parameter 조합 일괄 예측")
    st.write("`ID + 26개 Parameter` 형식의 Excel(.xlsx) 또는 CSV 파일을 업로드하면 전체 조합을 한 번에 예측합니다.")

    template_df = pd.DataFrame(columns=["ID"] + PARAMS)
    st.download_button(
        "빈 입력 템플릿 CSV 다운로드",
        template_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="GPR_v8_입력템플릿.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Excel 또는 CSV 업로드", type=["xlsx", "csv"])

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".xlsx"):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded)

            missing = [c for c in PARAMS if c not in df.columns]
            if missing:
                st.error("필수 Parameter 열이 없습니다: " + ", ".join(missing))
            else:
                if "ID" not in df.columns:
                    df.insert(0, "ID", [f"ROW{i + 1:03d}" for i in range(len(df))])

                X = df[PARAMS].astype(float).to_numpy()
                out = predict_matrix(X)
                outside_count, _ = count_outside_training_range(X)
                ccw_pred, ccw_std = out["CCW"]
                cw_pred, cw_std = out["CW"]

                result = df.copy()
                result["GPR_CCW_예측_um"] = ccw_pred
                result["CCW_예측σ_um"] = ccw_std
                result["GPR_CW_예측_um"] = cw_pred
                result["CW_예측σ_um"] = cw_std
                result["예측_양방향평균_um"] = (ccw_pred + cw_pred) / 2
                result["최대_예측σ_um"] = np.maximum(ccw_std, cw_std)
                result["학습범위밖_파라미터수"] = outside_count
                result = result.sort_values("예측_양방향평균_um", ascending=True).reset_index(drop=True)
                result.insert(1, "예측평균_순위", np.arange(1, len(result) + 1))

                st.write(f"{len(result)}개 조합 예측 완료")
                st.dataframe(result, use_container_width=True, hide_index=True)

                st.download_button(
                    "예측 결과 Excel 다운로드",
                    make_excel_download(result),
                    file_name="GPR_v8_일괄예측결과.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                )
        except Exception as e:
            st.exception(e)

with tab4:
    st.markdown("### 모델 구조")
    st.markdown(
        """
        **입력:** 26개 CNC 설비 Parameter  
        **출력:** CCW / CW Ballbar 원형편차  
        **학습 타깃:** 정확히 180.000° 지점 한 점을 제외해 재계산한 Circularity  
        **반복 측정 처리:** 동일 Parameter 조합의 반복 측정값을 평균하여 GPR 학습  
        **모델:** CCW GPR과 CW GPR을 각각 독립적으로 학습  
        **커널:** Constant Kernel × Matérn(ν=1.5) + White Kernel
        """
    )

    st.markdown("### GPR 특징")
    st.write(
        "원형편차 예측값 μ와 함께 예측 표준편차 σ를 제공합니다. "
        "최적 Parameter 추천에서는 예상 원형편차뿐 아니라 이 불확실성과 현재 조건 대비 변경량도 함께 평가합니다."
    )

    st.markdown("### 현재 v8 데이터")
    c1, c2, c3 = st.columns(3)
    c1.metric("측정 데이터", f'{cfg["training_rows"]:,}건')
    c2.metric("고유 Parameter 조합", f'{cfg["unique_combinations"]:,}개')
    c3.metric("입력 변수", "26개")

    if holdout:
        st.markdown("### 260903 신규 100조합 홀드아웃 검증")
        metrics_df = pd.DataFrame([
            ["CCW", holdout["CCW"]["R2"], holdout["CCW"]["MAE_um"], holdout["CCW"]["RMSE_um"], holdout["CCW"]["PI95_coverage"]],
            ["CW", holdout["CW"]["R2"], holdout["CW"]["MAE_um"], holdout["CW"]["RMSE_um"], holdout["CW"]["PI95_coverage"]],
        ], columns=["방향", "R²", "MAE (μm)", "RMSE (μm)", "95% 구간 포함률"])
        st.dataframe(
            metrics_df.style.format({
                "R²": "{:.3f}",
                "MAE (μm)": "{:.3f}",
                "RMSE (μm)": "{:.3f}",
                "95% 구간 포함률": "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Parameter 추천 방식")
    st.markdown(
        """
        1. 현재 조건 주변 또는 전체 학습 범위에서 다수의 후보 Parameter 조합 생성  
        2. 각 후보를 GPR에 입력해 CCW/CW 원형편차와 σ 계산  
        3. 예상 원형편차, 불확실성, 현재값 대비 변경량을 이용해 후보 점수 계산  
        4. 서로 지나치게 유사한 후보는 제거  
        5. 최종 추천 조합을 순위별로 제시
        """
    )
