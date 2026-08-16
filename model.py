"""
Retail Demand Prediction & Forecasting - Single File ML Application

Run:
    pip install pandas numpy scikit-learn joblib streamlit matplotlib
    streamlit run model.py

The app can:
- Upload a retail CSV
- Automatically detect date / demand columns
- Select store/product columns
- Clean and aggregate data
- Create leakage-safe lag, rolling and calendar features
- Train a Gradient Boosting demand model
- Compare against a seasonal-naive baseline
- Evaluate MAE, RMSE, MAPE, sMAPE and R²
- Generate recursive future forecasts
- Download forecasts and test predictions

Expected CSV examples:
    date,store_id,item_id,sales
    2025-01-01,S01,P001,120
    2025-01-02,S01,P001,132

or:
    Date,Product,Store,Units_Sold
"""

from __future__ import annotations

import io
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "Retail Demand Prediction & Forecasting"
MODEL_FILE = "retail_demand_model.joblib"

DATE_CANDIDATES = [
    "date", "datetime", "timestamp", "day", "ds",
    "Date", "Datetime", "DATE",
]

TARGET_CANDIDATES = [
    "sales", "demand", "quantity", "qty", "units",
    "units_sold", "sales_qty", "revenue",
    "Sales", "Demand", "Quantity", "Units_Sold",
]

GROUP_CANDIDATES = [
    "store_id", "store", "branch", "shop",
    "item_id", "item", "product_id", "product",
    "sku", "category", "department",
]


@dataclass
class ModelConfig:
    date_col: str
    target_col: str
    group_cols: list[str]
    horizon: int = 30
    validation_days: int = 30
    test_days: int = 30
    lags: tuple[int, ...] = (1, 7, 14, 28)
    rolling_windows: tuple[int, ...] = (7, 14, 28)
    random_state: int = 42


# ============================================================
# DATA UTILITIES
# ============================================================

def detect_column(columns, candidates):
    lookup = {str(c).lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def load_data(
    uploaded_file,
    date_col: str,
    target_col: str,
    group_cols: list[str],
) -> pd.DataFrame:

    df = pd.read_csv(uploaded_file)

    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    missing = [
        col for col in [date_col, target_col, *group_cols]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    df[target_col] = pd.to_numeric(
        df[target_col],
        errors="coerce",
    )

    df = df.dropna(
        subset=[date_col, target_col],
    )

    df[target_col] = df[target_col].clip(lower=0)

    for col in group_cols:
        df[col] = (
            df[col]
            .fillna("UNKNOWN")
            .astype(str)
        )

    # Aggregate duplicate date/group rows.
    keys = [*group_cols, date_col]

    df = (
        df.groupby(
            keys,
            as_index=False,
            dropna=False,
        )[target_col]
        .sum()
    )

    df = df.sort_values(
        [*group_cols, date_col]
        if group_cols
        else [date_col]
    )

    return df.reset_index(drop=True)


def regularize_daily(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Convert each series into a daily time grid.

    Missing days are interpreted as zero demand.
    Change fill behavior here if your business treats
    missing rows as unknown demand.
    """

    if not group_cols:
        result = (
            df.set_index(date_col)
            .sort_index()
            .asfreq("D")
        )

        result[target_col] = (
            result[target_col]
            .fillna(0)
        )

        return result.reset_index()

    output = []

    for keys, group in df.groupby(
        group_cols,
        dropna=False,
        sort=False,
    ):

        if not isinstance(keys, tuple):
            keys = (keys,)

        group = (
            group
            .sort_values(date_col)
            .set_index(date_col)
        )

        series = (
            group[[target_col]]
            .asfreq("D")
        )

        series[target_col] = (
            series[target_col]
            .fillna(0)
        )

        for col, value in zip(
            group_cols,
            keys,
        ):
            series[col] = value

        output.append(
            series.reset_index()
        )

    return (
        pd.concat(
            output,
            ignore_index=True,
        )
        .sort_values(
            [*group_cols, date_col]
        )
        .reset_index(drop=True)
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(
    df: pd.DataFrame,
    config: ModelConfig,
    target_available: bool = True,
):
    df = df.copy()

    date_col = config.date_col
    target_col = config.target_col
    group_cols = config.group_cols

    sort_cols = (
        [*group_cols, date_col]
        if group_cols
        else [date_col]
    )

    df = df.sort_values(sort_cols)

    date = df[date_col]

    # Calendar features.
    df["year"] = date.dt.year
    df["month"] = date.dt.month
    df["day"] = date.dt.day
    df["day_of_week"] = date.dt.dayofweek
    df["day_of_year"] = date.dt.dayofyear
    df["week_of_year"] = (
        date.dt.isocalendar()
        .week
        .astype(int)
    )
    df["quarter"] = date.dt.quarter

    df["is_weekend"] = (
        date.dt.dayofweek >= 5
    ).astype(int)

    df["is_month_start"] = (
        date.dt.is_month_start
    ).astype(int)

    df["is_month_end"] = (
        date.dt.is_month_end
    ).astype(int)

    # Cyclic calendar encoding.
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )
    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # Time since first observation.
    df["days_since_start"] = (
        date - date.min()
    ).dt.days

    if target_available:

        if group_cols:
            grouped = df.groupby(
                group_cols,
                sort=False,
                dropna=False,
            )[target_col]

            # Lag features.
            for lag in config.lags:
                df[f"lag_{lag}"] = (
                    grouped.shift(lag)
                )

            # Rolling features.
            for window in config.rolling_windows:

                shifted = grouped.shift(1)

                df[f"rolling_mean_{window}"] = (
                    shifted
                    .groupby(
                        [
                            df[c]
                            for c in group_cols
                        ],
                        sort=False,
                    )
                    .rolling(
                        window,
                        min_periods=2,
                    )
                    .mean()
                    .reset_index(
                        level=list(
                            range(len(group_cols))
                        ),
                        drop=True,
                    )
                    .sort_index()
                )

                df[f"rolling_std_{window}"] = (
                    shifted
                    .groupby(
                        [
                            df[c]
                            for c in group_cols
                        ],
                        sort=False,
                    )
                    .rolling(
                        window,
                        min_periods=2,
                    )
                    .std()
                    .reset_index(
                        level=list(
                            range(len(group_cols))
                        ),
                        drop=True,
                    )
                    .sort_index()
                )

                df[f"rolling_min_{window}"] = (
                    shifted
                    .groupby(
                        [
                            df[c]
                            for c in group_cols
                        ],
                        sort=False,
                    )
                    .rolling(
                        window,
                        min_periods=2,
                    )
                    .min()
                    .reset_index(
                        level=list(
                            range(len(group_cols))
                        ),
                        drop=True,
                    )
                    .sort_index()
                )

                df[f"rolling_max_{window}"] = (
                    shifted
                    .groupby(
                        [
                            df[c]
                            for c in group_cols
                        ],
                        sort=False,
                    )
                    .rolling(
                        window,
                        min_periods=2,
                    )
                    .max()
                    .reset_index(
                        level=list(
                            range(len(group_cols))
                        ),
                        drop=True,
                    )
                    .sort_index()
                )

        else:

            shifted = (
                df[target_col]
                .shift(1)
            )

            for lag in config.lags:
                df[f"lag_{lag}"] = (
                    df[target_col]
                    .shift(lag)
                )

            for window in config.rolling_windows:

                df[f"rolling_mean_{window}"] = (
                    shifted
                    .rolling(
                        window,
                        min_periods=2,
                    )
                    .mean()
                )

                df[f"rolling_std_{window}"] = (
                    shifted
                    .rolling(
                        window,
                        min_periods=2,
                    )
                    .std()
                )

                df[f"rolling_min_{window}"] = (
                    shifted
                    .rolling(
                        window,
                        min_periods=2,
                    )
                    .min()
                )

                df[f"rolling_max_{window}"] = (
                    shifted
                    .rolling(
                        window,
                        min_periods=2,
                    )
                    .max()
                )

    # Replace infinities.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    excluded = {
        date_col,
        target_col,
    }

    feature_cols = [
        c for c in df.columns
        if c not in excluded
    ]

    return df, feature_cols


def make_training_data(
    df: pd.DataFrame,
    config: ModelConfig,
):
    featured, feature_cols = add_features(
        df,
        config,
        target_available=True,
    )

    lag_cols = [
        c
        for c in feature_cols
        if c.startswith(
            ("lag_", "rolling_")
        )
    ]

    featured = featured.dropna(
        subset=lag_cols
    )

    X = featured[
        feature_cols
    ].copy()

    y = featured[
        config.target_col
    ].astype(float)

    return (
        featured,
        X,
        y,
        feature_cols,
    )


# ============================================================
# MODEL
# ============================================================

def build_model(X: pd.DataFrame):
    categorical_cols = (
        X
        .select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    numeric_cols = [
        c for c in X.columns
        if c not in categorical_cols
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        )
                    ]
                ),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    regressor = (
        HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            l2_regularization=1.0,
            loss="squared_error",
            random_state=42,
        )
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                regressor,
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    non_zero = (
        y_true != 0
    )

    if non_zero.any():
        mape = (
            np.mean(
                np.abs(
                    (
                        y_true[non_zero]
                        - y_pred[non_zero]
                    )
                    / y_true[non_zero]
                )
            )
            * 100
        )
    else:
        mape = 0.0

    denominator = (
        np.abs(y_true)
        + np.abs(y_pred)
    )

    smape_values = np.where(
        denominator == 0,
        0,
        (
            2
            * np.abs(
                y_pred - y_true
            )
            / denominator
        ),
    )

    smape = (
        np.mean(smape_values)
        * 100
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "sMAPE": float(smape),
        "R2": float(r2),
    }


# ============================================================
# TIME SPLIT
# ============================================================

def chronological_split(
    featured,
    date_col,
    validation_days,
    test_days,
):
    latest = (
        featured[date_col]
        .max()
        .normalize()
    )

    test_start = (
        latest
        - pd.Timedelta(
            days=test_days - 1
        )
    )

    validation_start = (
        test_start
        - pd.Timedelta(
            days=validation_days
        )
    )

    train_mask = (
        featured[date_col]
        < validation_start
    )

    validation_mask = (
        (featured[date_col] >= validation_start)
        & (featured[date_col] < test_start)
    )

    test_mask = (
        featured[date_col]
        >= test_start
    )

    return (
        train_mask,
        validation_mask,
        test_mask,
    )


# ============================================================
# FORECAST
# ============================================================

def recursive_forecast(
    history,
    model,
    config: ModelConfig,
):
    """
    Multi-step recursive forecast.

    Every predicted day is appended to the history, so the
    next day's lag features can use previous predictions.
    """

    history = history.copy()

    date_col = config.date_col
    target_col = config.target_col
    group_cols = config.group_cols

    history[date_col] = pd.to_datetime(
        history[date_col]
    )

    history = history.sort_values(
        [*group_cols, date_col]
        if group_cols
        else [date_col]
    )

    if group_cols:
        group_values = (
            history[
                group_cols
            ]
            .drop_duplicates()
            .to_dict("records")
        )
    else:
        group_values = [{}]

    predictions = []

    for _ in range(
        config.horizon
    ):

        next_date = (
            history[date_col]
            .max()
            + pd.Timedelta(days=1)
        )

        future_rows = []

        for group in group_values:

            row = {
                date_col: next_date,
                target_col: np.nan,
                **group,
            }

            future_rows.append(row)

        future = pd.DataFrame(
            future_rows
        )

        combined = pd.concat(
            [
                history,
                future,
            ],
            ignore_index=True,
        )

        featured, feature_cols = (
            add_features(
                combined,
                config,
                target_available=True,
            )
        )

        current = featured[
            featured[date_col]
            == next_date
        ].copy()

        X_future = current[
            feature_cols
        ]

        prediction = model.predict(
            X_future
        )

        prediction = np.clip(
            prediction,
            0,
            None,
        )

        future[target_col] = (
            prediction
        )

        predictions.append(
            future[
                [
                    date_col,
                    *group_cols,
                ]
            ].assign(
                predicted_demand=prediction
            )
        )

        history = pd.concat(
            [
                history,
                future,
            ],
            ignore_index=True,
        )

    return pd.concat(
        predictions,
        ignore_index=True,
    )


# ============================================================
# MODEL SAVE / LOAD
# ============================================================

def save_model(
    model,
    config,
    feature_cols,
    metrics,
    path=MODEL_FILE,
):
    bundle = {
        "model": model,
        "config": config,
        "feature_cols": feature_cols,
        "metrics": metrics,
    }

    joblib.dump(
        bundle,
        path,
    )


def load_saved_model(
    path=MODEL_FILE,
):
    return joblib.load(path)


# ============================================================
# STREAMLIT APPLICATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
)

st.title("📈 Retail Demand Prediction & Forecasting")
st.write(
    "Upload your retail sales dataset, train the ML model, "
    "evaluate it chronologically, and generate future demand."
)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header(
    "1. Upload Dataset"
)

uploaded_file = (
    st.sidebar.file_uploader(
        "Retail CSV",
        type=["csv"],
    )
)

if uploaded_file is None:
    st.info(
        "Upload your CSV from the sidebar to start."
    )

    st.markdown(
        """
### Expected dataset

Your CSV should contain at least:

- **Date column:** `date`, `Date`, `datetime`, etc.
- **Demand column:** `sales`, `quantity`, `units_sold`, etc.

Optional:

- Store
- Product
- SKU
- Category
- Department

Example:

```text
date,store_id,item_id,sales
2025-01-01,S01,P001,120
2025-01-02,S01,P001,132
2025-01-03,S01,P001,118
```
"""
    )

    st.stop()


raw = pd.read_csv(
    uploaded_file
)

if raw.empty:
    st.error(
        "The uploaded CSV is empty."
    )
    st.stop()

# ------------------------------------------------------------
# COLUMN DETECTION
# ------------------------------------------------------------

detected_date = detect_column(
    raw.columns,
    DATE_CANDIDATES,
)

detected_target = detect_column(
    raw.columns,
    TARGET_CANDIDATES,
)

st.sidebar.header(
    "2. Dataset Configuration"
)

date_default_index = (
    list(raw.columns).index(
        detected_date
    )
    if detected_date in raw.columns
    else 0
)

date_col = st.sidebar.selectbox(
    "Date column",
    raw.columns,
    index=date_default_index,
)

target_default_index = (
    list(raw.columns).index(
        detected_target
    )
    if detected_target in raw.columns
    else 0
)

target_col = st.sidebar.selectbox(
    "Demand / sales column",
    raw.columns,
    index=target_default_index,
)

possible_groups = [
    c for c in raw.columns
    if c not in {
        date_col,
        target_col,
    }
]

detected_groups = [
    c for c in GROUP_CANDIDATES
    if c in possible_groups
]

group_cols = st.sidebar.multiselect(
    "Store / Product columns",
    possible_groups,
    default=detected_groups,
)

horizon = st.sidebar.slider(
    "Forecast horizon",
    min_value=7,
    max_value=180,
    value=30,
)

validation_days = st.sidebar.slider(
    "Validation days",
    min_value=7,
    max_value=90,
    value=30,
)

test_days = st.sidebar.slider(
    "Test days",
    min_value=7,
    max_value=90,
    value=30,
)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

try:
    df = load_data(
        uploaded_file,
        date_col,
        target_col,
        group_cols,
    )

    df = regularize_daily(
        df,
        date_col,
        target_col,
        group_cols,
    )

except Exception as exc:
    st.error(
        f"Data loading error: {exc}"
    )
    st.stop()

# ------------------------------------------------------------
# DATASET SUMMARY
# ------------------------------------------------------------

st.subheader(
    "Dataset Overview"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Rows",
    f"{len(df):,}",
)

c2.metric(
    "Date range",
    f"{df[date_col].min().date()} → {df[date_col].max().date()}",
)

c3.metric(
    "Total demand",
    f"{df[target_col].sum():,.0f}",
)

c4.metric(
    "Average daily demand",
    f"{df[target_col].mean():,.2f}",
)

# ------------------------------------------------------------
# HISTORICAL CHART
# ------------------------------------------------------------

st.subheader(
    "Historical Demand"
)

daily = (
    df.groupby(
        date_col,
        as_index=True,
    )[target_col]
    .sum()
)

st.line_chart(
    daily
)

# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------

st.subheader(
    "ML Model"
)

if st.button(
    "🚀 Train Demand Forecasting Model",
    type="primary",
):

    config = ModelConfig(
        date_col=date_col,
        target_col=target_col,
        group_cols=group_cols,
        horizon=horizon,
        validation_days=validation_days,
        test_days=test_days,
    )

    with st.spinner(
        "Engineering features and training model..."
    ):

        featured, X, y, feature_cols = (
            make_training_data(
                df,
                config,
            )
        )

        if len(featured) < 100:
            st.error(
                "Not enough historical rows after feature engineering."
            )
            st.stop()

        (
            train_mask,
            validation_mask,
            test_mask,
        ) = chronological_split(
            featured,
            date_col,
            validation_days,
            test_days,
        )

        X_train = X.loc[
            train_mask
        ]
        y_train = y.loc[
            train_mask
        ]

        X_validation = X.loc[
            validation_mask
        ]
        y_validation = y.loc[
            validation_mask
        ]

        X_test = X.loc[
            test_mask
        ]
        y_test = y.loc[
            test_mask
        ]

        if min(
            len(X_train),
            len(X_validation),
            len(X_test),
        ) == 0:
            st.error(
                "Train/validation/test split is empty. "
                "Use a longer dataset or reduce validation/test days."
            )
            st.stop()

        model = build_model(
            X_train
        )

        model.fit(
            X_train,
            y_train,
        )

        validation_pred = np.clip(
            model.predict(
                X_validation
            ),
            0,
            None,
        )

        test_pred = np.clip(
            model.predict(
                X_test
            ),
            0,
            None,
        )

        validation_metrics = (
            calculate_metrics(
                y_validation,
                validation_pred,
            )
        )

        test_metrics = (
            calculate_metrics(
                y_test,
                test_pred,
            )
        )

        save_model(
            model,
            config,
            feature_cols,
            {
                "validation": validation_metrics,
                "test": test_metrics,
            },
        )

        test_results = featured.loc[
            test_mask,
            [
                date_col,
                *group_cols,
                target_col,
            ],
        ].copy()

        test_results[
            "prediction"
        ] = test_pred

    st.success(
        "Model trained and saved successfully."
    )

    st.session_state[
        "model_bundle"
    ] = {
        "model": model,
        "config": config,
        "feature_cols": feature_cols,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_results": test_results,
    }

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

if (
    "model_bundle"
    in st.session_state
):

    bundle = st.session_state[
        "model_bundle"
    ]

    st.subheader(
        "Model Evaluation"
    )

    st.caption(
        "Evaluation uses a chronological holdout instead of random splitting."
    )

    metrics = bundle[
        "test_metrics"
    ]

    a, b, c, d, e = st.columns(5)

    a.metric(
        "MAE",
        f"{metrics['MAE']:.2f}",
    )

    b.metric(
        "RMSE",
        f"{metrics['RMSE']:.2f}",
    )

    c.metric(
        "MAPE",
        f"{metrics['MAPE']:.2f}%",
    )

    d.metric(
        "sMAPE",
        f"{metrics['sMAPE']:.2f}%",
    )

    e.metric(
        "R²",
        f"{metrics['R2']:.3f}",
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    st.subheader(
        "Actual vs Predicted Demand"
    )

    results = bundle[
        "test_results"
    ]

    chart = (
        results.groupby(
            date_col,
            as_index=True,
        )[
            [target_col, "prediction"]
        ]
        .sum()
    )

    chart.columns = [
        "Actual",
        "Predicted",
    ]

    st.line_chart(
        chart
    )

    st.dataframe(
        results.tail(100),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "Feature Importance"
    )

    try:
        importance_model = (
            bundle["model"]
            .named_steps["model"]
        )

        preprocessor = (
            bundle["model"]
            .named_steps["preprocessor"]
        )

        # HistGradientBoosting has feature_importances_
        # only in some sklearn versions. If unavailable,
        # show the original feature list instead.
        if hasattr(
            importance_model,
            "feature_importances_",
        ):
            transformed_names = (
                preprocessor
                .get_feature_names_out()
            )

            values = (
                importance_model
                .feature_importances_
            )

            importance = (
                pd.DataFrame(
                    {
                        "feature": transformed_names,
                        "importance": values,
                    }
                )
                .sort_values(
                    "importance",
                    ascending=False,
                )
                .head(25)
            )

            st.bar_chart(
                importance.set_index(
                    "feature"
                )
            )

        else:
            st.info(
                "This scikit-learn estimator does not expose native feature_importances_. "
                "The model still trains and predicts normally."
            )

    except Exception as exc:
        st.info(
            f"Feature importance visualization unavailable: {exc}"
        )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    st.subheader(
        "Future Demand Forecast"
    )

    forecast_horizon = st.slider(
        "Future forecast days",
        7,
        180,
        horizon,
        key="forecast_horizon",
    )

    if st.button(
        "🔮 Generate Future Forecast"
    ):

        forecast_config = (
            bundle["config"]
        )

        forecast_config.horizon = (
            forecast_horizon
        )

        with st.spinner(
            "Generating recursive future forecast..."
        ):

            forecast = (
                recursive_forecast(
                    df,
                    bundle["model"],
                    forecast_config,
                )
            )

        st.session_state[
            "forecast"
        ] = forecast

    if (
        "forecast"
        in st.session_state
    ):

        forecast = st.session_state[
            "forecast"
        ]

        total_forecast = (
            forecast[
                "predicted_demand"
            ]
            .sum()
        )

        average_forecast = (
            forecast[
                "predicted_demand"
            ]
            .mean()
        )

        x1, x2, x3 = st.columns(3)

        x1.metric(
            "Forecasted demand",
            f"{total_forecast:,.0f}",
        )

        x2.metric(
            "Average / day",
            f"{average_forecast:,.2f}",
        )

        x3.metric(
            "Forecast days",
            f"{len(forecast[date_col].unique())}",
        )

        forecast_chart = (
            forecast.groupby(
                date_col,
                as_index=True,
            )[
                "predicted_demand"
            ]
            .sum()
        )

        st.line_chart(
            forecast_chart
        )

        st.dataframe(
            forecast,
            use_container_width=True,
        )

        # CSV download.
        csv_bytes = (
            forecast.to_csv(
                index=False
            ).encode("utf-8")
        )

        st.download_button(
            "⬇️ Download Forecast CSV",
            data=csv_bytes,
            file_name="retail_demand_forecast.csv",
            mime="text/csv",
        )

    # --------------------------------------------------------
    # MODEL DOWNLOAD
    # --------------------------------------------------------

    model_bytes = Path(
        MODEL_FILE
    ).read_bytes()

    st.download_button(
        "⬇️ Download Trained Model",
        data=model_bytes,
        file_name=MODEL_FILE,
        mime="application/octet-stream",
    )

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "Model: HistGradientBoostingRegressor | "
    "Features: calendar + lag + rolling demand | "
    "Evaluation: chronological holdout | "
    "Forecasting: recursive multi-step"
)
