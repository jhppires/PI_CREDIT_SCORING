"""
Modelo de credit scoring para o case Quantum Finance.

Uso principal:
    python modelo_credit_scoring_quantum.py

Uso como simulador interativo:
    python modelo_credit_scoring_quantum.py --simulate

Dependencias:
    pandas numpy scikit-learn statsmodels matplotlib joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera


BASE_CSV = Path("Base_ScoreCredito_QuantumFinance(8).csv")
OUTPUT_DIR = Path("outputs")
MODEL_PATH = OUTPUT_DIR / "modelo_credit_scoring_linear.joblib"
RESULTS_PATH = OUTPUT_DIR / "resultados_modelo_credit_scoring.json"


def carregar_base(csv_path: Path = BASE_CSV) -> pd.DataFrame:
    """Carrega a base no formato regional do arquivo: ; como separador e , decimal."""
    return pd.read_csv(csv_path, sep=";", decimal=",")


def preparar_dados(df: pd.DataFrame):
    y = df["SCORE_CREDITO"]
    X = df.drop(columns=["SCORE_CREDITO", "id"])
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_cols),
        ],
        verbose_feature_names_out=False,
    )
    return X, y, num_cols, cat_cols, preprocessor


def avaliar_modelos(df: pd.DataFrame) -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)

    X, y, num_cols, cat_cols, preprocessor = preparar_dados(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    modelos = {
        "Regressao Linear": LinearRegression(),
        "Arvore de Decisao": DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42),
    }

    resultados = {
        "n_linhas": int(df.shape[0]),
        "n_colunas": int(df.shape[1]),
        "variaveis_numericas": num_cols,
        "variaveis_categoricas": cat_cols,
        "descritiva_numerica": df.describe().T.round(4).to_dict(orient="index"),
        "categorias": {
            col: {str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()}
            for col in cat_cols
        },
        "correlacao_score": (
            df.select_dtypes(include=np.number)
            .corr(numeric_only=True)["SCORE_CREDITO"]
            .sort_values(ascending=False)
            .round(4)
            .to_dict()
        ),
        "modelos": {},
    }

    for nome, estimador in modelos.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimador)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        cv = cross_val_score(pipe, X, y, cv=5, scoring="r2")
        perm = permutation_importance(pipe, X_test, y_test, n_repeats=20, random_state=42, scoring="r2")
        importancia = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

        resultados["modelos"][nome] = {
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
            "R2": float(r2_score(y_test, pred)),
            "MAPE": float(mean_absolute_percentage_error(y_test, pred)),
            "CV_R2_media": float(cv.mean()),
            "CV_R2_desvio": float(cv.std()),
            "importancia_permutacao": importancia.round(6).to_dict(),
        }

        if nome == "Regressao Linear":
            nomes_features = pipe.named_steps["preprocess"].get_feature_names_out()
            coeficientes = pd.Series(pipe.named_steps["model"].coef_, index=nomes_features)
            resultados["modelos"][nome]["coeficientes"] = (
                coeficientes.sort_values(key=np.abs, ascending=False).round(6).to_dict()
            )
            joblib.dump(pipe, MODEL_PATH)

            Xt = pipe.named_steps["preprocess"].fit_transform(X)
            Xt_const = sm.add_constant(Xt)
            ols = sm.OLS(y, Xt_const).fit()
            resid = ols.resid
            jb_stat, jb_p, _, _ = jarque_bera(resid)
            bp_stat, bp_p, _, _ = het_breuschpagan(resid, ols.model.exog)
            resultados["diagnostico_regressao"] = {
                "R2_ajustado_base_completa": float(ols.rsquared_adj),
                "Durbin_Watson": float(durbin_watson(resid)),
                "Jarque_Bera_pvalor": float(jb_p),
                "Breusch_Pagan_pvalor": float(bp_p),
            }

            vif_features = pd.DataFrame(Xt, columns=nomes_features)
            vif = []
            for i, col in enumerate(vif_features.columns):
                try:
                    vif.append((col, variance_inflation_factor(vif_features.values, i)))
                except Exception:
                    vif.append((col, np.nan))
            resultados["VIF_top"] = (
                pd.Series(dict(vif)).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False).head(12).round(4).to_dict()
            )

        if nome == "Arvore de Decisao":
            feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
            resultados["modelos"][nome]["regras_arvore_resumo"] = export_text(
                pipe.named_steps["model"], feature_names=list(feature_names), max_depth=3
            )

    RESULTS_PATH.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    return resultados


def classificar_risco(score: float) -> str:
    if score < 400:
        return "alto risco"
    if score < 600:
        return "risco medio"
    return "baixo risco"


def simulador_interativo() -> None:
    if not MODEL_PATH.exists():
        print("Modelo ainda nao treinado. Executando treino antes do simulador...")
        avaliar_modelos(carregar_base())

    modelo = joblib.load(MODEL_PATH)
    campos = {
        "idade": int,
        "sexo": str,
        "estado_civil": str,
        "escola": str,
        "Qte_dependentes": int,
        "tempo_ultimoservico": int,
        "trabalha": int,
        "vl_salario_mil": float,
        "reg_moradia": int,
        "casa_propria": int,
        "vl_imovel_em_mil": float,
        "Qte_cartoes": int,
        "Qte_carros": int,
    }
    print("Informe os dados do cliente. Exemplos: sexo=F/M, trabalha=1/0, casa_propria=1/0.")
    registro = {}
    for campo, conversor in campos.items():
        valor = input(f"{campo}: ").strip().replace(",", ".")
        registro[campo] = conversor(valor)
    entrada = pd.DataFrame([registro])
    score = float(modelo.predict(entrada)[0])
    print(f"\nScore estimado: {score:.1f}")
    print(f"Classificacao sugerida: {classificar_risco(score)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true", help="Abre o simulador interativo do score.")
    args = parser.parse_args()

    if args.simulate:
        simulador_interativo()
        return

    df = carregar_base()
    resultados = avaliar_modelos(df)
    print(f"Resultados salvos em: {RESULTS_PATH}")
    print(f"Modelo salvo em: {MODEL_PATH}")
    for nome, metricas in resultados["modelos"].items():
        print(
            f"{nome}: R2={metricas['R2']:.4f}, RMSE={metricas['RMSE']:.2f}, "
            f"MAE={metricas['MAE']:.2f}, MAPE={metricas['MAPE']:.2%}"
        )


if __name__ == "__main__":
    main()
