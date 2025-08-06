# -------------------------------------------------------------
# 📊 DASHBOARD FINANCIERO AVANZADO – VERSIÓN MEJORADA
# -------------------------------------------------------------
# • WACC:  Ke dinámico (CAPM editable) + Kd real extraído de Interest Expense
#          con tasa impositiva efectiva por empresa.
# • ROIC:  NOPAT / (Equity + Deuda – Efectivo), sin anularse cuando la deuda = 0.
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns  # (no se usa en los gráficos finales, pero lo mantenemos por compatibilidad)
import time

# -------------------------------------------------------------
# ⚙️ Configuración global de la página
# -------------------------------------------------------------
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Parámetros WACC por defecto (se pueden ajustar en sidebar)
# -------------------------------------------------------------
Rf = 0.0435  # Tasa libre de riesgo
Rm = 0.085   # Retorno esperado del mercado
Tc = 0.21    # Tasa impositiva corporativa (solo como valor por defecto)

# -------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------
def obtener_kd(bs: dict, fin: dict):
    """Devuelve (kd, total_debt). Si no hay deuda retorna (0, 0)."""
    total_debt = bs.get("Total Debt", 0)
    interest_expense = abs(fin.get("Interest Expense", 0))
    kd = interest_expense / total_debt if total_debt else 0
    return kd, total_debt

def tasa_impuestos_efectiva(fin: dict, default: float = 0.21):
    ebt = fin.get("Ebt", None)
    tax = fin.get("Income Tax Expense", None)
    if ebt and tax is not None and ebt != 0:
        return tax / ebt
    return default

def calcular_wacc(info: dict, bs: dict, fin: dict):
    """
    WACC con Kd dinámico, Ke vía CAPM y tasa impositiva efectiva.
    Devuelve (wacc, total_debt, kd, tasa_impuestos).
    """
    beta = info.get("beta", 1.0)
    price = info.get("currentPrice")
    shares = info.get("sharesOutstanding")
    market_cap = (price or 0) * (shares or 0)

    kd, total_debt = obtener_kd(bs, fin)
    t = tasa_impuestos_efectiva(fin)

    ke = Rf + beta * (Rm - Rf)        # CAPM
    total_capital = market_cap + total_debt
    if total_capital == 0:
        return None, total_debt, kd, t

    wacc = ((market_cap / total_capital) * ke +
            (total_debt / total_capital) * kd * (1 - t))
    return wacc, total_debt, kd, t

def calcular_crecimiento_historico(financials_df: pd.DataFrame, metric: str):
    """
    CAGR de los últimos 4 periodos (o los que haya) de un metric dado.
    """
    try:
        if metric not in financials_df.index:
            return None
        datos = financials_df.loc[metric].dropna().iloc[:4]
        if len(datos) < 2:
            return None
        primer_valor, ultimo_valor = datos.iloc[-1], datos.iloc[0]
        años = len(datos) - 1
        if primer_valor == 0:
            return None
        cagr = (ultimo_valor / primer_valor) ** (1 / años) - 1
        return cagr
    except Exception:
        return None

# -------------------------------------------------------------
# Extracción y cálculo de métricas para cada empresa
# -------------------------------------------------------------
def obtener_datos_financieros(ticker: str):
    try:
        stock = yf.Ticker(ticker)

        # --- DataFrames originales -------------------------------------------------
        bs_df  = stock.balance_sheet
        fin_df = stock.financials
        cf_df  = stock.cashflow

        # Si Yahoo no trae datos, devolvemos error para este ticker
        if bs_df.empty and fin_df.empty:
            return {"Ticker": ticker, "Error": "Información financiera no disponible"}

        # --- Dicts "flat" (último periodo) -----------------------------------------
        bs  = bs_df.fillna(0).iloc[:, 0].to_dict()   if not bs_df.empty  else {}
        fin = fin_df.fillna(0).iloc[:, 0].to_dict()  if not fin_df.empty else {}
        cf  = cf_df.fillna(0).iloc[:, 0].to_dict()   if not cf_df.empty  else {}

        # --- Datos básicos ---------------------------------------------------------
        info = stock.info
        price   = info.get("currentPrice")
        name    = info.get("longName", ticker)
        sector  = info.get("sector", "N/D")
        country = info.get("country", "N/D")
        industry= info.get("industry", "N/D")

        # --- Ratios de valoración --------------------------------------------------
        pe             = info.get("trailingPE")
        pb             = info.get("priceToBook")
        dividend       = info.get("dividendRate")
        dividend_yield = info.get("dividendYield")
        payout_ratio   = info.get("payoutRatio")

        # --- Ratios de rentabilidad / liquidez / deuda ----------------------------
        roa = info.get("returnOnAssets")
        roe = info.get("returnOnEquity")

        current_ratio = info.get("currentRatio")
        quick_ratio   = info.get("quickRatio")

        ltde = info.get("longTermDebtToEquity")
        de   = info.get("debtToEquity")

        op_margin     = info.get("operatingMargins")
        profit_margin = info.get("profitMargins")

        # --- Flujo de caja  ---------------------------------------------------------
        fcf    = cf.get("Free Cash Flow")
        shares = info.get("sharesOutstanding")
        pfcf   = price / (fcf / shares) if fcf and shares else None

        # --- Cálculos avanzados: WACC, ROIC, EVA -----------------------------------
        wacc, total_debt, kd, tax_rate = calcular_wacc(info, bs, fin)

        ebit   = fin.get("EBIT", fin.get("Operating Income"))
        equity = bs.get("Total Stockholder Equity",
                        bs.get("Common Stock Equity"))

        cash   = bs.get("Cash And Cash Equivalents", 0)
        capital_invertido = None
        if equity is not None and total_debt is not None:
            capital_invertido = equity + (total_debt - cash)
            if capital_invertido == 0:
                capital_invertido = None  # evita división por cero

        roic = None
        nopat = None
        if ebit is not None and capital_invertido:
            nopat = ebit * (1 - tax_rate)
            roic = nopat / capital_invertido

        eva = None
        if roic is not None and wacc is not None and capital_invertido:
            eva = (roic - wacc) * capital_invertido

        # --- Crecimientos -----------------------------------------------------------
        revenue_growth = calcular_crecimiento_historico(fin_df, "Total Revenue")
        eps_growth     = calcular_crecimiento_historico(fin_df, "Net Income")
        fcf_growth     = (calcular_crecimiento_historico(cf_df, "Free Cash Flow")
                          or calcular_crecimiento_historico(cf_df, "Operating Cash Flow"))

        # --- Liquidez avanzada ------------------------------------------------------
        cash_ratio          = info.get("cashRatio")
        operating_cash_flow = cf.get("Operating Cash Flow")
        current_liabilities = bs.get("Total Current Liabilities")
        cash_flow_ratio     = (operating_cash_flow / current_liabilities
                               if operating_cash_flow and current_liabilities else None)

        # --- Pausa para no saturar la API ------------------------------------------
        time.sleep(0.8)

        # --- Resultado --------------------------------------------------------------
        return {
            "Ticker": ticker,
            "Nombre": name,
            "Sector": sector,
            "País": country,
            "Industria": industry,
            "Precio": price,
            "P/E": pe,
            "P/B": pb,
            "P/FCF": pfcf,
            "Dividend Year": dividend,
            "Dividend Yield %": dividend_yield,
            "Payout Ratio": payout_ratio,
            "ROA": roa,
            "ROE": roe,
            "Current Ratio": current_ratio,
            "Quick Ratio": quick_ratio,
            "LtDebt/Eq": ltde,
            "Debt/Eq": de,
            "Oper Margin": op_margin,
            "Profit Margin": profit_margin,
            "WACC": wacc,
            "ROIC": roic,
            "EVA": eva,
            "Kd": kd,
            "Tax Rate": tax_rate,
            "Deuda Total": total_debt,
            "Patrimonio Neto": equity,
            "Revenue Growth": revenue_growth,
            "EPS Growth": eps_growth,
            "FCF Growth": fcf_growth,
            "Cash Ratio": cash_ratio,
            "Cash Flow Ratio": cash_flow_ratio,
            "Operating Cash Flow": operating_cash_flow,
            "Current Liabilities": current_liabilities,
        }

    except Exception as e:
        return {"Ticker": ticker, "Error": str(e)}

# -------------------------------------------------------------
# INTERFAZ DE USUARIO (Streamlit)
# -------------------------------------------------------------
def main():
    st.title("📊 Dashboard de Análisis Financiero Avanzado")

    # -------- Sidebar ------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuración")
        tickers_input = st.text_area(
            "🔎 Ingresa tickers (separados por coma)",
            "AAPL, MSFT, GOOGL, AMZN, TSLA",
            help="Ejemplo: AAPL, MSFT, GOOG",
        )
        max_tickers = st.slider("Número máximo de tickers", 1, 100, 50)

        st.markdown("---")
        st.markdown("**Parámetros WACC (CAPM)**")

        global Rf, Rm, Tc
        Rf = st.number_input(
            "Tasa libre de riesgo (%)", min_value=0.0, max_value=20.0, value=4.35
        ) / 100
        Rm = st.number_input(
            "Retorno esperado del mercado (%)", min_value=0.0, max_value=30.0, value=8.5
        ) / 100
        Tc = st.number_input(
            "Tasa impositiva corporativa (%)", min_value=0.0, max_value=50.0, value=21.0
        ) / 100

    # -------- Procesamiento de tickers ------------------------------------------
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][
        : max_tickers
    ]

    if st.button("🔍 Analizar Acciones", type="primary"):
        if not tickers:
            st.warning("Por favor ingresa al menos un ticker")
            return

        resultados = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Procesamos los tickers en lotes de 10 para ser suaves con la API
        batch_size = 10
        for batch_start in range(0, len(tickers), batch_size):
            batch_end = min(batch_start + batch_size, len(tickers))
            batch_tickers = tickers[batch_start:batch_end]

            for i, t in enumerate(batch_tickers):
                status_text.text(
                    f"⏳ Procesando {t} ({batch_start + i + 1}/{len(tickers)})..."
                )
                resultados[t] = obtener_datos_financieros(t)
                progress_bar.progress((batch_start + i + 1) / len(tickers))

        status_text.text("✅ Análisis completado!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

        # -------- Mostrar resultados -------------------------------------------
        datos = list(resultados.values())
        datos_validos = [d for d in datos if "Error" not in d]
        if not datos_validos:
            st.error("No se pudo obtener datos válidos para ningún ticker")
            return

        df = pd.DataFrame(datos_validos)

        # -------- Sección 1: Resumen General -----------------------------------
        st.header("📋 Resumen General")

        porcentajes = [
            "Dividend Yield %",
            "Payout Ratio",
            "ROA",
            "ROE",
            "Oper Margin",
            "Profit Margin",
            "WACC",
            "ROIC",
        ]
        for col in porcentajes:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"{x:.2%}" if pd.notnull(x) else "N/D"
                )

        columnas_mostrar = [
            "Ticker",
            "Nombre",
            "Sector",
            "Precio",
            "P/E",
            "P/B",
            "P/FCF",
            "Dividend Yield %",
            "Payout Ratio",
            "ROA",
            "ROE",
            "Current Ratio",
            "Debt/Eq",
            "Oper Margin",
            "Profit Margin",
            "WACC",
            "ROIC",
            "EVA",
        ]

        st.dataframe(
            df[columnas_mostrar].dropna(how="all", axis=1),
            use_container_width=True,
            height=400,
        )

        # -------- Sección 2: Análisis de Valoración ----------------------------
        st.header("💰 Análisis de Valoración")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ratios de Valoración")
            fig, ax = plt.subplots(figsize=(10, 4))
            df_plot = (
                df[["Ticker", "P/E", "P/B", "P/FCF"]]
                .set_index("Ticker")
                .apply(pd.to_numeric, errors="coerce")
            )
            df_plot.plot(kind="bar", ax=ax, rot=45)
            ax.set_title("Comparativa de Ratios de Valoración")
            ax.set_ylabel("Ratio")
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Dividendos")
            fig, ax = plt.subplots(figsize=(10, 4))
            df_div = df[["Ticker", "Dividend Yield %"]].set_index("Ticker")
            df_div["Dividend Yield %"] = df_div["Dividend Yield %"].replace("N/D", 0)
            df_div["Dividend Yield %"] = (
                df_div["Dividend Yield %"].str.rstrip("%").astype("float")
            )
            df_div.plot(kind="bar", ax=ax, rot=45, color="green")
            ax.set_title("Rendimiento de Dividendos (%)")
            ax.set_ylabel("Dividend Yield %")
            st.pyplot(fig)
            plt.close()

        # -------- Sección 3: Rentabilidad y Eficiencia -------------------------
        st.header("📈 Rentabilidad y Eficiencia")
        tabs = st.tabs(["ROE vs ROA", "Márgenes", "WACC vs ROIC"])

        with tabs[0]:
            fig, ax = plt.subplots(figsize=(10, 5))
            df_plot = df[["Ticker", "ROE", "ROA"]].set_index("Ticker")
            df_plot["ROE"] = df_plot["ROE"].str.rstrip("%").astype("float")
            df_plot["ROA"] = df_plot["ROA"].str.rstrip("%").astype("float")
            df_plot.plot(kind="bar", ax=ax, rot=45)
            ax.set_title("ROE vs ROA (%)")
            ax.set_ylabel("Porcentaje")
            st.pyplot(fig)
            plt.close()

        with tabs[1]:
            fig, ax = plt.subplots(figsize=(10, 5))
            df_plot = df[["Ticker", "Oper Margin", "Profit Margin"]].set_index("Ticker")
            df_plot["Oper Margin"] = df_plot["Oper Margin"].str.rstrip("%").astype("float")
            df_plot["Profit Margin"] = (
                df_plot["Profit Margin"].str.rstrip("%").astype("float")
            )
            df_plot.plot(kind="bar", ax=ax, rot=45)
            ax.set_title("Margen Operativo vs Margen Neto (%)")
            ax.set_ylabel("Porcentaje")
            st.pyplot(fig)
            plt.close()

        with tabs[2]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for _, row in df.iterrows():
                wacc = (
                    float(row["WACC"].rstrip("%"))
                    if pd.notnull(row["WACC"]) and row["WACC"] != "N/D"
                    else None
                )
                roic = (
                    float(row["ROIC"].rstrip("%"))
                    if pd.notnull(row["ROIC"]) and row["ROIC"] != "N/D"
                    else None
                )
                if wacc is not None and roic is not None:
                    color = "green" if roic > wacc else "red"
                    ax.bar(row["Ticker"], roic, color=color, alpha=0.6, label="ROIC")
                    ax.bar(row["Ticker"], wacc, color="gray", alpha=0.3, label="WACC")

            ax.set_title("Creación de Valor: ROIC vs WACC (%)")
            ax.set_ylabel("Porcentaje")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        # -------- Sección 4: Análisis de Deuda ----------------------------------
        st.header("🏦 Estructura de Capital y Deuda")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Apalancamiento")
            fig, ax = plt.subplots(figsize=(10, 5))
            df_plot = (
                df[["Ticker", "Debt/Eq", "LtDebt/Eq"]]
                .set_index("Ticker")
                .apply(pd.to_numeric, errors="coerce")
            )
            df_plot.plot(kind="bar", stacked=True, ax=ax, rot=45)
            ax.axhline(1, color="red", linestyle="--")
            ax.set_title("Deuda/Patrimonio")
            ax.set_ylabel("Ratio")
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Liquidez")
            fig, ax = plt.subplots(figsize=(10, 5))
            df_plot = (
                df[["Ticker", "Current Ratio", "Quick Ratio", "Cash Ratio"]]
                .set_index("Ticker")
                .apply(pd.to_numeric, errors="coerce")
            )
            df_plot.plot(kind="bar", ax=ax, rot=45)
            ax.axhline(1, color="green", linestyle="--")
            ax.set_title("Ratios de Liquidez")
            ax.set_ylabel("Ratio")
            st.pyplot(fig)
            plt.close()

        # -------- Sección 5: Crecimiento ----------------------------------------
        st.header("🚀 Crecimiento Histórico")
        growth_metrics = ["Revenue Growth", "EPS Growth", "FCF Growth"]
        df_growth = df[["Ticker"] + growth_metrics].set_index("Ticker") * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        df_growth.plot(kind="bar", ax=ax, rot=45)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Tasas de Crecimiento Anual (%)")
        ax.set_ylabel("Crecimiento %")
        st.pyplot(fig)
        plt.close()

        # -------- Sección 6: Análisis Individual --------------------------------
        st.header("🔍 Análisis por Empresa")
        selected_ticker = st.selectbox("Selecciona una empresa", df["Ticker"].unique())
        empresa = df[df["Ticker"] == selected_ticker].iloc[0]

        st.subheader(f"Análisis Detallado: {empresa['Nombre']}")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Precio", f"${empresa['Precio']:,.2f}" if empresa["Precio"] else "N/D")
            st.metric("P/E", empresa["P/E"])
            st.metric("P/B", empresa["P/B"])

        with col2:
            st.metric("ROE", empresa["ROE"])
            st.metric("ROIC", empresa["ROIC"])
            st.metric("WACC", empresa["WACC"])

        with col3:
            st.metric("Deuda/Patrimonio", empresa["Debt/Eq"])
            st.metric("Margen Neto", empresa["Profit Margin"])
            st.metric("Dividend Yield", empresa["Dividend Yield %"])

        # --- Gráfico de creación de valor individual ---------------------------
        st.subheader("Creación de Valor")
        fig, ax = plt.subplots(figsize=(6, 4))
        if empresa["ROIC"] != "N/D" and empresa["WACC"] != "N/D":
            roic_val = float(empresa["ROIC"].rstrip("%"))
            wacc_val = float(empresa["WACC"].rstrip("%"))
            color = "green" if roic_val > wacc_val else "red"
            ax.bar(["ROIC", "WACC"], [roic_val, wacc_val], color=[color, "gray"])
            ax.set_title("Creación de Valor (ROIC vs WACC)")
            ax.set_ylabel("%")
            st.pyplot(fig)
            plt.close()

            if roic_val > wacc_val:
                st.success("✅ La empresa está creando valor (ROIC > WACC)")
            else:
                st.error("❌ La empresa está destruyendo valor (ROIC < WACC)")
        else:
            st.warning("Datos insuficientes para análisis ROIC/WACC")

# -------------------------------------------------------------
# Punto de entrada
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
