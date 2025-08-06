import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime, timedelta

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
# Parámetros WACC por defecto (ajustables en el sidebar)
# -------------------------------------------------------------
Rf = 0.0435  # Tasa libre de riesgo
Rm = 0.085   # Retorno esperado del mercado
Tc = 0.21    # Tasa impositiva corporativa

# -------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------
def calcular_wacc(info, balance_sheet):
    """Devuelve WACC y deuda total."""
    try:
        beta = info.get("beta", 1.0)
        price = info.get("currentPrice")
        shares = info.get("sharesOutstanding")
        market_cap = price * shares if price and shares else None

        lt_debt = balance_sheet.loc["Long Term Debt"].iloc[0] if "Long Term Debt" in balance_sheet.index else 0
        st_debt = balance_sheet.loc["Short Term Debt"].iloc[0] if "Short Term Debt" in balance_sheet.index else 0
        total_debt = (lt_debt or 0) + (st_debt or 0)

        Re = Rf + beta * (Rm - Rf)        # Coste del capital
        Rd = 0.055 if total_debt else 0   # Coste de la deuda

        E = market_cap or 0
        D = total_debt

        if E + D == 0:
            return None, total_debt

        wacc = (E / (E + D)) * Re + (D / (E + D)) * Rd * (1 - Tc)
        return wacc, total_debt
    except Exception as e:
        st.error(f"Error calculando WACC: {e}")
        return None, None


def calcular_crecimiento_historico(financials, metric):
    """CAGR a 4 periodos si el dato existe."""
    try:
        if metric not in financials.index:
            return None
        datos = financials.loc[metric].dropna().iloc[:4]
        if len(datos) < 2:
            return None
        primer_valor = datos.iloc[-1]
        ultimo_valor = datos.iloc[0]
        años = len(datos) - 1
        if primer_valor == 0:
            return None
        return (ultimo_valor / primer_valor) ** (1 / años) - 1
    except:
        return None


def extraer_ebit(financials):
    """Intenta encontrar EBIT independientemente del nombre exacto."""
    posibles = ["EBIT", "Ebit", "Earnings Before Interest and Taxes", "Operating Income"]
    for key in posibles:
        if key in financials.index:
            return financials.loc[key].iloc[0]
    return None


def obtener_datos_financieros(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        bs = stock.balance_sheet
        fin = stock.financials
        cf = stock.cashflow

        # ==== Datos básicos ====
        price = info.get("currentPrice")
        name = info.get("longName", ticker)
        sector = info.get("sector", "N/D")
        country = info.get("country", "N/D")
        industry = info.get("industry", "N/D")

        # ==== Ratios de valoración ====
        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        dividend = info.get("dividendRate")
        dividend_yield = info.get("dividendYield")
        payout = info.get("payoutRatio")

        # ==== Ratios de rentabilidad ====
        roa = info.get("returnOnAssets")
        roe = info.get("returnOnEquity")

        # ==== Ratios de liquidez ====
        current_ratio = info.get("currentRatio")
        quick_ratio = info.get("quickRatio")

        # ==== Ratios de deuda ====
        ltde = info.get("longTermDebtToEquity")
        de = info.get("debtToEquity")

        # ==== Márgenes ====
        op_margin = info.get("operatingMargins")
        profit_margin = info.get("profitMargins")

        # ==== Flujo de caja ====
        fcf = cf.loc["Free Cash Flow"].iloc[0] if "Free Cash Flow" in cf.index else None
        shares = info.get("sharesOutstanding")
        pfcf = price / (fcf / shares) if fcf and shares else None

        # ==== Cálculos avanzados ====
        ebit = extraer_ebit(fin)
        equity = bs.loc["Total Stockholder Equity"].iloc[0] if "Total Stockholder Equity" in bs.index else None
        wacc, total_debt = calcular_wacc(info, bs)

        # Capital invertido: si falta algún componente lo tratamos como 0
        total_debt = total_debt or 0
        equity = equity or 0
        capital_invertido = total_debt + equity

        roic = (
            (ebit * (1 - Tc)) / capital_invertido
            if (ebit is not None) and capital_invertido
            else None
        )
        eva = (
            (roic - wacc) * capital_invertido
            if (roic is not None) and (wacc is not None) and capital_invertido
            else None
        )

        # ==== Crecimientos ====
        revenue_growth = calcular_crecimiento_historico(fin, "Total Revenue")
        eps_growth = calcular_crecimiento_historico(fin, "Net Income")
        fcf_growth = calcular_crecimiento_historico(cf, "Free Cash Flow") \
            or calcular_crecimiento_historico(cf, "Operating Cash Flow")

        # ==== Liquidez avanzada ====
        cash_ratio = info.get("cashRatio")
        operating_cash_flow = cf.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cf.index else None
        current_liabilities = bs.loc["Total Current Liabilities"].iloc[0] if "Total Current Liabilities" in bs.index else None
        cash_flow_ratio = (
            operating_cash_flow / current_liabilities
            if operating_cash_flow and current_liabilities
            else None
        )

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
            "Payout Ratio": payout,
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
# Interfaz principal
# -------------------------------------------------------------
def main():
    st.title("📊 Dashboard de Análisis Financiero Avanzado")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        tickers_input = st.text_area(
            "🔎 Ingresa tickers (separados por coma)",
            "AAPL, MSFT, GOOGL, AMZN, TSLA",
            help="Ejemplo: AAPL, MSFT, GOOG"
        )
        max_tickers = st.slider("Número máximo de tickers", 1, 100, 50)

        st.markdown("---")
        st.markdown("**Parámetros WACC**")
        global Rf, Rm, Tc
        Rf = st.number_input("Tasa libre de riesgo (%)", 0.0, 20.0, 4.35) / 100
        Rm = st.number_input("Retorno esperado del mercado (%)", 0.0, 30.0, 8.5) / 100
        Tc = st.number_input("Tasa impositiva corporativa (%)", 0.0, 50.0, 21.0) / 100

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][:max_tickers]

    if st.button("🔍 Analizar Acciones", type="primary"):
        if not tickers:
            st.warning("Por favor ingresa al menos un ticker")
            return

        resultados = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        batch_size = 10
        for batch_start in range(0, len(tickers), batch_size):
            batch_tickers = tickers[batch_start:batch_start+batch_size]
            for i, t in enumerate(batch_tickers):
                status_text.text(f"⏳ Procesando {t} ({batch_start + i + 1}/{len(tickers)})...")
                resultados[t] = obtener_datos_financieros(t)
                progress_bar.progress((batch_start + i + 1) / len(tickers))
                time.sleep(1)  # Evitar rate-limit

        status_text.text("✅ Análisis completado!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

        # ==== DataFrame ====
        datos = [d for d in resultados.values() if "Error" not in d]
        if not datos:
            st.error("No se pudo obtener datos válidos para ningún ticker")
            return

        df = pd.DataFrame(datos)

        # -----------------------------------------------------
        # Sección 1 - Resumen General
        # -----------------------------------------------------
        st.header("📋 Resumen General")

        porcentajes = ["Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                       "Oper Margin", "Profit Margin", "WACC", "ROIC"]
        for col in porcentajes:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/D")

        columnas_mostrar = [
            "Ticker", "Nombre", "Sector", "Precio", "P/E", "P/B", "P/FCF",
            "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
            "Current Ratio", "Debt/Eq", "Oper Margin", "Profit Margin",
            "WACC", "ROIC", "EVA"
        ]
        st.dataframe(
            df[columnas_mostrar].dropna(how='all', axis=1),
            use_container_width=True,
            height=400
        )
  
            # Sección 2: Análisis de Valoración
            st.header("💰 Análisis de Valoración")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Ratios de Valoración")
                fig, ax = plt.subplots(figsize=(10, 4))
                df_plot = df[["Ticker", "P/E", "P/B", "P/FCF"]].set_index("Ticker").apply(pd.to_numeric, errors='coerce')
                df_plot.plot(kind="bar", ax=ax, rot=45)
                ax.set_title("Comparativa de Ratios de Valoración")
                ax.set_ylabel("Ratio")
                st.pyplot(fig)
                plt.close()
                
            with col2:
                st.subheader("Dividendos")
                fig, ax = plt.subplots(figsize=(10, 4))
                df_plot = df[["Ticker", "Dividend Yield %"]].set_index("Ticker")
                df_plot["Dividend Yield %"] = df_plot["Dividend Yield %"].replace("N/D", 0)
                df_plot["Dividend Yield %"] = df_plot["Dividend Yield %"].str.rstrip("%").astype("float")
                df_plot.plot(kind="bar", ax=ax, rot=45, color="green")
                ax.set_title("Rendimiento de Dividendos (%)")
                ax.set_ylabel("Dividend Yield %")
                st.pyplot(fig)
                plt.close()
            
            # Sección 3: Rentabilidad y Eficiencia
            st.header("📈 Rentabilidad y Eficiencia")
            
            tabs = st.tabs(["ROE vs ROA", "Margenes", "WACC vs ROIC"])
            
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
                df_plot["Profit Margin"] = df_plot["Profit Margin"].str.rstrip("%").astype("float")
                df_plot.plot(kind="bar", ax=ax, rot=45)
                ax.set_title("Margen Operativo vs Margen Neto (%)")
                ax.set_ylabel("Porcentaje")
                st.pyplot(fig)
                plt.close()
                
            with tabs[2]:
                fig, ax = plt.subplots(figsize=(10, 5))
                for _, row in df.iterrows():
                    wacc = float(row["WACC"].rstrip("%")) if row["WACC"] != "N/D" else None
                    roic = float(row["ROIC"].rstrip("%")) if row["ROIC"] != "N/D" else None
                    
                    if wacc and roic:
                        color = "green" if roic > wacc else "red"
                        ax.bar(row["Ticker"], roic, color=color, alpha=0.6, label="ROIC")
                        ax.bar(row["Ticker"], wacc, color="gray", alpha=0.3, label="WACC")
                
                ax.set_title("Creación de Valor: ROIC vs WACC (%)")
                ax.set_ylabel("Porcentaje")
                ax.legend()
                st.pyplot(fig)
                plt.close()
            
            # Sección 4: Análisis de Deuda
            st.header("🏦 Estructura de Capital y Deuda")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Apalancamiento")
                fig, ax = plt.subplots(figsize=(10, 5))
                df_plot = df[["Ticker", "Debt/Eq", "LtDebt/Eq"]].set_index("Ticker")
                df_plot = df_plot.apply(pd.to_numeric, errors='coerce')
                df_plot.plot(kind="bar", stacked=True, ax=ax, rot=45)
                ax.axhline(1, color="red", linestyle="--")
                ax.set_title("Deuda/Patrimonio")
                ax.set_ylabel("Ratio")
                st.pyplot(fig)
                plt.close()
                
            with col2:
                st.subheader("Liquidez")
                fig, ax = plt.subplots(figsize=(10, 5))
                df_plot = df[["Ticker", "Current Ratio", "Quick Ratio", "Cash Ratio"]].set_index("Ticker")
                df_plot = df_plot.apply(pd.to_numeric, errors='coerce')
                df_plot.plot(kind="bar", ax=ax, rot=45)
                ax.axhline(1, color="green", linestyle="--")
                ax.set_title("Ratios de Liquidez")
                ax.set_ylabel("Ratio")
                st.pyplot(fig)
                plt.close()
            
            # Sección 5: Crecimiento
            st.header("🚀 Crecimiento Histórico")
            
            growth_metrics = ["Revenue Growth", "EPS Growth", "FCF Growth"]
            df_growth = df[["Ticker"] + growth_metrics].set_index("Ticker")
            df_growth = df_growth * 100  # Convertir a porcentaje
            
            fig, ax = plt.subplots(figsize=(12, 6))
            df_growth.plot(kind="bar", ax=ax, rot=45)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title("Tasas de Crecimiento Anual (%)")
            ax.set_ylabel("Crecimiento %")
            st.pyplot(fig)
            plt.close()
            
            # Sección 6: Análisis Individual
            st.header("🔍 Análisis por Empresa")
            
            selected_ticker = st.selectbox("Selecciona una empresa", df["Ticker"].unique())
            empresa = df[df["Ticker"] == selected_ticker].iloc[0]
            
            st.subheader(f"Análisis Detallado: {empresa['Nombre']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Precio", f"${empresa['Precio']:,.2f}" if empresa['Precio'] else "N/D")
                st.metric("P/E", empresa['P/E'])
                st.metric("P/B", empresa['P/B'])
                
            with col2:
                st.metric("ROE", empresa['ROE'])
                st.metric("ROIC", empresa['ROIC'])
                st.metric("WACC", empresa['WACC'])
                
            with col3:
                st.metric("Deuda/Patrimonio", empresa['Debt/Eq'])
                st.metric("Margen Neto", empresa['Profit Margin'])
                st.metric("Dividend Yield", empresa['Dividend Yield %'])
            
            # Gráfico de creación de valor individual
            st.subheader("Creación de Valor")
            fig, ax = plt.subplots(figsize=(6, 4))
            if empresa['ROIC'] != "N/D" and empresa['WACC'] != "N/D":
                roic_val = float(empresa['ROIC'].rstrip("%"))
                wacc_val = float(empresa['WACC'].rstrip("%"))
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

if __name__ == "__main__":
    main()
