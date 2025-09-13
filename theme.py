import streamlit as st


def inject_css():
	st.markdown(
		"""
		<style>
		/* App background */
		.main, .block-container { background-color: #f7f9fc; }

		/* Rounded cards for metrics */
		div[data-testid="stMetric"] {
		  background: #ffffff;
		  padding: 16px 18px;
		  border-radius: 14px;
		  border: 1px solid #eef2f7;
		  box-shadow: 0 4px 14px rgba(17,24,39,0.06);
		  text-align: center;
		}

		/* Center label/value inside metric card */
		div[data-testid="stMetric"] > div {
		  display: flex;
		  flex-direction: column;
		  align-items: center;
		  justify-content: center;
		}

		div[data-testid="stMetric"] [data-testid="stMetricLabel"],
		div[data-testid="stMetric"] [data-testid="stMetricValue"],
		div[data-testid="stMetric"] span,
		div[data-testid="stMetric"] label { text-align: center !important; }

		/* Plotly charts container */
		div[data-testid="stPlotlyChart"] {
		  background: #ffffff;
		  padding: 12px;
		  border-radius: 14px !important;
		  border: 1px solid #eef2f7;
		  box-shadow: 0 6px 18px rgba(17,24,39,0.06);
		}

		/* Expander content */
		details {
		  background: #ffffff;
		  border-radius: 14px;
		  border: 1px solid #eef2f7;
		  box-shadow: 0 4px 14px rgba(17,24,39,0.04);
		  padding: 6px 10px;
		}

		/* Titles */
		h3, h4 { color: #0f172a; }
		</style>
		""",
		unsafe_allow_html=True,
	)
	return