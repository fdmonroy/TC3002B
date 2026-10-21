import os
import argparse
import numpy as np
import pandas as pd

def generate_city_dataset(lat: float, lon: float, num_points: int = 150, spread: float = 50.0) -> pd.DataFrame:
    """Generate synthetic GPS coordinates around a central location."""
    points = np.random.randn(num_points, 2) / spread + [lat, lon]
    df = pd.DataFrame(points, columns=["lat", "lon"])
    df["lat"] = df["lat"].round(4)
    df["lon"] = df["lon"].round(4)
    return df

def save_datasets(output_dir: str | None = None, num_points: int = 150):
    """Generate and save CSV datasets for two distant cities (150 points each, 300 total)."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    # City 1: Buenos Aires, Argentina (South West quadrant relative to Equator and Greenwich)
    c1 = generate_city_dataset(-34.6037, -58.3816, num_points=num_points, spread=50.0)
    c1_path = os.path.join(output_dir, "city1.csv")
    c1.to_csv(c1_path, index=False)
    print(f"Generated {len(c1)} points for City 1 (Buenos Aires) at {c1_path}")
    
    # City 2: Cairo, Egypt (North East quadrant relative to Equator and Greenwich)
    c2 = generate_city_dataset(30.0444, 31.2357, num_points=num_points, spread=50.0)
    c2_path = os.path.join(output_dir, "city2.csv")
    c2.to_csv(c2_path, index=False)
    print(f"Generated {len(c2)} points for City 2 (Cairo) at {c2_path}")

def run_streamlit_app():
    """Streamlit interactive dashboard for GPS point visualization."""
    import streamlit as st

    st.title("Generador de Datasets GPS")
    st.write("Herramienta interactiva para generar y visualizar coordenadas GPS de ciudades.")

    city_choice = st.selectbox(
        "Seleccionar Ciudad Base",
        ["Buenos Aires, Argentina", "El Cairo, Egipto", "San Francisco, USA (37.76, -122.4)", "Personalizada"]
    )

    if city_choice == "Buenos Aires, Argentina":
        default_lat, default_lon = -34.6037, -58.3816
    elif city_choice == "El Cairo, Egipto":
        default_lat, default_lon = 30.0444, 31.2357
    elif city_choice == "San Francisco, USA (37.76, -122.4)":
        default_lat, default_lon = 37.7600, -122.4000
    else:
        default_lat, default_lon = 19.4326, -99.1332

    lat = st.number_input("Latitud Base", value=default_lat, format="%.4f")
    lon = st.number_input("Longitud Base", value=default_lon, format="%.4f")
    num_points = st.slider("Numero de Puntos", min_value=50, max_value=1000, value=150, step=50)

    df = generate_city_dataset(lat, lon, num_points=num_points)

    st.subheader(f"Mapa ({len(df)} coordenadas)")
    st.map(df)

    st.subheader("Datos Generados")
    st.dataframe(df)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar Dataset CSV",
        data=csv_data,
        file_name="dataset_gps.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPS dataset generator and Streamlit dashboard")
    parser.add_argument("--cli", action="store_true", help="Generate datasets via CLI without Streamlit")
    parser.add_argument("--points", type=int, default=150, help="Number of points per city")
    args = parser.parse_args()

    # Determine execution mode
    try:
        import streamlit as st
        # Detect if launched via streamlit run
        is_streamlit = os.environ.get("STREAMLIT_SERVER_PORT") is not None or "streamlit" in os.path.basename(os.getenv("_", ""))
    except ImportError:
        is_streamlit = False

    if is_streamlit and not args.cli:
        run_streamlit_app()
    else:
        save_datasets(num_points=args.points)
