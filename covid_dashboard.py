import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# konfigurasi awal halaman
st.set_page_config(page_title="Dashboard Klastering COVID-19", layout="wide")
st.title("Analisis Klastering COVID-19 - Modul Data Mining")

# load dataset (dengan cache biar cepet)
@st.cache_data
def ambil_data():
    data = pd.read_csv("covid_19_indonesia_time_series_all.csv")
    return data

data_covid = ambil_data()

# filter dan bersihin data
data_covid = data_covid[['Date', 'Location', 'Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']]
data_covid.dropna(inplace=True)

# pilih lokasi buat visualisasi tren
daftar_provinsi = data_covid['Location'].unique()
provinsi_dipilih = st.sidebar.selectbox("🗺️ Pilih Provinsi", daftar_provinsi)

data_provinsi = data_covid[data_covid['Location'] == provinsi_dipilih]

# visualisasi tren kasus
st.subheader(f"📈 Perkembangan Kasus di {provinsi_dipilih}")
fig_tren, ax_tren = plt.subplots(figsize=(10, 4))
data_harian = data_provinsi.groupby("Date")['Total Cases'].sum()
data_harian.plot(ax=ax_tren, color='darkred')
ax_tren.set_xlabel("Tanggal")
ax_tren.set_ylabel("Total Kasus")
st.pyplot(fig_tren)

# proses klastering
st.subheader("🔍 Proses Klasterisasi Provinsi")

fitur_klaster = data_covid.groupby("Location")[['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']].mean()
normalizer = StandardScaler()
fitur_klaster_scaled = normalizer.fit_transform(fitur_klaster)

model_klaster = KMeans(n_clusters=4, random_state=42)
hasil_klaster = model_klaster.fit_predict(fitur_klaster_scaled)

fitur_klaster['Cluster ID'] = hasil_klaster

# gabungkan hasil klaster ke data utama
data_klaster = data_covid.merge(fitur_klaster['Cluster ID'], on='Location')

# koordinat lokasi manual
lokasi_koordinat = pd.DataFrame({
    'Location': ['DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Bali', 'Sumatera Utara', 'Kalimantan Timur', 'Sulawesi Selatan'],
    'Latitude': [-6.2088, -6.9039, -7.1500, -7.2504, -8.4095, 3.5952, 0.5383, -5.1477],
    'Longitude': [106.8456, 107.6186, 110.1403, 112.7688, 115.1889, 98.6722, 116.4194, 119.4327]
})

peta_data = fitur_klaster.reset_index().merge(lokasi_koordinat, on='Location')

# visualisasi map
st.subheader("🗺️ Visualisasi Klaster pada Peta")
peta_klaster = px.scatter_mapbox(
    peta_data,
    lat="Latitude", lon="Longitude",
    hover_name="Location",
    color="Cluster ID",
    size="Total Cases",
    zoom=4,
    height=500,
    mapbox_style="carto-positron"
)
st.plotly_chart(peta_klaster, use_container_width=True)

# tabel ringkasan
st.subheader("📊 Ringkasan Tiap Klaster")
st.dataframe(fitur_klaster.sort_values("Cluster ID"))
