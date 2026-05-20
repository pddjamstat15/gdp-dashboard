import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import folium
from streamlit_folium import st_folium

# ======================
# JUDUL DASHBOARD
# ======================

st.title("Dashboard Klasterisasi Pengelolaan Sampah Indonesia")

# ======================
# UPLOAD FILE
# ======================

uploaded_file = st.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

# ======================
# JIKA FILE DIUPLOAD
# ======================

if uploaded_file is not None:

    # Baca data
    data = pd.read_csv(uploaded_file)

    # Tampilkan data
    st.subheader("Data")
    st.dataframe(data)

    # ======================
    # PILIH VARIABEL
    # ======================

    fitur = st.multiselect(
        "Pilih Variabel Clustering",
        ['X1', 'X2', 'X3', 'X4', 'X5', 'X6'],
        default=['X1', 'X2', 'X3']
    )

    # Cek minimal 2 variabel
    if len(fitur) < 2:
        st.warning("Pilih minimal 2 variabel")
        st.stop()

    # ======================
    # JUMLAH CLUSTER
    # ======================

    k = st.slider(
        "Jumlah Cluster",
        min_value=2,
        max_value=6,
        value=3
    )

    # ======================
    # DATA CLUSTERING
    # ======================

    X = data[fitur]

    # Standardisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ======================
    # K-MEANS++
    # ======================

    model = KMeans(
        n_clusters=k,
        init='k-means++',
        random_state=42,
        n_init=10
    )

    # Cluster mulai dari 1
    data['Cluster'] = model.fit_predict(X_scaled) + 1

    # ======================
    # SILHOUETTE SCORE
    # ======================

    score = silhouette_score(
        X_scaled,
        data['Cluster']
    )

    st.metric(
        "Silhouette Score",
        round(score, 3)
    )

    # ======================
    # SCATTER PLOT
    # ======================

    st.subheader("Visualisasi Cluster")

    fig = px.scatter(
        data,
        x=fitur[0],
        y=fitur[1],
        color=data['Cluster'].astype(str),
        hover_name='Provinsi',
        title='Scatter Plot Cluster'
    )

    st.plotly_chart(fig)

    # ======================
    # PETA CLUSTER
    # ======================

    st.subheader("Peta Cluster Indonesia")

    m = folium.Map(
        location=[-2, 118],
        zoom_start=4
    )

    # Warna cluster
    colors = [
        'red',
        'blue',
        'green',
        'purple',
        'orange',
        'darkred'
    ]

    # Tambahkan marker
    for i in range(len(data)):

        cluster_num = int(data.iloc[i]['Cluster']) - 1

        folium.CircleMarker(
            location=[
                data.iloc[i]['Latitude'],
                data.iloc[i]['Longitude']
            ],
            radius=8,
            color=colors[cluster_num],
            fill=True,
            fill_color=colors[cluster_num],
            fill_opacity=0.7,
            popup=f"""
            <b>{data.iloc[i]['Provinsi']}</b>
            <br>
            Cluster: {data.iloc[i]['Cluster']}
            """
        ).add_to(m)

    # Tampilkan map
    st_folium(m, width=1000)

    # ======================
    # HASIL CLUSTERING
    # ======================

    st.subheader("Hasil Clustering")

    st.dataframe(data)

else:
    st.info("Silakan upload file CSV terlebih dahulu")
