import streamlit as st
import pandas as pd
import numpy as np
import random

st.title("Sistem Pemilihan Vendor - Metode WP wuu perahu + Optimasi Bobot")

uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Vendor")
    st.dataframe(df)

    vendor_names = df.iloc[:, 0].tolist()
    kriteria = df.columns[1:].tolist()
    data = df.iloc[:, 1:].values

    st.subheader("Tentukan Atribut Cost / Benefit")
    atribut = []
    cols = st.columns(len(kriteria))
    for i, k in enumerate(kriteria):
        with cols[i]:
            atribut.append(st.selectbox(f"{k}", ['cost', 'benefit'], key=f"attr_{i}"))

    st.subheader("Metode Pembobotan")
    metode = st.radio("Pilih Metode Pembobotan:", ("Manual", "Otomatis (Genetic Algorithm)"))

    if metode == "Manual":
        st.write("Masukkan bobot untuk masing-masing kriteria (total harus = 1)")
        bobot_manual = []
        cols_bobot = st.columns(len(kriteria))
        for i, k in enumerate(kriteria):
            with cols_bobot[i]:
                bobot_manual.append(st.number_input(f"{k}", min_value=0.0, max_value=1.0, step=0.01, key=f"bobot_{i}"))

        total_bobot = sum(bobot_manual)
        if total_bobot > 1.001 or total_bobot < 0.999:
            st.error("Total bobot harus sama dengan 1!")
            st.stop()
        bobot = np.array(bobot_manual)

    else:  # Otomatis - GA
        def weighted_product(matrix, weights, atribut):
            weights = weights / np.sum(weights)
            bobot_wp = np.array([-w if attr == 'cost' else w for w, attr in zip(weights, atribut)])
            norm = matrix / np.max(matrix, axis=0)
            wp_score = np.prod(norm ** bobot_wp, axis=1)
            return wp_score

        def score(weights):
            weights = np.maximum(weights, 0.05)
            weights = weights / np.sum(weights)
            scores = weighted_product(data, weights, atribut)
            return np.max(scores)

        def crossover(p1, p2):
            alpha = random.random()
            return alpha * p1 + (1 - alpha) * p2

        def mutate(ind, rate=0.05):
            i = random.randint(0, len(ind) - 1)
            ind[i] += np.random.normal(0, rate)
            ind = np.maximum(ind, 0.05)
            ind = ind / np.sum(ind)
            return ind

        def evolve(pop, ngen=300):
            elite_size = 10
            mutation_rate = 0.1
            for _ in range(ngen):
                pop = sorted(pop, key=lambda x: -score(x))
                new_pop = pop[:elite_size]
                while len(new_pop) < len(pop):
                    p1, p2 = random.sample(pop[:elite_size], 2)
                    child = crossover(p1, p2)
                    if random.random() < mutation_rate:
                        child = mutate(child)
                    else:
                        child = np.maximum(child, 0.05)
                        child = child / np.sum(child)
                    new_pop.append(child)
                pop = new_pop
            return sorted(pop, key=lambda x: -score(x))[0]

        pop = [np.random.dirichlet(np.ones(len(kriteria))) for _ in range(50)]
        bobot = evolve(pop)

    if st.button("Hitung Vendor Terbaik"):

        def weighted_product(matrix, weights, atribut):
            weights = weights / np.sum(weights)
            bobot_wp = np.array([-w if attr == 'cost' else w for w, attr in zip(weights, atribut)])
            norm = matrix / np.max(matrix, axis=0)
            wp_score = np.prod(norm ** bobot_wp, axis=1)
            return wp_score

        # Hitung skor WP (S_i)
        nilai_vendor = weighted_product(data, bobot, atribut)

        # Normalisasi skor WP → v_i = S_i / ΣS_j
        total_skor = np.sum(nilai_vendor)
        nilai_normalisasi = nilai_vendor / total_skor

        # Hasil akhir
        hasil = pd.DataFrame({
            'Vendor': vendor_names,
            'Skor WP (S_i)': nilai_vendor,
            'Skor Normalisasi (v_i)': nilai_normalisasi
        }).sort_values(by='Skor Normalisasi (v_i)', ascending=False)

        st.subheader("Hasil Akhir WP")
        st.dataframe(hasil.reset_index(drop=True))

        st.success(f"🏆 Vendor terbaik: **{hasil.iloc[0]['Vendor']}** dengan skor normalisasi {hasil.iloc[0]['Skor Normalisasi (v_i)']:.4f}")

        st.write("Bobot yang digunakan:")
        st.write(pd.DataFrame({'Kriteria': kriteria, 'Bobot': np.round(bobot, 4)}))
