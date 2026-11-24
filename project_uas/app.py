import streamlit as st
import sympy as sp
import plotly.graph_objects as go
import numpy as np
from engine_aljabar import (
    parse_matrix, parse_vector, det_sarrus_3x3,
    det_cofactor_steps, gauss_jordan_solver, analisis_eigen
)

# ==========================================
# KONFIGURASI HALAMAN & LOAD CSS EKSTERNAL
# ==========================================
st.set_page_config(
    page_title="Aljabar Linear & Matriks Pro",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

st.markdown('<div class="main-header">📐 Sistem Komputasi Aljabar Linear & Matriks</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Kalkulator Eksak Berbasis Simbolik dengan Pembuktian Langkah Manual & Visualisasi Geometri</div>', unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAVIGASI
# ==========================================
with st.sidebar:
    st.title("📚 Navigasi Materi")
    modul = st.radio(
        "Pilih Topik Kuliah:",
        [
            "1. Operasi & Jenis Matriks",
            "2. Determinan & Invers",
            "3. Sistem Persamaan Linear (SPL)",
            "4. Vektor & Ruang Vektor",
            "5. Eigenvalue, Eigenvector & Grafika"
        ]
    )
    st.divider()
    st.info("ℹ️ **Standar Akurasi:** Semua perhitungan diproses sebagai bilangan rasional eksak ($p/q$) tanpa pembulatan desimal.")

# ==========================================
# MODUL 1: OPERASI & JENIS MATRIKS
# ==========================================
if modul == "1. Operasi & Jenis Matriks":
    st.header("1. Operasi Dasar & Identifikasi Jenis Matriks")
    col_in, col_out = st.columns([1, 1.3], gap="large")
    
    with col_in:
        operasi = st.selectbox("Pilih Operasi:", [
            "Penjumlahan (A + B)",
            "Pengurangan (A - B)",
            "Perkalian Skalar (k · A)",
            "Perkalian Matriks (A × B)",
            "Transpose & Identifikasi Jenis Matriks A"
        ])
        
        c1, c2 = st.columns(2)
        rA = c1.number_input("Baris A", 1, 6, 2)
        cA = c2.number_input("Kolom A", 1, 6, 2)
        
        st.markdown("**Elemen Matriks A:**")
        raw_A = []
        for i in range(rA):
            cols = st.columns(cA)
            raw_A.append([cols[j].text_input(f"A{i}{j}", value="2" if i==j else "1", key=f"m1a_{i}_{j}", label_visibility="collapsed") for j in range(cA)])
            
        if operasi in ["Penjumlahan (A + B)", "Pengurangan (A - B)", "Perkalian Matriks (A × B)"]:
            c3, c4 = st.columns(2)
            rB = c3.number_input("Baris B", 1, 6, 2)
            cB = c4.number_input("Kolom B", 1, 6, 2)
            st.markdown("**Elemen Matriks B:**")
            raw_B = []
            for i in range(rB):
                cols = st.columns(cB)
                raw_B.append([cols[j].text_input(f"B{i}{j}", value="3" if i==j else "4", key=f"m1b_{i}_{j}", label_visibility="collapsed") for j in range(cB)])
        elif operasi == "Perkalian Skalar (k · A)":
            skalar_str = st.text_input("Nilai Skalar (k):", value="3/2")
            
        btn_m1 = st.button("Proses Perhitungan", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Hasil & Pembuktian")
        if btn_m1:
            try:
                A = parse_matrix(raw_A)
                st.latex(f"A = {sp.latex(A)}")
                
                if operasi == "Transpose & Identifikasi Jenis Matriks A":
                    st.success("**Hasil Transpose ($A^T$):**")
                    st.latex(f"A^T = {sp.latex(A.T)}")
                    jenis = []
                    if A.is_square: jenis.append("Matriks Persegi")
                    if A.is_zero_matrix: jenis.append("Matriks Nol")
                    if A.is_square and A == sp.eye(A.rows): jenis.append("Matriks Identitas")
                    if A.is_diagonal(): jenis.append("Matriks Diagonal")
                    if A.is_symmetric(): jenis.append("Matriks Simetris ($A = A^T$)")
                    if A.is_upper: jenis.append("Matriks Segitiga Atas")
                    if A.is_lower: jenis.append("Matriks Segitiga Bawah")
                    
                    st.info("**Klasifikasi Jenis Matriks A:** " + (", ".join(jenis) if jenis else "Matriks Umum (Persegi Panjang)"))
                    
                elif operasi == "Perkalian Skalar (k · A)":
                    k = sp.Rational(skalar_str.strip())
                    hasil = k * A
                    st.success("**Hasil Akhir Perkalian Skalar:**")
                    st.latex(f"{sp.latex(k)} \\cdot {sp.latex(A)} = {sp.latex(hasil)}")
                    with st.expander("Langkah Manual Per Elemen", expanded=True):
                        jabaran_mat = sp.Matrix([[f"{sp.latex(k)} \\cdot ({sp.latex(A[i,j])})" for j in range(A.cols)] for i in range(A.rows)])
                        st.latex(f"k \\cdot A = {sp.latex(jabaran_mat)} = {sp.latex(hasil)}")
                    
                elif operasi in ["Penjumlahan (A + B)", "Pengurangan (A - B)"]:
                    B = parse_matrix(raw_B)
                    st.latex(f"B = {sp.latex(B)}")
                    if A.shape != B.shape:
                        st.error(f"Ordo tidak cocok! A berukuran {A.shape}, sedangkan B berukuran {B.shape}.")
                    else:
                        is_plus = "Penjumlahan" in operasi
                        hasil = A + B if is_plus else A - B
                        op_sym = "+" if is_plus else "-"
                        st.success(f"**Hasil Akhir ($A {op_sym} B$):**")
                        st.latex(f"A {op_sym} B = {sp.latex(hasil)}")
                        with st.expander("Langkah Manual Per Elemen", expanded=True):
                            jabaran_mat = sp.Matrix([[f"{sp.latex(A[i,j])} {op_sym} ({sp.latex(B[i,j])})" for j in range(A.cols)] for i in range(A.rows)])
                            st.latex(f"A {op_sym} B = {sp.latex(jabaran_mat)} = {sp.latex(hasil)}")
                        
                elif operasi == "Perkalian Matriks (A × B)":
                    B = parse_matrix(raw_B)
                    st.latex(f"B = {sp.latex(B)}")
                    if A.cols != B.rows:
                        st.error(f"Syarat perkalian gagal! Kolom A ({A.cols}) ≠ Baris B ({B.rows}).")
                    else:
                        hasil = A * B
                        st.success("**Hasil Akhir Perkalian Matriks ($A \\times B$):**")
                        st.latex(f"A \\times B = {sp.latex(hasil)}")
                        with st.expander("Jabaran Manual Baris × Kolom", expanded=True):
                            for i in range(A.rows):
                                for j in range(B.cols):
                                    suku = " + ".join([f"({sp.latex(A[i,k])} \\cdot {sp.latex(B[k,j])})" for k in range(A.cols)])
                                    st.latex(f"c_{{{i+1},{j+1}}} = {suku} = {sp.latex(hasil[i,j])}")
            except Exception as e:
                st.error(f"Input tidak valid: {e}")
        else:
            st.info("👈 Masukkan elemen matriks di sebelah kiri, lalu klik **Proses Perhitungan**.")

# ==========================================
# MODUL 2: DETERMINAN & INVERS
# ==========================================
elif modul == "2. Determinan & Invers":
    st.header("2. Determinan & Invers Matriks")
    col_in, col_out = st.columns([1, 1.3], gap="large")
    
    with col_in:
        ordo = st.number_input("Ordo Matriks Persegi (n × n)", 2, 5, 3)
        metode_det = st.selectbox("Metode Determinan:", ["Ekspansi Kofaktor (Semua Ordo)", "Metode Sarrus (Khusus 3×3)"])
        
        default_m = [["2", "1", "1"], ["1", "3", "2"], ["1", "0", "0"]]
        raw_M = []
        for i in range(ordo):
            cols = st.columns(ordo)
            raw_M.append([cols[j].text_input(f"D{i}{j}", value=(default_m[i][j] if ordo==3 else ("1" if i==j else "0")), key=f"m2_{i}_{j}", label_visibility="collapsed") for j in range(ordo)])
            
        btn_m2 = st.button("Hitung Determinan & Invers", type="primary", use_container_width=True)
        
    with col_out:
        st.subheader("Hasil & Pembuktian")
        if btn_m2:
            try:
                M = parse_matrix(raw_M)
                st.latex(f"A = {sp.latex(M)}")
                
                if "Sarrus" in metode_det:
                    if ordo != 3:
                        st.error("Metode Sarrus hanya berlaku untuk matriks ordo 3×3.")
                        st.stop()
                    det_val, det_steps = det_sarrus_3x3(M)
                else:
                    det_val, det_steps = det_cofactor_steps(M)
                    
                st.info(f"**Nilai Determinan:** $\\det(A) = {sp.latex(det_val)}$")
                with st.expander("Langkah Perhitungan Determinan", expanded=True):
                    for s in det_steps:
                        st.latex(s)
                        
                st.divider()
                if det_val == 0:
                    st.warning("⚠️ Karena $\\det(A) = 0$, matriks ini bersifat **Singular** dan **tidak memiliki invers**.")
                else:
                    inv_M = M.inv()
                    adj_M = M.adjugate()
                    st.success("**Matriks Invers ($A^{-1}$):**")
                    st.latex(f"A^{{-1}} = {sp.latex(inv_M)}")
                    
                    tab1, tab2 = st.tabs(["Metode Adjoin", "Metode OBE (Gauss-Jordan)"])
                    with tab1:
                        st.latex(f"\\text{{Adj}}(A) = {sp.latex(adj_M)}")
                        st.latex(f"A^{{-1}} = \\frac{{1}}{{{sp.latex(det_val)}}} {sp.latex(adj_M)} = {sp.latex(inv_M)}")
                    with tab2:
                        aug_I = M.row_join(sp.eye(M.rows))
                        rref_I, steps_I, _, _ = gauss_jordan_solver(aug_I, is_augmented=False)
                        for desc, ltx in steps_I:
                            st.markdown(desc)
                            st.latex(ltx)
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# MODUL 3: SISTEM PERSAMAAN LINEAR (SPL)
# ==========================================
elif modul == "3. Sistem Persamaan Linear (SPL)":
    st.header("3. Penyelesaian Sistem Persamaan Linear (OBE)")
    col_in, col_out = st.columns([1, 1.3], gap="large")
    
    with col_in:
        r_spl = st.number_input("Jumlah Persamaan", 2, 5, 3)
        c_spl = st.number_input("Jumlah Variabel", 2, 5, 3)
        
        st.markdown("**Matriks Diperbesar $[A \\mid b]$:**")
        def_spl = [["1", "1", "1", "6"], ["2", "-1", "1", "3"], ["1", "2", "-1", "2"]]
        raw_spl = []
        for i in range(r_spl):
            cols = st.columns(c_spl + 1)
            row_v = []
            for j in range(c_spl + 1):
                lbl = f"x{j+1}" if j < c_spl else "b"
                dv = def_spl[i][j] if (r_spl==3 and c_spl==3) else "1"
                row_v.append(cols[j].text_input(lbl, value=dv, key=f"m3_{i}_{j}"))
            raw_spl.append(row_v)
            
        btn_m3 = st.button("Selesaikan SPL", type="primary", use_container_width=True)
        
    with col_out:
        st.subheader("Analisis Solusi & Langkah OBE")
        if btn_m3:
            try:
                aug = parse_matrix(raw_spl)
                rref_m, steps, status, sol_teks = gauss_jordan_solver(aug, is_augmented=True)
                
                if "Tidak Ada Solusi" in status:
                    st.error(f"**Status Sistem:** {status}")
                elif "Tak Hingga" in status:
                    st.warning(f"**Status Sistem:** {status}")
                else:
                    st.success(f"**Status Sistem:** {status}")
                    
                for s in sol_teks:
                    st.latex(s)
                    
                with st.expander("Lihat Langkah Lengkap Eliminasi Gauss-Jordan", expanded=True):
                    for desc, ltx in steps:
                        st.markdown(desc)
                        st.latex(ltx)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("👈 Masukkan koefisien SPL pada matriks diperbesar $[A \\mid b]$, lalu klik **Selesaikan SPL**.")

# ==========================================
# MODUL 4: VEKTOR & RUANG VEKTOR
# ==========================================
elif modul == "4. Vektor & Ruang Vektor":
    st.header("4. Vektor di Ruang Euklidian & Kebebasan Linear")
    tab_v, tab_rv = st.tabs(["Operasi Vektor & Visualisasi (R² / R³)", "Uji Kebebasan Linear & Basis"])
    
    with tab_v:
        col_in, col_out = st.columns([1, 1.3], gap="large")
        with col_in:
            dim = st.radio("Dimensi Ruang:", [2, 3], horizontal=True)
            st.markdown("**Vektor $\\mathbf{u}$:**")
            cols_u = st.columns(dim)
            raw_u = [cols_u[i].text_input(f"u{i+1}", value=str(i+2), key=f"vu_{i}") for i in range(dim)]
            
            st.markdown("**Vektor $\\mathbf{v}$:**")
            cols_v = st.columns(dim)
            raw_v = [cols_v[i].text_input(f"v{i+1}", value=str(3-i), key=f"vv_{i}") for i in range(dim)]
            
            btn_v = st.button("Hitung & Plot Vektor", type="primary", use_container_width=True)
            
        with col_out:
            if btn_v:
                try:
                    u = parse_vector(raw_u)
                    v = parse_vector(raw_v)
                    dot_uv = u.dot(v)
                    norm_u = u.norm()
                    norm_v = v.norm()
                    
                    st.latex(f"\\mathbf{{u}} = {sp.latex(u.T)}, \\quad \\mathbf{{v}} = {sp.latex(v.T)}")
                    if norm_u != 0 and norm_v != 0:
                        cos_theta = dot_uv / (norm_u * norm_v)
                        sudut_deg = sp.deg(sp.acos(cos_theta)).evalf(4)
                        st.info(f"**Dot Product ($\\mathbf{{u}} \\cdot \\mathbf{{v}}$):** ${sp.latex(dot_uv)}$ | **Sudut:** ${sudut_deg}^\\circ$")
                    else:
                        st.info(f"**Dot Product ($\\mathbf{{u}} \\cdot \\mathbf{{v}}$):** ${sp.latex(dot_uv)}$")
                        
                    st.markdown(f"**Panjang Vektor:** $\\|\\mathbf{{u}}\\| = {sp.latex(norm_u)}$, $\\|\\mathbf{{v}}\\| = {sp.latex(norm_v)}$")
                    
                    if dim == 3:
                        cross_uv = u.cross(v)
                        st.success("**Cross Product ($\\mathbf{u} \\times \\mathbf{v}$):**")
                        st.latex(f"\\mathbf{{u}} \\times \\mathbf{{v}} = {sp.latex(cross_uv.T)}")
                    
                    # Visualisasi Grafik Plotly
                    u_f = [float(x) for x in u]
                    v_f = [float(x) for x in v]
                    fig = go.Figure()
                    if dim == 2:
                        fig.add_trace(go.Scatter(x=[0, u_f[0]], y=[0, u_f[1]], mode='lines+markers', name='Vektor u', line=dict(width=4, color='#2563EB')))
                        fig.add_trace(go.Scatter(x=[0, v_f[0]], y=[0, v_f[1]], mode='lines+markers', name='Vektor v', line=dict(width=4, color='#059669')))
                        fig.update_layout(title="Visualisasi Vektor di R²", xaxis_title="X", yaxis_title="Y", height=400)
                    else:
                        c_f = [float(x) for x in u.cross(v)]
                        fig.add_trace(go.Scatter3d(x=[0, u_f[0]], y=[0, u_f[1]], z=[0, u_f[2]], mode='lines+markers', name='Vektor u', line=dict(width=6, color='#2563EB')))
                        fig.add_trace(go.Scatter3d(x=[0, v_f[0]], y=[0, v_f[1]], z=[0, v_f[2]], mode='lines+markers', name='Vektor v', line=dict(width=6, color='#059669')))
                        fig.add_trace(go.Scatter3d(x=[0, c_f[0]], y=[0, c_f[1]], z=[0, c_f[2]], mode='lines+markers', name='u × v (Tegak Lurus)', line=dict(width=6, color='#DC2626')))
                        fig.update_layout(title="Visualisasi Vektor 3D di R³", height=450)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.info("👈 Masukkan komponen vektor $\\mathbf{u}$ dan $\\mathbf{v}$, lalu klik **Hitung & Plot Vektor**.")

    with tab_rv:
        st.subheader("Uji Kebebasan Linear (Linear Independence) & Dimensi")
        jml_vek = st.number_input("Jumlah Vektor", 2, 5, 3)
        dim_rv = st.number_input("Dimensi Komponen (Rⁿ)", 2, 5, 3)
        
        raw_vecs = []
        for idx in range(jml_vek):
            cols = st.columns(dim_rv)
            raw_vecs.append([cols[j].text_input(f"v{idx+1}_{j}", value="1" if idx==j else "0", key=f"rv_{idx}_{j}", label_visibility="collapsed") for j in range(dim_rv)])
            
        if st.button("Uji Kebebasan Linear", type="primary"):
            try:
                M_cols = parse_matrix(raw_vecs).T
                rank_m = M_cols.rank()
                st.latex(f"A = {sp.latex(M_cols)}")
                st.markdown(f"**Rank Matriks (Dimensi Ruang Kolom):** {rank_m} (dari {jml_vek} vektor)")
                if rank_m == jml_vek:
                    st.success("✅ Himpunan vektor tersebut **BEBAS LINEAR (Linearly Independent)**.")
                else:
                    st.warning("⚠️ Himpunan vektor tersebut **BERGANTUNG LINEAR (Linearly Dependent)**.")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# MODUL 5: EIGENVALUE, EIGENVECTOR & GRAFIKA
# ==========================================
elif modul == "5. Eigenvalue, Eigenvector & Grafika":
    st.header("5. Eigenvalue, Eigenvector & Aplikasi Transformasi 2D")
    tab_eig, tab_graf = st.tabs(["Nilai Eigen & Diagonalisasi", "Aplikasi Informatika: Transformasi Grafika 2D"])
    
    with tab_eig:
        col_in, col_out = st.columns([1, 1.3], gap="large")
        with col_in:
            ordo_e = st.number_input("Ordo Matriks Persegi", 2, 4, 2)
            def_e = [["4", "1"], ["2", "3"]]
            raw_e = []
            for i in range(ordo_e):
                cols = st.columns(ordo_e)
                raw_e.append([cols[j].text_input(f"E{i}{j}", value=(def_e[i][j] if ordo_e==2 else ("2" if i==j else "0")), key=f"eig_{i}_{j}", label_visibility="collapsed") for j in range(ordo_e)])
            btn_e = st.button("Cari Nilai & Vektor Eigen", type="primary", use_container_width=True)
            
        with col_out:
            if btn_e:
                try:
                    M_e = parse_matrix(raw_e)
                    res = analisis_eigen(M_e)
                    st.markdown(r"1. Persamaan Karakteristik $\det(A - \lambda I) = 0$:")
                    st.latex(f"\\det {res['char_mat']} = {res['char_poly']} = 0")
                    st.markdown(r"2. Nilai Eigen ($\lambda$) & Vektor Eigen ($\mathbf{v}$):")
                    for val, mult, vecs in res["eigen_data"]:
                        vec_str = ", ".join([sp.latex(v) for v in vecs])
                        st.success(f"$\\lambda = {sp.latex(val)}$ (Multiplisitas: {mult})")
                        st.latex(r"\text{Basis Ruang Eigen: } \left\{ " + vec_str + r" \right\}")
                    st.markdown(r"3. Uji Diagonalisasi ($A = P D P^{-1}$):")
                    if res["is_diagonalizable"]:
                        st.info("Matriks Dapat Didiagonalisasi:")
                        st.latex(f"P = {res['P']}, \\quad D = {res['D']}")
                    else:
                        st.warning("Matriks Tidak Dapat Didiagonalisasi karena jumlah vektor eigen bebas linear kurang dari ordo matriks.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.info("👈 Masukkan elemen matriks persegi, lalu klik Cari Nilai & Vektor Eigen.")
                
    with tab_graf:
        st.subheader("Simulasi Transformasi Linear pada Objek 2D (Grafika Komputer)")
        col_g1, col_g2 = st.columns([1, 1.5])
        with col_g1:
            jenis_tf = st.selectbox("Pilih Matriks Transformasi:", [
                "Rotasi (Putar Sudut θ)",
                "Dilatasi / Scaling (Perbesaran)",
                "Refleksi terhadap Sumbu-X",
                "Shearing Horizontal"
            ])
            if "Rotasi" in jenis_tf:
                sudut = st.slider("Sudut Rotasi (Derajat):", 0, 360, 45)
                rad = np.radians(sudut)
                T_mat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
                st.latex(f"T = \\begin{{bmatrix}} \\cos({sudut}^\\circ) & -\\sin({sudut}^\\circ) \\\\ \\sin({sudut}^\\circ) & \\cos({sudut}^\\circ) \\end{{bmatrix}}")
            elif "Dilatasi" in jenis_tf:
                sx = st.slider("Skala X:", 0.5, 3.0, 2.0)
                sy = st.slider("Skala Y:", 0.5, 3.0, 1.5)
                T_mat = np.array([[sx, 0], [0, sy]])
                st.latex(f"T = \\begin{{bmatrix}} {sx} & 0 \\\\ 0 & {sy} \\end{{bmatrix}}")
            elif "Refleksi" in jenis_tf:
                T_mat = np.array([[1, 0], [0, -1]])
                st.latex("T = \\begin{bmatrix} 1 & 0 \\\\ 0 & -1 \\end{bmatrix}")
            else:
                sh = st.slider("Faktor Shear (k):", -2.0, 2.0, 1.0)
                T_mat = np.array([[1, sh], [0, 1]])
                st.latex(f"T = \\begin{{bmatrix}} 1 & {sh} \\\\ 0 & {1} \\end{{bmatrix}}")
                
        with col_g2:
            # Objek awal: Segitiga dengan titik (0,0), (2,0), (1,2)
            pts = np.array([[0, 2, 1], [0, 0, 2]])
            pts_tf = T_mat @ pts
            fig_tf = go.Figure()
            fig_tf.add_trace(go.Scatter(x=pts[0], y=pts[1], fill="toself", name="Objek Awal", line=dict(color="#94A3B8")))
            fig_tf.add_trace(go.Scatter(x=pts_tf[0], y=pts_tf[1], fill="toself", name="Hasil Transformasi (T·v)", line=dict(color="#2563EB")))
            fig_tf.update_layout(
                title="Visualisasi Pemetaan T: R² → R²",
                xaxis=dict(range=[-5, 6], zeroline=True),
                yaxis=dict(range=[-5, 6], zeroline=True, scaleanchor="x", scaleratio=1),
                height=420
            )
            st.plotly_chart(fig_tf, use_container_width=True)