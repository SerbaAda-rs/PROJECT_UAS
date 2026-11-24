import sympy as sp

def parse_matrix(raw_matrix):
    """Mengonversi input string (bulat/pecahan/desimal) menjadi SymPy Rational Matrix secara eksak."""
    parsed = []
    for row in raw_matrix:
        parsed_row = []
        for val in row:
            val_clean = str(val).strip().replace(",", ".")
            if val_clean == "":
                val_clean = "0"
            parsed_row.append(sp.Rational(val_clean))
        parsed.append(parsed_row)
    return sp.Matrix(parsed)

def parse_vector(raw_list):
    """Mengonversi list input menjadi Vektor Kolom SymPy Rational."""
    return sp.Matrix([sp.Rational(str(x).strip().replace(",", ".") or "0") for x in raw_list])

def det_sarrus_3x3(M):
    """Menghitung determinan 3x3 menggunakan metode Sarrus beserta penjabaran manualnya."""
    a, b, c = M[0, 0], M[0, 1], M[0, 2]
    d, e, f = M[1, 0], M[1, 1], M[1, 2]
    g, h, i = M[2, 0], M[2, 1], M[2, 2]
    
    utama = (a*e*i) + (b*f*g) + (c*d*h)
    sekunder = (c*e*g) + (a*f*h) + (b*d*i)
    det = utama - sekunder
    
    langkah = [
        f"\\text{{Diagonal Utama }} (\\searrow) = ({sp.latex(a)}\\cdot{sp.latex(e)}\\cdot{sp.latex(i)}) + ({sp.latex(b)}\\cdot{sp.latex(f)}\\cdot{sp.latex(g)}) + ({sp.latex(c)}\\cdot{sp.latex(d)}\\cdot{sp.latex(h)}) = {sp.latex(utama)}",
        f"\\text{{Diagonal Sekunder }} (\\swarrow) = ({sp.latex(c)}\\cdot{sp.latex(e)}\\cdot{sp.latex(g)}) + ({sp.latex(a)}\\cdot{sp.latex(f)}\\cdot{sp.latex(h)}) + ({sp.latex(b)}\\cdot{sp.latex(d)}\\cdot{sp.latex(i)}) = {sp.latex(sekunder)}",
        f"\\det(A) = {sp.latex(utama)} - ({sp.latex(sekunder)}) = {sp.latex(det)}"
    ]
    return det, langkah

def det_cofactor_steps(M):
    """Menghitung determinan dengan Ekspansi Kofaktor Baris Pertama."""
    n = M.rows
    if n == 2:
        det = M[0, 0]*M[1, 1] - M[0, 1]*M[1, 0]
        step = f"\\det(A) = ({sp.latex(M[0, 0])}\\cdot{sp.latex(M[1, 1])}) - ({sp.latex(M[0, 1])}\\cdot{sp.latex(M[1, 0])}) = {sp.latex(det)}"
        return det, [step]
    
    terms = []
    det_total = sp.Rational(0)
    for j in range(n):
        sub_mat = M.minor_submatrix(0, j)
        minor_val = sub_mat.det()
        sign = (-1)**j
        cofactor = sign * minor_val
        elem = M[0, j]
        det_total += elem * cofactor
        tanda_str = "+" if (j > 0 and sign > 0) else ("-" if sign < 0 else "")
        terms.append(f"{tanda_str} ({sp.latex(elem)}) \\det {sp.latex(sub_mat)}")
        
    langkah = [
        "\\text{Ekspansi Kofaktor sepanjang Baris ke-1:}",
        "\\det(A) = " + " ".join(terms),
        f"\\det(A) = {sp.latex(det_total)}"
    ]
    return det_total, langkah

def gauss_jordan_solver(M, is_augmented=True):
    """Melakukan OBE Gauss-Jordan dan menganalisis jenis solusi SPL."""
    mat = M.copy()
    rows, cols = mat.shape
    steps = [("Matriks Awal:", sp.latex(mat))]
    
    limit_col = cols - 1 if is_augmented else cols
    lead = 0
    for r in range(rows):
        if lead >= limit_col:
            break
        i = r
        while mat[i, lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if limit_col == lead:
                    break
        if limit_col == lead:
            break
            
        if i != r:
            mat.row_swap(i, r)
            steps.append((f"Tukar Baris $B_{{{r+1}}} \\leftrightarrow B_{{{i+1}}}$:", sp.latex(mat)))
            
        lv = mat[r, lead]
        if lv != 1:
            mat.row_op(r, lambda v, _: v / lv)
            steps.append((f"Jadikan 1 Utama ($B_{{{r+1}}} \\leftarrow \\frac{{1}}{{{sp.latex(lv)}}} B_{{{r+1}}}$):", sp.latex(mat)))
            
        for i in range(rows):
            if i != r:
                lv = mat[i, lead]
                if lv != 0:
                    mat.row_op(i, lambda v, j: v - lv * mat[r, j])
                    tanda = "-" if lv > 0 else "+"
                    abs_lv = abs(lv)
                    koef_str = "" if abs_lv == 1 else sp.latex(abs_lv)
                    steps.append((f"Eliminasi ($B_{{{i+1}}} \\leftarrow B_{{{i+1}}} {tanda} {koef_str}B_{{{r+1}}}$):", sp.latex(mat)))
        lead += 1

    status = "Selesai"
    solusi_teks = []
    if is_augmented:
        A_part = mat[:, :-1]
        rank_A = A_part.rank()
        rank_aug = mat.rank()
        num_vars = cols - 1
        
        if rank_A < rank_aug:
            status = "Tidak Ada Solusi (Sistem Inkonsisten)"
            solusi_teks.append("\\text{Terdapat baris kontradiksi } [0 \\dots 0 \\mid c] \\text{ dengan } c \\neq 0.")
        elif rank_A < num_vars:
            status = f"Solusi Tak Hingga ({num_vars - rank_A} Parameter Bebas)"
            syms = sp.symbols(f'x1:{num_vars+1}')
            sol_set = sp.linsolve((M[:, :-1], M[:, -1]), syms)
            sol_tuple = list(sol_set)[0]
            for idx, expr in enumerate(sol_tuple):
                solusi_teks.append(f"x_{{{idx+1}}} = {sp.latex(expr)}")
        else:
            status = "Solusi Tunggal (Konsisten)"
            for r in range(num_vars):
                solusi_teks.append(f"x_{{{r+1}}} = {sp.latex(mat[r, -1])}")
                
    return mat, steps, status, solusi_teks

def analisis_eigen(M):
    """Menghitung Polinomial Karakteristik, Eigenvalue, Eigenvector, dan Diagonalisasi."""
    lam = sp.Symbol('lambda')
    n = M.rows
    char_mat = M - lam * sp.eye(n)
    char_poly = sp.expand(char_mat.det())
    eigen_data = M.eigenvects()
    
    is_diag = M.is_diagonalizable()
    P, D = (M.diagonalize() if is_diag else (None, None))
    
    return {
        "char_mat": sp.latex(char_mat),
        "char_poly": sp.latex(char_poly),
        "eigen_data": eigen_data,
        "is_diagonalizable": is_diag,
        "P": sp.latex(P) if P is not None else None,
        "D": sp.latex(D) if D is not None else None
    }
