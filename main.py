"""
Atividade Prática 02 - Visão Computacional (UFPI)

Q1: detectores Harris, SIFT e ORB em 3 pares de imagens, variando parâmetros:
    nº de pontos detectados e sobreposição entre os algoritmos.
Q2: correspondência de pontos com descritores SIFT e ORB (+ BRISK se
    existirem): taxa de acerto de cada par detector x descritor.

Uso: python -m pip install opencv-python numpy pandas matplotlib
     python main.py
Saídas em resultados/:
  q1_deteccao.csv, q1_sobreposicao.csv, q2_correspondencia.csv,
  q2_resumo.csv, q2_resumo.png
  + (só do último par, o seu) pontos.png, matches_melhor.jpg, matches_pior.jpg
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARES = {
    "par1": ("imgs/par1_a.jpg", "imgs/par1_b.jpg"),
    "par2": ("imgs/par2_a.jpg", "imgs/par2_b.jpg"),
    "par3_meu": ("imgs/par3_a.jpg", "imgs/par3_b.jpg"),   # o seu
}
OUT = "resultados"
os.makedirs(OUT, exist_ok=True)
TOL_OVERLAP = 3.0   # px: dois pontos "coincidem" se distância <= TOL
TOL_ACERTO = 3.0    # px: erro de reprojeção máximo para match correto
RATIO = 0.75        # teste da razão de Lowe


# ---------------------------------------------------------------- detectores
def det_harris(gray, k=0.04, thr=0.01, block=2, ksize=3, max_pts=20000):
    r = cv2.cornerHarris(np.float32(gray), block, ksize, k)
    r = cv2.dilate(r, None)
    mask = (r == cv2.dilate(r, np.ones((5, 5)))) & (r > thr * r.max())
    ys, xs = np.nonzero(mask)
    idx = np.argsort(-r[ys, xs])[:max_pts]
    return [cv2.KeyPoint(float(xs[i]), float(ys[i]), 10) for i in idx]


def det_sift(gray, contrast=0.04):
    return cv2.SIFT_create(contrastThreshold=contrast).detect(gray, None)


def det_orb(gray, fast_thr=20, nfeat=10000):
    return cv2.ORB_create(nfeatures=nfeat, fastThreshold=fast_thr).detect(gray, None)


# 3 configurações por detector (a primeira é a "padrão" usada no resto)
CONFIGS = {
    "Harris": [("k=0.04,thr=0.01", dict(k=0.04, thr=0.01)),
               ("k=0.04,thr=0.05", dict(k=0.04, thr=0.05)),
               ("k=0.06,thr=0.01", dict(k=0.06, thr=0.01))],
    "SIFT": [("contr=0.04", dict(contrast=0.04)),
             ("contr=0.02", dict(contrast=0.02)),
             ("contr=0.08", dict(contrast=0.08))],
    "ORB": [("fast=20", dict(fast_thr=20)),
            ("fast=10", dict(fast_thr=10)),
            ("fast=40", dict(fast_thr=40))],
}
FUNCS = {"Harris": det_harris, "SIFT": det_sift, "ORB": det_orb}


# ---------------------------------------------------------------- utilitários
def redimensiona(g, largura_max=1600):
    h, w = g.shape[:2]
    if w <= largura_max:
        return g
    return cv2.resize(g, (largura_max, int(h * largura_max / w)),
                      interpolation=cv2.INTER_AREA)


def pts(kps):
    return np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)


def limpa(kps):
    """Keypoints só com posição e tamanho (evita erro de memória/octave)."""
    return [cv2.KeyPoint(float(k.pt[0]), float(k.pt[1]), float(max(k.size, 8)))
            for k in kps]


def overlap(kp_a, kp_b, tol=TOL_OVERLAP):
    """Fração dos pontos de A com algum ponto de B a <= tol px."""
    A, B = pts(kp_a), pts(kp_b)
    if len(A) == 0 or len(B) == 0:
        return 0.0
    achou = 0
    for i in range(0, len(A), 500):       # em blocos para poupar memória
        d = np.linalg.norm(A[i:i + 500, None, :] - B[None, :, :], axis=2)
        achou += int((d.min(axis=1) <= tol).sum())
    return achou / len(A)


# ---------------------------------------------------------------- Q1
def questao1(nome, g1, g2):
    linhas, kps_ref = [], {}
    area = g1.shape[0] * g1.shape[1] / 1e6
    for det, cfgs in CONFIGS.items():
        for rotulo, params in cfgs:
            k1, k2 = FUNCS[det](g1, **params), FUNCS[det](g2, **params)
            linhas.append(dict(par=nome, detector=det, config=rotulo,
                               pts_img1=len(k1), pts_img2=len(k2),
                               pts_por_Mpx=round(len(k1) / area, 1)))
            if rotulo == cfgs[0][0]:
                kps_ref[det] = (k1, k2)
    dets = list(kps_ref)
    M = pd.DataFrame(index=dets, columns=dets, dtype=float)
    for a in dets:
        for b in dets:
            M.loc[a, b] = round(overlap(kps_ref[a][0], kps_ref[b][0]), 3)
    M.index.name = "detector"
    M.insert(0, "par", nome)
    return pd.DataFrame(linhas), M.reset_index(), kps_ref


# ---------------------------------------------------------------- Q2
def descritores():
    d = {"SIFT": (cv2.SIFT_create(), cv2.NORM_L2),
         "ORB": (cv2.ORB_create(), cv2.NORM_HAMMING)}
    for nome in ("BRISK",):               # opcional, depende do OpenCV
        f = getattr(cv2, f"{nome}_create", None)
        if f:
            d[nome] = (f(), cv2.NORM_HAMMING)
    return d


def homografia_referencia(g1, g2):
    """'Verdade' aproximada: SIFT + RANSAC (cena com plano dominante)."""
    s = cv2.SIFT_create(nfeatures=5000)
    k1, d1 = s.detectAndCompute(g1, None)
    k2, d2 = s.detectAndCompute(g2, None)
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    bons = [a for a, b in m if a.distance < 0.7 * b.distance]
    src = pts([k1[a.queryIdx] for a in bons])
    dst = pts([k2[a.trainIdx] for a in bons])
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    return H


def desenha_matches(g1, g2, k1c, k2c, bons, ok, caminho, max_linhas=150):
    """Verde = match correto, vermelho = errado."""
    h = max(g1.shape[0], g2.shape[0])
    c = np.zeros((h, g1.shape[1] + g2.shape[1], 3), np.uint8)
    c[:g1.shape[0], :g1.shape[1]] = cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR)
    c[:g2.shape[0], g1.shape[1]:] = cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)
    for i in np.random.default_rng(0).permutation(len(bons))[:max_linhas]:
        p1 = tuple(int(v) for v in k1c[bons[i].queryIdx].pt)
        x, y = k2c[bons[i].trainIdx].pt
        p2 = (int(x) + g1.shape[1], int(y))
        cor = (0, 200, 0) if ok[i] else (0, 0, 255)
        cv2.line(c, p1, p2, cor, 1, cv2.LINE_AA)
        cv2.circle(c, p1, 3, cor, -1)
        cv2.circle(c, p2, 3, cor, -1)
    cv2.imwrite(caminho, c)


def questao2(nome, g1, g2, kps_ref, H, figuras):
    linhas, guardados = [], []
    for det, (k1, k2) in kps_ref.items():
        for dsc, (ext, norma) in descritores().items():
            k1c, d1 = ext.compute(g1, limpa(k1))
            k2c, d2 = ext.compute(g2, limpa(k2))
            if d1 is None or d2 is None or len(k1c) < 2 or len(k2c) < 2:
                continue
            m = cv2.BFMatcher(norma).knnMatch(d1, d2, k=2)
            bons = [p[0] for p in m
                    if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
            if not bons:
                linhas.append(dict(par=nome, detector=det, descritor=dsc,
                                   matches=0, corretos=0, taxa_acerto=0.0))
                continue
            src = pts([k1c[x.queryIdx] for x in bons]).reshape(-1, 1, 2)
            dst = pts([k2c[x.trainIdx] for x in bons])
            proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
            ok = np.linalg.norm(proj - dst, axis=1) <= TOL_ACERTO
            linhas.append(dict(par=nome, detector=det, descritor=dsc,
                               matches=len(bons), corretos=int(ok.sum()),
                               taxa_acerto=round(float(ok.mean()), 4)))
            guardados.append((float(ok.mean()), det, dsc, k1c, k2c, bons, ok))
    if figuras and guardados:             # só melhor e pior combinação
        guardados.sort(key=lambda t: t[0])
        for rot, (_, det, dsc, k1c, k2c, bons, ok) in (("pior", guardados[0]),
                                                       ("melhor", guardados[-1])):
            print(f"  {rot}: {det} + {dsc}")
            desenha_matches(g1, g2, k1c, k2c, bons, ok,
                            f"{OUT}/matches_{rot}_{det}_{dsc}.jpg")
    return pd.DataFrame(linhas)


def salvar_pontos(g1, kps_ref):
    fig, ax = plt.subplots(1, len(kps_ref), figsize=(5 * len(kps_ref), 4))
    for a, (det, (k1, _)) in zip(np.atleast_1d(ax), kps_ref.items()):
        a.imshow(g1, cmap="gray")
        p = pts(k1)
        a.plot(p[:, 0], p[:, 1], "r.", ms=2)
        a.set_title(f"{det} ({len(k1)} pts)")
        a.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT}/pontos.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------- main
def main():
    ultimo = list(PARES)[-1]              # figuras só do seu par
    q1_all, ov_all, q2_all = [], [], []
    for nome, (p1, p2) in PARES.items():
        g1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        g2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        assert g1 is not None and g2 is not None, f"Imagem não encontrada em {nome}"
        g1, g2 = redimensiona(g1), redimensiona(g2)
        print(f"\n===== {nome} =====")
        df1, M, kps_ref = questao1(nome, g1, g2)
        H = homografia_referencia(g1, g2)
        df2 = questao2(nome, g1, g2, kps_ref, H, figuras=(nome == ultimo))
        if nome == ultimo:
            salvar_pontos(g1, kps_ref)
        print(df1.to_string(index=False))
        print("\nSobreposição:\n", M.to_string(index=False))
        print("\nCorrespondência:\n", df2.to_string(index=False))
        q1_all.append(df1)
        ov_all.append(M)
        q2_all.append(df2)

    pd.concat(q1_all).to_csv(f"{OUT}/q1_deteccao.csv", index=False)
    pd.concat(ov_all).to_csv(f"{OUT}/q1_sobreposicao.csv", index=False)
    q2 = pd.concat(q2_all)
    q2.to_csv(f"{OUT}/q2_correspondencia.csv", index=False)

    resumo = q2.pivot_table(index="detector", columns="descritor",
                            values="taxa_acerto", aggfunc="mean").round(3)
    resumo.to_csv(f"{OUT}/q2_resumo.csv")
    print("\nTaxa de acerto média (detector x descritor):\n", resumo)
    resumo.plot(kind="bar", figsize=(6, 4), ylabel="taxa de acerto")
    plt.tight_layout()
    plt.savefig(f"{OUT}/q2_resumo.png", dpi=150)


if __name__ == "__main__":
    main()