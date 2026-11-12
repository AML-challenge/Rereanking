"""
Re-ranker gradient boosting: il modello che fa di piu' (KAG test locale 0.643, train out-of-fold 0.595).

Tutto il codice usato per allenarlo e per fare la submission (submissions/submission_gb.csv), in un file solo e
in ordine di esecuzione. Prima era sparso in diag/common.py, p2common.py, oa.py, phase2_flip.py, phase3_rerank.py,
pipeline.py e test_eval.py. Legge solo data/ (csv, immagini, maschere) e i pesi dei modelli.

    python reranker_gbdt.py train     # tabella dei candidati sul train + allena i due modelli   (~30 min, GPU)
    python reranker_gbdt.py cv        # KAG out-of-fold sul train, GroupKFold(5) per categoria  (secondi)
    python reranker_gbdt.py predict   # data/test.csv -> reranker_gbdt_out/submission.csv       (~8 min, GPU)

COME FUNZIONA
  1. Feature. DINOv2 ViT-L/14 su un crop quadrato attorno alla maschera (+10% di margine) portato a 896 px,
     cioe' una griglia 64x64 di patch. Fuori maschera l'immagine e' dipinta col colore medio ImageNet.
     Feature dell'ultimo blocco, con la LayerNorm finale, normalizzate L2.
  2. Candidati. La feature del patch che contiene il keypoint sorgente da' una mappa di similarita' coseno sui
     patch del target dentro la maschera. Si tengono i 6 massimi locali piu' alti (NMS su finestra 7x7 patch).
     Ogni candidato e' rifinito con un soft-argmax sulla finestra 7x7 attorno al picco (temperatura 0.05).
  3. Indizi. 23 numeri per candidato (sotto).
  4. Etichetta. 1 se il centro del patch candidato e' a meno di 0.1*scale da un target_point (metrica KAG).
  5. Modello. HistGradientBoostingClassifier su tutti i candidati del train (392 coppie, 2495 query, ~14.8k righe).
     Al test vince il candidato con la probabilita' piu' alta; si restituisce la sua posizione rifinita.
  L'argmax da solo fa 0.571 OOF; scegliendo sempre il candidato giusto tra i 6 si arriverebbe a 0.772 OOF / 0.776 test.

GLI INDIZI (tra parentesi: punti KAG persi togliendo il gruppo, OOF sul train; il rumore e' circa +-1 punto)
  1 similarita' (1.6)
      sim           similarita' coseno al picco
      rank          posizione del picco nella classifica (0 = argmax)
      sim_gap       sim meno il picco piu' alto
      sim_ratio     sim diviso il picco piu' alto
      n_peaks85     quanti picchi superano 0.85*massimo: mappa ambigua (uguale per tutti i candidati della query)
  2 ritorno (0.2)
      cyc_d         dal candidato si cerca il patch piu' simile nella maschera sorgente: distanza dal keypoint,
                    in frazione dell'estensione dell'oggetto
      cyc_ratio     sim(keypoint, candidato) / sim del miglior ritorno (1 = il ritorno cade sul keypoint)
  3 orientamento (0.2)
      sim_flip      similarita' al candidato partendo dalla sorgente specchiata
      p_flip        probabilita' che la sorgente specchiata funzioni meglio (classificatore di coppia, sotto)
      sim_flip_x_p  sim_flip * p_flip
  4 seconda vista (0.1)
      sim_L2        similarita' con il penultimo blocco della ViT-L
      sim_bb2       similarita' con DINOv2 ViT-L/14 con registri (ultimo blocco)
  5 contesto (0.4)
      ctx           8 patch attorno al keypoint sorgente (a ~4 patch): per ognuno la miglior similarita' nel target
                    allo stesso spostamento dal candidato (finestra 5x5); media sugli 8
      ctx_mirror    uguale, con gli spostamenti orizzontali specchiati
      ctx_diff      ctx - ctx_mirror
  6 posizione nell'oggetto (1.7)
      du, dv        differenza di posizione relativa (x, y) nel riquadro della maschera, keypoint vs candidato
      du_flip       come du con la x del keypoint specchiata (1 - u)
      border_src    distanza del keypoint dal bordo della maschera (frazione della griglia)
      border_cand   idem per il candidato
      border_diff   border_src - border_cand
  7 parti ripetute (0.1)
      n_self_peaks  quanti picchi forti ha il keypoint nella sua stessa immagine (zampe, ruote, occhi)
  8 nitidezza (0.4)
      sharp         picco meno la media della mappa sull'anello a 3-5 patch

p_flip E' UN SECONDO MODELLO, A LIVELLO DI COPPIA: regressione logistica su 9 indizi.
  Etichetta: 1 se con la sorgente specchiata l'argmax (+ soft-argmax) azzecca piu' keypoint, 0 se meno; i pareggi
  non entrano nell'allenamento. Da solo e' vicino al caso (46-52%), ma fa parte del modello inviato.
      A_corr, A_n   match mutui densi (un patch ogni 2): correlazione tra x sorgente e x target, numero di match
      B_cls, B_mean similarita' sorgente-target meno specchiata-target, col token CLS e con la media dei patch
      C_n, C_sim    numero e similarita' media dei match mutui, normale meno specchiata
      D_opposite    Orient Anything v1: i due oggetti guardano verso lati opposti dell'immagine?
      D_daz, D_conf differenza di azimut (gradi) e confidenza minima di Orient Anything
  Nella tabella di train p_flip e' out-of-fold (stessi fold per categoria); al test viene dal classificatore
  allenato su tutte le coppie.

RIPRODUCIBILITA'. Con i modelli allenati sulle tabelle originali, predict_pair da' le stesse coordinate della
submission inviata (verificato su 63 query di test). La tabella originale pero' usava feature salvate in fp16:
rifacendo 'train' gli indizi cambiano di ~1e-3 e raramente due picchi quasi pari si scambiano, quindi il GBDT
riallenato puo' scegliere diversamente su qualche query.
"""
import argparse, json, os, pickle, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]                                   # AML-chall/
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "reranker_gbdt_out"
DINO_HUB = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"      # copia locale del repo torch.hub
OA_CKPT = Path.home() / "oa_weights" / "croplargeEX2_dino_weight.pt"          # Orient Anything v1

# ================================================================== 0. iperparametri (scelti sul train, fasi 1-3)
S = 896                     # lato del crop in pixel: griglia 64x64 con patch 14
MARGIN = 0.10               # margine attorno al riquadro della maschera
ALPHA = 0.10                # KAG: corretto se a meno di ALPHA * scale da un target_point
NMS_R = 3                   # raggio NMS dei picchi, in patch (finestra 7x7)
K = 6                       # candidati per keypoint
SOFT_R, SOFT_T = 3, 0.05    # soft-argmax: finestra 7x7 attorno al picco, temperatura
CTX_OFFS = [(0, 4), (0, -4), (4, 0), (-4, 0), (3, 3), (3, -3), (-3, 3), (-3, -3)]   # (dy, dx) dei patch di contesto
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

CUES = ["sim", "rank", "sim_gap", "sim_ratio", "n_peaks85",                         # 1 similarita'
        "cyc_d", "cyc_ratio",                                                        # 2 ritorno
        "sim_flip", "p_flip", "sim_flip_x_p",                                        # 3 orientamento
        "sim_L2", "sim_bb2",                                                         # 4 seconda vista
        "ctx", "ctx_mirror", "ctx_diff",                                             # 5 contesto
        "du", "dv", "du_flip", "border_src", "border_cand", "border_diff",           # 6 posizione
        "n_self_peaks",                                                              # 7 parti ripetute
        "sharp"]                                                                     # 8 nitidezza
FLIP_CUES = ["A_corr", "A_n", "B_cls", "B_mean", "C_n", "C_sim", "D_opposite", "D_daz", "D_conf"]

# con piu' di 10k righe sklearn attiva da solo l'early stopping (10% delle righe in validazione, random_state 0)
GBDT = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=0)
FLIP_CLF = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


# ================================================================== 1. dati
def load_split(name):
    """data/<name>.csv -> {pair_id: [query]} in ordine di pair_id. Sul train le query hanno anche tps e scale."""
    df = pd.read_csv(DATA / f"{name}.csv"); pairs = defaultdict(list)
    for r in df.itertuples():
        q = dict(id=r.id, src=r.source_image, tgt=r.target_image, sxy=(float(r.source_x), float(r.source_y)))
        if "target_points" in df.columns: q.update(tps=np.array(json.loads(r.target_points), float), scale=float(r.scale))
        pairs[r.pair_id].append(q)
    return dict(sorted(pairs.items()))


def load_image(im):
    img = Image.open(DATA / im).convert("RGB")
    mask = np.array(Image.open(DATA / "masks" / Path(im).with_suffix(".png"))) > 127
    return img, mask


def mask_bbox(m):
    ys, xs = np.nonzero(m); return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def flip_xy(xy, W): return np.array([W - xy[0], xy[1]], float)


def kag_hit(q, xy):
    return bool(np.min(np.linalg.norm(q["tps"] - np.asarray(xy, float), axis=1)) < ALPHA * q["scale"])


def category_folds(pairs):
    """GroupKFold(5) sulle coppie con gruppo = categoria: una categoria sta tutta in un fold"""
    cat = json.load(open(DATA / "image_categories.json")); pids = list(pairs); folds = {}
    for f, (_, va) in enumerate(GroupKFold(5).split(pids, groups=[cat[pairs[p][0]["src"]] for p in pids])):
        for i in va: folds[pids[i]] = f
    return folds


# ================================================================== 2. feature DINOv2
def load_dinov2(name):
    repo, source = (str(DINO_HUB), "local") if DINO_HUB.exists() else ("facebookresearch/dinov2", "github")
    return torch.hub.load(repo, name, source=source).eval().cuda()


def load_nets():
    return {"vitl": load_dinov2("dinov2_vitl14"), "vitl_reg": load_dinov2("dinov2_vitl14_reg")}


class View:
    """feature densa di un'immagine per un backbone e un blocco.
    full [h,h,D]: tutte le patch, L2-norm; feat: full azzerata fuori maschera; mask: patch toccate dalla maschera;
    box: crop quadrato in pixel originali (x0, y0, x1, y1); cls: token CLS normalizzato."""
    def __init__(self, full, cls, cov, box):
        self.full, self.cls, self.box, self.h = full, cls, box, full.shape[0]
        self.mask = np.round(cov * 255) > 0
        self.feat = full * self.mask[..., None]

    def patch_of(self, xy):
        """pixel originali -> indice patch (py, px)"""
        c = (np.asarray(xy, float) - self.box[:2]) * (S / (self.box[2] - self.box[0]))
        return int(np.clip(c[1] * self.h / S, 0, self.h - 1)), int(np.clip(c[0] * self.h / S, 0, self.h - 1))

    def to_orig(self, xu, yu):
        """unita' patch (il centro del patch i vale i + 0.5) -> pixel originali"""
        s = (self.box[2] - self.box[0]) / S
        return np.array([xu * S / self.h * s + self.box[0], yu * S / self.h * s + self.box[1]])

    def nearest(self, xy):
        """feature del patch che contiene il punto (anche fuori maschera)"""
        return self.full[self.patch_of(xy)]


def dino_views(net, img, mask, layers):
    """un forward -> una View per ogni blocco richiesto (-1 = ultimo, -2 = penultimo)"""
    a = np.array(img); a[~mask] = (MEAN * 255).astype(np.uint8)              # sfondo = media ImageNet (0 dopo la normalizzazione)
    ys, xs = np.nonzero(mask); x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side, cx, cy = max(x1 - x0, y1 - y0) * (1 + 2 * MARGIN), (x0 + x1) / 2, (y0 + y1) / 2
    box = np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2])
    crop = lambda im: np.asarray(im.transform((S, S), Image.EXTENT, tuple(float(v) for v in box), Image.BILINEAR), np.float32) / 255
    ic, mc = crop(Image.fromarray(a)), crop(Image.fromarray(mask.astype(np.uint8) * 255))
    x = torch.from_numpy(((ic - MEAN) / STD).transpose(2, 0, 1))[None].cuda()
    nb = len(net.blocks); idx = sorted(nb + l for l in layers)                  # la rete restituisce i blocchi in ordine crescente
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        outs = net.get_intermediate_layers(x, n=idx, reshape=True, return_class_token=True, norm=True)
    views = []
    for l in layers:
        p, c = outs[idx.index(nb + l)]
        f = F.normalize(p[0].float(), dim=0).permute(1, 2, 0).cpu().numpy()
        cov = F.adaptive_avg_pool2d(torch.from_numpy(mc)[None, None], f.shape[:2])[0, 0].numpy()   # copertura maschera per patch
        views.append(View(f, F.normalize(c[0].float(), dim=-1).cpu().numpy(), cov, box))
    return views


# ================================================================== 3. mappe di similarita', picchi, rifinitura
def sim_map(f, v):
    s = v.feat @ f; s[~v.mask] = -np.inf; return s


def find_peaks(s, mask, k, rel=0.0):
    """massimi locali (NMS 7x7) dentro la maschera, in ordine decrescente, con valore >= rel * massimo
    (con rel = 0 restano fuori i picchi a similarita' negativa)"""
    lm = (s == maximum_filter(s, size=2 * NMS_R + 1, mode="constant", cval=-np.inf)) & mask
    ys, xs = np.nonzero(lm); v = s[ys, xs]; o = np.argsort(-v); ys, xs, v = ys[o], xs[o], v[o]
    if len(v) == 0: return [], v
    keep = v >= rel * v[0]
    return list(zip(ys[keep][:k], xs[keep][:k])), v[keep][:k]


def refine(s, v, py, px):
    """soft-argmax sulla finestra 7x7 attorno al picco (solo patch in maschera) -> pixel originali"""
    ys, xs = np.mgrid[max(0, py - SOFT_R):min(v.h, py + SOFT_R + 1), max(0, px - SOFT_R):min(v.h, px + SOFT_R + 1)]
    m = v.mask[ys, xs]; ys, xs = ys[m], xs[m]; val = s[ys, xs]
    w = np.exp((val - val.max()) / SOFT_T); w /= w.sum()
    return v.to_orig(float((w * (xs + 0.5)).sum()), float((w * (ys + 0.5)).sum()))


def argmax_pred(s, v):
    py, px = np.unravel_index(np.argmax(s), s.shape); return refine(s, v, py, px)


# ================================================================== 4. una coppia: viste e indizi di coppia (p_flip)
class Pair:
    """viste di una coppia.  vs, vt: ViT-L ultimo blocco (sorgente, target);  vf: sorgente specchiata;
    sL2, tL2: ViT-L penultimo blocco;  s2, t2: ViT-L con registri, ultimo blocco"""
    def __init__(self, nets, src, tgt):
        si, sm = load_image(src); ti, tm = load_image(tgt); self.W = si.size[0]
        self.vs, self.sL2 = dino_views(nets["vitl"], si, sm, (-1, -2))
        self.vt, self.tL2 = dino_views(nets["vitl"], ti, tm, (-1, -2))
        self.vf, = dino_views(nets["vitl"], si.transpose(Image.FLIP_LEFT_RIGHT), sm[:, ::-1].copy(), (-1,))
        self.s2, = dino_views(nets["vitl_reg"], si, sm, (-1,))
        self.t2, = dino_views(nets["vitl_reg"], ti, tm, (-1,))
        self.mbs, self.mbt = mask_bbox(sm), mask_bbox(tm)                   # riquadri delle maschere, pixel originali
        self.ys, self.xs = np.nonzero(self.vs.mask); self.Fs = self.vs.feat[self.ys, self.xs]
        self.side_s = max(np.ptp(self.ys), np.ptp(self.xs)) + 1               # estensione dell'oggetto sorgente, in patch
        self.dts = distance_transform_edt(self.vs.mask) / self.vs.h           # distanza dal bordo della maschera
        self.dtt = distance_transform_edt(self.vt.mask) / self.vt.h


def mutual_stats(vs, vt, stride=2):
    """match mutui densi dalla maschera sorgente (un patch ogni 2) al target e ritorno
    -> correlazione tra x sorgente e x target, numero di match, similarita' media"""
    ys, xs = np.nonzero(vs.mask); yt, xt = np.nonzero(vt.mask)
    Sm = vs.feat[ys, xs] @ vt.feat[yt, xt].T
    nt, ns = Sm.argmax(1), Sm.argmax(0)
    grid = (ys % stride == 0) & (xs % stride == 0)
    idx = np.flatnonzero(grid & (ns[nt] == np.arange(len(ys))))
    if len(idx) < 10: return 0.0, len(idx), float(Sm[idx, nt[idx]].mean()) if len(idx) else 0.0
    nu = lambda x, xx: (x - xx.min()) / max(xx.max() - xx.min(), 1)
    us, ut = nu(xs[idx], xs), nu(xt[nt[idx]], xt)
    c = 0.0 if us.std() == 0 or ut.std() == 0 else float(np.corrcoef(us, ut)[0, 1])
    return c, len(idx), float(Sm[idx, nt[idx]].mean())


def masked_mean(v):
    m = v.feat[v.mask].mean(0); return m / np.linalg.norm(m)


def flip_cues(P, oa_s, oa_t):
    """i 9 indizi del classificatore di coppia"""
    cA, nA, sA = mutual_stats(P.vs, P.vt); cAf, nAf, sAf = mutual_stats(P.vf, P.vt)
    side = lambda az: np.sign(np.sin(np.deg2rad(az)))          # Orient Anything: az ~90 guarda a sinistra, ~270 a destra
    return dict(A_corr=cA, A_n=nA,
                B_cls=float(P.vs.cls @ P.vt.cls - P.vf.cls @ P.vt.cls),
                B_mean=float(masked_mean(P.vs) @ masked_mean(P.vt) - masked_mean(P.vf) @ masked_mean(P.vt)),
                C_n=nA - nAf, C_sim=sA - sAf,
                D_opposite=float(side(oa_s["azimuth"]) * side(oa_t["azimuth"]) < 0),
                D_daz=abs((oa_s["azimuth"] - oa_t["azimuth"] + 180) % 360 - 180), D_conf=min(oa_s["conf"], oa_t["conf"]))


def flip_label(P, qs):
    """1 se la sorgente specchiata azzecca piu' keypoint (argmax + soft-argmax), 0 se meno, -1 se pari"""
    n = f = 0
    for q in qs:
        n += kag_hit(q, argmax_pred(sim_map(P.vs.nearest(q["sxy"]), P.vt), P.vt))
        f += kag_hit(q, argmax_pred(sim_map(P.vf.nearest(flip_xy(q["sxy"], P.W)), P.vt), P.vt))
    return 1 if f > n else 0 if n > f else -1


# ================================================================== 5. candidati e indizi di un keypoint
def candidate_rows(P, sxy):
    """i K candidati di un keypoint sorgente con i 23 indizi. p_flip e sim_flip_x_p si aggiungono dopo:
    dipendono dal classificatore di coppia. px, py = posizione rifinita; cx, cy = centro del patch."""
    vs, vt = P.vs, P.vt
    kp = vs.patch_of(sxy); f = vs.nearest(sxy)                              # patch e feature del keypoint sorgente
    s = sim_map(f, vt)                                                      # mappa principale
    sf = sim_map(P.vf.nearest(flip_xy(sxy, P.W)), vt)                       # dalla sorgente specchiata
    sL = sim_map(P.sL2.nearest(sxy), P.tL2)                                 # penultimo blocco
    sb = sim_map(P.s2.nearest(sxy), P.t2)                                   # backbone con registri
    pk, vals = find_peaks(s, vt.mask, K); top = float(vals[0])
    n85 = len(find_peaks(s, vt.mask, 50, rel=0.85)[0])
    nself = len(find_peaks(sim_map(f, vs), vs.mask, 50, rel=0.85)[0])
    us = ((sxy[0] - P.mbs[0]) / (P.mbs[2] - P.mbs[0]), (sxy[1] - P.mbs[1]) / (P.mbs[3] - P.mbs[1]))
    rows = []
    for rank, (py, px) in enumerate(pk):
        center, pred, ft = vt.to_orig(px + 0.5, py + 0.5), refine(s, vt, py, px), vt.feat[py, px]
        # 2. ritorno: dal candidato al patch piu' simile della maschera sorgente
        back = P.Fs @ ft; j = int(np.argmax(back))
        cyc_d = np.hypot(P.ys[j] - kp[0], P.xs[j] - kp[1]) / P.side_s
        cyc_ratio = float(f @ ft) / max(float(back[j]), 1e-6)
        # 5. contesto: i vicini del keypoint ritrovati allo stesso spostamento dal candidato (e specchiati)
        ctx = {False: [], True: []}
        for dy, dx in CTX_OFFS:
            sy, sx = kp[0] + dy, kp[1] + dx
            if not (0 <= sy < vs.h and 0 <= sx < vs.h and vs.mask[sy, sx]): continue
            fo = vs.feat[sy, sx]
            for mir in (False, True):
                cy, cx = py + dy, px + (-dx if mir else dx)
                y0, y1, x0, x1 = max(0, cy - 2), min(vt.h, cy + 3), max(0, cx - 2), min(vt.h, cx + 3)
                if y0 >= y1 or x0 >= x1: continue
                w = vt.mask[y0:y1, x0:x1]
                if w.any(): ctx[mir].append(float((vt.feat[y0:y1, x0:x1][w] @ fo).max()))
        cn = np.mean(ctx[False]) if ctx[False] else 0.0; cm = np.mean(ctx[True]) if ctx[True] else 0.0
        # 6. posizione relativa nel riquadro della maschera
        uc = ((center[0] - P.mbt[0]) / (P.mbt[2] - P.mbt[0]), (center[1] - P.mbt[1]) / (P.mbt[3] - P.mbt[1]))
        # 8. nitidezza: picco meno la media sull'anello a 3-5 patch (distanza di Chebyshev)
        yy, xx = np.mgrid[max(0, py - 5):min(vt.h, py + 6), max(0, px - 5):min(vt.h, px + 6)]
        ring = (np.maximum(abs(yy - py), abs(xx - px)) >= 3) & vt.mask[yy, xx]
        sharp = float(s[py, px] - s[yy[ring], xx[ring]].mean()) if ring.any() else 0.0
        rows.append(dict(cand_rank=rank, px=float(pred[0]), py=float(pred[1]), cx=float(center[0]), cy=float(center[1]),
                         sim=float(s[py, px]), rank=rank, sim_gap=float(s[py, px] - top), sim_ratio=float(s[py, px] / top), n_peaks85=n85,
                         cyc_d=cyc_d, cyc_ratio=cyc_ratio,
                         sim_flip=float(sf[py, px]),
                         sim_L2=float(sL[py, px]) if P.tL2.mask[py, px] else 0.0, sim_bb2=float(sb[py, px]) if P.t2.mask[py, px] else 0.0,
                         ctx=cn, ctx_mirror=cm, ctx_diff=cn - cm,
                         du=abs(us[0] - uc[0]), dv=abs(us[1] - uc[1]), du_flip=abs((1 - us[0]) - uc[0]),
                         border_src=float(P.dts[kp]), border_cand=float(P.dtt[py, px]), border_diff=float(P.dts[kp] - P.dtt[py, px]),
                         n_self_peaks=nself, sharp=sharp))
    return rows


# ================================================================== 6. Orient Anything v1 (solo per gli indizi D)
class _MLP(nn.Module):                     # come vision_tower.MLP_dim del repo Orient-Anything
    def __init__(self, i, o):
        super().__init__(); self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(nn.Linear(i, o), nn.BatchNorm1d(o), self.act)
        self.net2 = nn.Sequential(nn.Linear(o, o), nn.BatchNorm1d(o))

    def forward(self, x): return self.net2(self.net1(x))


class OrientAnything(nn.Module):
    """DINOv2-L (HF) + MLP: CLS -> 360 bin di azimut, 180 polare, 180 rotazione, 2 confidenza"""
    def __init__(self):
        super().__init__()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")        # pesi gia' nella cache HF (qui il download si blocca)
        from transformers import Dinov2Model
        self.dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-large"); self.down_sampler = _MLP(1024, 722)

    def forward(self, x): return self.down_sampler(self.dinov2(pixel_values=x).last_hidden_state[:, 0])


def load_oa():
    m = OrientAnything().eval(); sd = torch.load(OA_CKPT, map_location="cpu", weights_only=True)
    if not set(sd) <= set(m.state_dict()):          # checkpoint salvato con transformers 4.x: nomi dell'attenzione diversi
        rules = [(".attention.attention.query.", ".attention.q_proj."), (".attention.attention.key.", ".attention.k_proj."),
                 (".attention.attention.value.", ".attention.v_proj."), (".attention.output.dense.", ".attention.o_proj.")]
        for a, b in rules: sd = {k.replace(a, b): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False); assert not missing and not unexpected, (missing[:3], unexpected[:3])
    return m


def oa_tensor(img, mask, size=224):
    """oggetto mascherato su nero, quadrato con l'oggetto all'85%, poi resize 256 + center crop 224 come nel repo"""
    img = np.asarray(img).copy(); img[~mask] = 0
    ys, xs = np.nonzero(mask); fg = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = int(round(max(fg.shape[:2]) / 0.85)); can = np.zeros((side, side, 3), np.uint8)
    y0, x0 = (side - fg.shape[0]) // 2, (side - fg.shape[1]) // 2; can[y0:y0 + fg.shape[0], x0:x0 + fg.shape[1]] = fg
    pil = Image.fromarray(can).resize((256, 256), Image.BICUBIC); o = (256 - size) // 2
    a = np.asarray(pil.crop((o, o, o + size, o + size)), np.float32) / 255
    return torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1))


def orient_all(images, device="cuda"):
    """{immagine: {azimuth, conf}}; gira prima di caricare DINOv2 e poi libera la GPU"""
    m = load_oa().to(device); out = {}
    for i in range(0, len(images), 16):
        b = images[i:i + 16]; x = torch.stack([oa_tensor(*load_image(im)) for im in b]).to(device)
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            p = m(x).float()
        az, conf = p[:, :360].argmax(1), torch.softmax(p[:, -2:], 1)[:, 0]
        for j, im in enumerate(b): out[im] = dict(azimuth=float(az[j]), conf=float(conf[j]))
    del m; torch.cuda.empty_cache()
    return out


# ================================================================== 7. allenamento
def build_table(pairs, folds):
    """-> (una riga per coppia con indizi ed etichetta di flip, una riga per candidato con indizi ed etichetta)"""
    images = sorted({q[k] for qs in pairs.values() for q in qs for k in ("src", "tgt")})
    oa = orient_all(images); nets = load_nets(); prows, rows = [], []; t0 = time.time()
    for n, (pid, qs) in enumerate(pairs.items()):
        P = Pair(nets, qs[0]["src"], qs[0]["tgt"])
        prows.append(dict(pair=pid, fold=folds[pid], label=flip_label(P, qs), **flip_cues(P, oa[qs[0]["src"]], oa[qs[0]["tgt"]])))
        for q in qs:
            for r in candidate_rows(P, q["sxy"]):
                rows.append(dict(pair_id=pid, query_id=q["id"], fold=folds[pid],
                                 label=int(kag_hit(q, (r["cx"], r["cy"]))),     # etichetta: centro del patch
                                 kag=int(kag_hit(q, (r["px"], r["py"]))),       # cio' che conta: posizione rifinita
                                 **r))
        if n % 20 == 0: print(f"  coppia {n + 1}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
    pf, t = pd.DataFrame(prows), pd.DataFrame(rows)
    # p_flip out-of-fold: per ogni fold il classificatore e' allenato sulle coppie (non in pareggio) degli altri fold
    pf["p_flip"] = 0.0
    for f in range(5):
        tr, va = (pf.fold != f) & (pf.label >= 0), pf.fold == f
        pf.loc[va, "p_flip"] = clone(FLIP_CLF).fit(pf.loc[tr, FLIP_CUES], pf.label[tr]).predict_proba(pf.loc[va, FLIP_CUES])[:, 1]
    t["p_flip"] = t.pair_id.map(pf.set_index("pair").p_flip); t["sim_flip_x_p"] = t.sim_flip * t.p_flip
    return pf, t


def train(out):
    pairs = load_split("train"); pf, t = build_table(pairs, category_folds(pairs))
    out.mkdir(parents=True, exist_ok=True)
    pf.to_csv(out / "train_coppie.csv", index=False); t.to_csv(out / "train_candidati.csv", index=False)
    ok = pf.label >= 0
    models = dict(flip=clone(FLIP_CLF).fit(pf.loc[ok, FLIP_CUES], pf.label[ok]),     # classificatore di coppia, tutte le coppie
                  gbdt=clone(GBDT).fit(t[CUES], t.label))                           # re-ranker, tutti i candidati
    pickle.dump(models, open(out / "models.pkl", "wb"))
    print("salvati", out / "models.pkl", len(pf), "coppie", len(t), "candidati")


def cv(out):
    """KAG out-of-fold: per ogni fold (categorie mai viste) il GBDT e' allenato sugli altri quattro"""
    t = pd.read_csv(out / "train_candidati.csv"); oof = np.zeros(len(t))
    for f in range(5):
        tr, va = (t.fold != f).values, (t.fold == f).values
        oof[va] = clone(GBDT).fit(t.loc[tr, CUES], t.label[tr]).predict_proba(t.loc[va, CUES])[:, 1]
    pick = lambda score: t.loc[t.assign(_s=score).groupby("query_id")._s.idxmax(), "kag"].mean()
    print(f"KAG  argmax {pick(-t.cand_rank.values):.3f}   re-ranker {pick(oof):.3f}   "
          f"oracolo tra i {K} {pick(t.label.values * 10 + t.sim.values):.3f}")


# ================================================================== 8. predizione e submission
def predict_pair(models, P, qs, oa_s, oa_t):
    """-> {query_id: [x, y]}"""
    p_flip = float(models["flip"].predict_proba(pd.DataFrame([flip_cues(P, oa_s, oa_t)])[FLIP_CUES])[0, 1])
    out = {}
    for q in qs:
        r = pd.DataFrame(candidate_rows(P, q["sxy"])); r["p_flip"] = p_flip; r["sim_flip_x_p"] = r.sim_flip * p_flip
        j = int(np.argmax(models["gbdt"].predict_proba(r[CUES])[:, 1]))
        out[q["id"]] = [float(r.px.iloc[j]), float(r.py.iloc[j])]
    return out


def predict(out):
    models = pickle.load(open(out / "models.pkl", "rb")); pairs = load_split("test")
    images = sorted({q[k] for qs in pairs.values() for q in qs for k in ("src", "tgt")})
    oa = orient_all(images); nets = load_nets(); preds = {}; t0 = time.time()
    for n, (pid, qs) in enumerate(pairs.items()):
        preds.update(predict_pair(models, Pair(nets, qs[0]["src"], qs[0]["tgt"]), qs, oa[qs[0]["src"]], oa[qs[0]["tgt"]]))
        if n % 20 == 0: print(f"  coppia {n + 1}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
    test = pd.read_csv(DATA / "test.csv")
    pd.DataFrame({"id": test.id, "prediction": [json.dumps(preds[i]) for i in test.id]}).to_csv(out / "submission.csv", index=False)
    print("scritto", out / "submission.csv", len(test), "righe")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cmd", choices=["train", "cv", "predict"])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    {"train": train, "cv": cv, "predict": predict}[a.cmd](a.out)
