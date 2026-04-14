# balanceamento_equalizado_verbose_v4.py
# Pipeline de balanceamento (limite 10x por imagem)
#
# Regras dirigidas:
# - n=2:
#     ASCUS,LSIL,ASCH,HSIL,CA = 1122 | Normal = 5610
# - n=3:
#     Baixo Grau: ASCUS=2000, LSIL=2000
#     Alto Grau: CA=1122, HSIL=1333, ASCH=1333
#     Normal: 4000
# - n=6:
#     Todas = 1122

import os
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------------------
# CONFIG
# ----------------------------
ROOT = r"E:\datasets\imagens\6classes"
OUT_BASE = r"..\Base balanceada dividida novo - TESTE"
TAMANHO = (90, 90)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CLASS_NAMES_6 = ["ASCH", "ASCUS", "CA", "HSIL", "LSIL", "Normal"]
ORIG_TO_DISPLAY = {"Normal":"NILM","ASCUS":"ASC-US","LSIL":"LSIL","ASCH":"ASC-H","HSIL":"HSIL","CA":"SCC"}
COL_ORDER = ["Normal","ASCUS","LSIL","ASCH","HSIL","CA"]

# ============================================================
# Utils 
# ============================================================

def garantir_imagem_valida_bgr(path_or_img, tamanho=TAMANHO):
    if isinstance(path_or_img, str):
        img = cv2.imread(path_or_img)
    else:
        img = path_or_img
    if img is None:
        img = np.zeros((tamanho[0], tamanho[1], 3), dtype=np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] != 3:
        img = img[:, :, :3]
    if img.shape[:2] != tamanho:
        img = cv2.resize(img, (tamanho[1], tamanho[0]))
    return img.astype(np.uint8)


def gerar_augmentations_para_classe_com_origem(images_with_names, needed, ops=list(range(1, 11))):
    if needed <= 0:
        return []
    augmented = []
    n = len(images_with_names)
    idxs = list(range(n))
    random.shuffle(idxs)
    i = 0
    op_i = 0
    while len(augmented) < needed:
        base_idx = idxs[i % n]
        base_img, base_orig = images_with_names[base_idx]
        op = ops[op_i % len(ops)]
        aug = operacaoAugmentation_retornar(op, base_img)
        augmented.append((aug, base_orig))
        i += 1
        op_i += 1
    return augmented


def detectar_subclasse_do_caminho(path):
    p = path.replace("/", os.sep)
    for cname in CLASS_NAMES_6:
        if os.sep + cname + os.sep in p:
            return cname
    return os.path.basename(os.path.dirname(p))

# ------------------------------------------------------------
# Augmentations (paper-like)
# ------------------------------------------------------------
def operacaoAugmentation_retornar(operacao, img):
    img = garantir_imagem_valida_bgr(img)
    if operacao == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif operacao == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif operacao == 3:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif operacao == 4:
        return cv2.flip(img, 1)
    elif operacao == 5:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1)
    elif operacao == 6:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_180), 1)
    elif operacao == 7:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    elif operacao == 8:
        sigma = 0.05
        nova = np.clip(random_noise(img, var=sigma**2) * 255, 0, 255)
        return nova.astype(np.uint8)
    elif operacao == 9:
        sigma = 0.005
        noisy = random_noise(img, var=sigma**2)
        nova = np.clip(denoise_tv_chambolle(noisy, weight=0.05, channel_axis=-1) * 255, 0, 255)
        return nova.astype(np.uint8)
    elif operacao == 10:
        sigma = 0.005
        noisy = random_noise(img, var=sigma**2)
        nova = np.clip(denoise_bilateral(noisy, sigma_color=0.01, sigma_spatial=5, channel_axis=-1) * 255, 0, 255)
        return nova.astype(np.uint8)
    return img

# ------------------------------------------------------------
# Split (fixo) - usamos test_size=0.20 do total e val_fraction=0.20 do restante
# ------------------------------------------------------------

def split_por_subclasse(all_files, test_size=0.20, val_fraction_within_train=0.20):
    """Retorna train_by_orig, val_by_orig, test_by_orig (dicionários de listas de paths)"""
    Xtr_raw, Xval_raw, Xte_raw = [], [], []
    Ytr_raw, Yval_raw, Yte_raw = [], [], []
    for cname, files in all_files.items():
        if len(files) == 0:
            continue
        labels = [0] * len(files)  # stratify exige rótulo, mas usamos dummy pois fazemos por subclasse
        # separa Test (test_size do total)
        files_trval, files_te = train_test_split(files, test_size=test_size, random_state=SEED)
        # separa Validation dentro do restante
        if len(files_trval) == 0:
            files_tr, files_val = [], []
        else:
            val_size_fraction = val_fraction_within_train
            files_tr, files_val = train_test_split(files_trval, test_size=val_size_fraction, random_state=SEED)
        Xtr_raw += files_tr
        Xval_raw += files_val
        Xte_raw  += files_te
    train_by_orig = defaultdict(list)
    val_by_orig   = defaultdict(list)
    test_by_orig  = defaultdict(list)
    for p in Xtr_raw: train_by_orig[detectar_subclasse_do_caminho(p)].append(p)
    for p in Xval_raw: val_by_orig[detectar_subclasse_do_caminho(p)].append(p)
    for p in Xte_raw:  test_by_orig[detectar_subclasse_do_caminho(p)].append(p)
    return train_by_orig, val_by_orig, test_by_orig


# ------------------------------------------------------------
# Mapeamento de labels por nº de classes
# ------------------------------------------------------------
def configurar_classes(n_classes):
    if n_classes == 6:
        return CLASS_NAMES_6, {"ASCH":0,"ASCUS":1,"CA":2,"HSIL":3,"LSIL":4,"Normal":5}
    elif n_classes == 3:
        map3 = {"LSIL":0, "ASCUS":0,
                "HSIL":1, "ASCH":1, "CA":1,
                "Normal":2}
        return list(map3.keys()), map3
    elif n_classes == 2:
        map2 = {"ASCH":0,"ASCUS":0,"CA":0,"HSIL":0,"LSIL":0,"Normal":1}
        return list(map2.keys()), map2
    else:
        raise ValueError("n_classes_balanceamento deve ser 2, 3 ou 6")
        
# ============================================================
# EQUALIZAÇãO
# ============================================================

def aplicar_equalizacao(n_classes_balanceamento, folder_class_names, label_map,
                        train_by_orig, Xtr_final, Ytr_final,
                        counts_train_orig_post, manifesto_rows):

    # --------------------------------------------------
    # DEFINIÇÃO DOS TARGETS (DIRIGIDOS)
    # --------------------------------------------------
    if n_classes_balanceamento == 2:
        targets = {
            "ASCUS":1122, "LSIL":1122, "ASCH":1122,
            "HSIL":1122, "CA":1122,
            "Normal":5610
        }

    elif n_classes_balanceamento == 3:
        targets = {
            # Baixo Grau
            "ASCUS":2000,
            "LSIL":2000,

            # Alto Grau
            "CA":1122,
            "HSIL":1333,
            "ASCH":1333,

            # Normal
            "Normal":4000
        }

    elif n_classes_balanceamento == 6:
        targets = {
            "ASCUS":1122,
            "LSIL":1122,
            "ASCH":1122,
            "HSIL":1122,
            "CA":1122,
            "Normal":1122
        }

    else:
        raise ValueError("n_classes_balanceamento inválido")

    print(f"\n[CONFIG] Balanceamento dirigido ({n_classes_balanceamento} classes)")
    for k,v in targets.items():
        print(f"  {k}: {v}")
    print("")

    # --------------------------------------------------
    # APLICAÇÃO DIRETA (sem divisão automática)
    # --------------------------------------------------
    for orig in folder_class_names:
        apply_train(
            orig,
            label_map[orig],
            targets.get(orig, 0),
            train_by_orig,
            Xtr_final,
            Ytr_final,
            counts_train_orig_post,
            manifesto_rows
        )

# ------------------------------------------------------------
# apply_train (mantido)
# ------------------------------------------------------------

def ajustar_target_pos_limite(orig, lab, X_final, Y_final, counts_post, manifesto_rows, desired):
    current = counts_post[orig]
    if current > desired:
        excess = current - desired
        print(f"[POS-LIMITE] {orig}: atingiu {current} > {desired}. Reduzindo por downsample ({excess} imagens).")
        idxs = [i for i,lbl in enumerate(Y_final) if lbl == lab]
        if len(idxs) <= excess:
            remove_set = set(idxs)
        else:
            remove_set = set(random.sample(idxs, excess))
        X_final[:] = [im for i,im in enumerate(X_final) if i not in remove_set]
        Y_final[:] = [lbl for i,lbl in enumerate(Y_final) if i not in remove_set]
        manifesto_rows[:] = [r for i,r in enumerate(manifesto_rows) if i not in remove_set]
        counts_post[orig] = desired
        print(f"[POS-LIMITE] {orig}: agora {counts_post[orig]} imagens (após downsample).")
    elif current < desired:
        print(f"[POS-LIMITE] {orig}: só conseguiu {current}, faltam {desired-current}. Não é possível aumentar (limite 10x).")


def apply_train(orig, lab, desired_target, train_by_orig, X_final, Y_final, counts_post, manifesto_rows):
    paths = train_by_orig.get(orig, [])
    cur = len(paths)
    target = int(max(0, desired_target))
    if target <= 0:
        print(f"[WARN] target para {orig} <= 0, pulando")
        return
    originals = [garantir_imagem_valida_bgr(p) for p in paths]
    originals_with_names = [(im, p) for im,p in zip(originals, paths)]

    if cur < target:
        need = target - cur
        print(f"[ACTION] {orig}: cur={cur} -> need augmentation={need} to reach target={target}")
        for p,im in zip(paths, originals):
            X_final.append(im); Y_final.append(lab)
            counts_post[orig] += 1
            manifesto_rows.append({"set":"Training","orig_subclass":orig,"mapped_label":lab,"generated":0,"arquivo_origem":p})
        augmented = gerar_augmentations_para_classe_com_origem(originals_with_names, need)
        for im,_ in augmented:
            X_final.append(im); Y_final.append(lab)
            counts_post[orig] += 1
            manifesto_rows.append({"set":"Training","orig_subclass":orig,"mapped_label":lab,"generated":1,"arquivo_origem":""})
        ajustar_target_pos_limite(orig, lab, X_final, Y_final, counts_post, manifesto_rows, target)
    elif cur > target:
        print(f"[ACTION] {orig}: cur={cur} -> downsample to target={target}")
        idxs = random.sample(range(cur), target)
        for idx in idxs:
            p = paths[idx]
            im = garantir_imagem_valida_bgr(p)
            X_final.append(im); Y_final.append(lab)
            counts_post[orig] += 1
            manifesto_rows.append({"set":"Training","orig_subclass":orig,"mapped_label":lab,"generated":0,"arquivo_origem":p})
    else:
        print(f"[ACTION] {orig}: cur==target=={target}, mantendo originais")
        for p in paths:
            im = garantir_imagem_valida_bgr(p)
            X_final.append(im); Y_final.append(lab)
            counts_post[orig] += 1
            manifesto_rows.append({"set":"Training","orig_subclass":orig,"mapped_label":lab,"generated":0,"arquivo_origem":p})

# ------------------------------------------------------------
# apply_eval, tabelas, salvar (mantidos) - copia das versões anteriores
# ------------------------------------------------------------

def apply_eval(set_name, by_orig, desired_map, X_out, Y_out, manifesto_rows, label_map):
    out_imgs, out_labs = [], []
    for orig in CLASS_NAMES_6:
        if orig not in desired_map:
            continue
        paths = by_orig.get(orig, [])
        cur = len(paths)
        target = min(cur, int(desired_map[orig]))
        if target <= 0:
            continue
        idxs = random.sample(range(cur), target) if cur > target else list(range(cur))
        for idx in idxs:
            p = paths[idx]
            img = garantir_imagem_valida_bgr(p)
            out_imgs.append(img); out_labs.append(label_map[orig])
            manifesto_rows.append({"set":set_name,"orig_subclass":orig,"mapped_label":label_map[orig],"generated":0,"arquivo_origem":p})
    X_out.extend(out_imgs); Y_out.extend(out_labs)


def _df_tabela_por_manifesto(manifesto_df):
    rows = {}
    for set_name in ['Training','Testing','Validation']:
        subset = manifesto_df[manifesto_df['set'] == set_name]
        contagem = subset.groupby('orig_subclass').size().to_dict()
        linha = []; total = 0
        for orig in COL_ORDER:
            v = contagem.get(orig,0); linha.append(v); total += v
        linha.append(total); rows[set_name] = linha
    total_row = [rows['Training'][i] + rows['Testing'][i] + rows['Validation'][i] for i in range(len(rows['Training']))]
    rows['Total'] = total_row
    display_cols = [ORIG_TO_DISPLAY[o] for o in COL_ORDER] + ['Total']
    df = pd.DataFrame.from_dict(rows, orient='index', columns=display_cols)
    return df


def salvar_tabela_jpeg(df, out_path_jpeg, dpi=220, pad=0.8, font_size=9, titulo=None):
    n_rows, n_cols = df.shape
    width = max(8, n_cols*1.0); height = max(3, n_rows*0.75)
    fig, ax = plt.subplots(figsize=(width,height), dpi=dpi)
    ax.axis('off')
    if titulo: ax.set_title(titulo, fontsize=12, weight='bold', pad=10)
    tbl = ax.table(cellText=df.values, rowLabels=df.index.tolist(), colLabels=df.columns.tolist(), loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(font_size); tbl.scale(1.0,1.2)
    plt.tight_layout(pad=pad); fig.savefig(out_path_jpeg, format='jpeg', dpi=dpi, bbox_inches='tight'); plt.close(fig)


def _base_dirs(num_classes):
    base = os.path.join(OUT_BASE, f"{num_classes} classes")
    dir_treino = os.path.join(base, 'Treino'); dir_val = os.path.join(base, 'Validacao'); dir_teste = os.path.join(base, 'Teste')
    return base, dir_treino, dir_val, dir_teste


def _tem_imagens(pasta):
    return os.path.isdir(pasta) and any(f.lower().endswith(('.png','.jpg','.jpeg')) for f in os.listdir(pasta))


def _manifesto_path(base):
    return os.path.join(base, 'manifesto.csv')


def _salvar_manifesto_csv(manifesto_rows, base, nomes_salvos):
    df = pd.DataFrame(manifesto_rows).copy()
    if len(df) != len(nomes_salvos):
        print(f"[Aviso] manifesto_rows ({len(df)}) != nomes_salvos ({len(nomes_salvos)}). Ajustando pelo menor...")
        m = min(len(df), len(nomes_salvos)); df = df.iloc[:m].copy(); nomes_salvos = nomes_salvos[:m]
    df['arquivo_saida'] = nomes_salvos
    out_csv = _manifesto_path(base); df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"[OK] Manifesto salvo: {out_csv}")
    return out_csv


def _gerar_tabela_de_manifesto(base, num_classes):
    csv = _manifesto_path(base)
    if not os.path.isfile(csv): raise FileNotFoundError(f"Manifesto não encontrado: {csv}")
    man = pd.read_csv(csv); df = _df_tabela_por_manifesto(man)
    out_jpeg = os.path.join(base, f"tabela_{num_classes}_classes.jpeg"); salvar_tabela_jpeg(df, out_jpeg, titulo=f"Tabela ({num_classes} classes)")
    print(f"[OK] Tabela JPEG gerada de manifesto: {out_jpeg}")
    return df

# ------------------------------------------------------------
# Core pipeline
# ------------------------------------------------------------

def dividirEBalancearPorClasse(n_classes_balanceamento, pasta_ler=ROOT, test_size=0.20, val_fraction_within_train=0.20):
    folder_class_names, label_map = configurar_classes(n_classes_balanceamento)
    all_files = {c:[] for c in folder_class_names}; total_por_subclasse = {}
    for class_name in CLASS_NAMES_6:
        folder = os.path.join(pasta_ler, class_name)
        if not os.path.isdir(folder): continue
        imgs = sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        if class_name in folder_class_names:
            all_files[class_name] = imgs; total_por_subclasse[class_name] = len(imgs)
    print('[LOG] Totais por subclasse (bruto):')
    for k,v in total_por_subclasse.items(): print(f'  {k}: {v}')

    train_by_orig, val_by_orig, test_by_orig = split_por_subclasse(all_files, test_size=test_size, val_fraction_within_train=val_fraction_within_train)
    print('[LOG] Totais por subclasse após split (TRAIN / VAL / TEST):')
    for c in folder_class_names:
        print(f'  {c}: train={len(train_by_orig.get(c,[]))}, val={len(val_by_orig.get(c,[]))}, test={len(test_by_orig.get(c,[]))}')

    Xtr_final, Ytr_final = [], []
    counts_train_orig_post = Counter(); manifesto_rows = []
    aplicar_equalizacao(n_classes_balanceamento, folder_class_names, label_map, train_by_orig, Xtr_final, Ytr_final, counts_train_orig_post, manifesto_rows)

    profile_val = {c: len(val_by_orig.get(c,[])) for c in folder_class_names}
    profile_test = {c: len(test_by_orig.get(c,[])) for c in folder_class_names}
    Xval_final, Yval_final = [], []
    apply_eval('Validation', val_by_orig, profile_val, Xval_final, Yval_final, manifesto_rows, label_map)
    Xte_final, Yte_final = [], []
    apply_eval('Testing', test_by_orig, profile_test, Xte_final, Yte_final, manifesto_rows, label_map)

    stats = {'train_by_orig_after_balance': counts_train_orig_post, 'val_by_orig': Counter({k: len(val_by_orig.get(k,[])) for k in folder_class_names}), 'test_by_orig': Counter({k: len(test_by_orig.get(k,[])) for k in folder_class_names})}
    return Xtr_final, Ytr_final, Xval_final, Yval_final, Xte_final, Yte_final, stats, manifesto_rows

# ------------------------------------------------------------
# salvar_BalanceamentoDividido
# ------------------------------------------------------------

def salvar_BalanceamentoDividido(num_classes, force=False):
    base, dir_treino, dir_val, dir_teste = _base_dirs(num_classes)
    os.makedirs(base, exist_ok=True); os.makedirs(dir_treino, exist_ok=True); os.makedirs(dir_val, exist_ok=True); os.makedirs(dir_teste, exist_ok=True)
    manifesto_csv = _manifesto_path(base)
    base_existe = _tem_imagens(dir_treino) and _tem_imagens(dir_val) and _tem_imagens(dir_teste) and os.path.isfile(manifesto_csv)
    if base_existe and not force:
        print(f"\n[Info] Base já existente para {num_classes} classes em: {base}")
        print("[Info] Regerando apenas a tabela .jpeg a partir do manifesto...")
        return _gerar_tabela_de_manifesto(base, num_classes)

    Xtr, Ytr, Xval, Yval, Xte, Yte, stats, manifesto_rows = dividirEBalancearPorClasse(num_classes)
    nomes_salvos = []
    for i, im in enumerate(Xtr): fname = f"{i}_{Ytr[i]}.png"; fpath = os.path.join(dir_treino, fname); cv2.imwrite(fpath, garantir_imagem_valida_bgr(im)); nomes_salvos.append(os.path.join('Treino', fname))
    for i, im in enumerate(Xval): fname = f"{i}_{Yval[i]}.png"; fpath = os.path.join(dir_val, fname); cv2.imwrite(fpath, garantir_imagem_valida_bgr(im)); nomes_salvos.append(os.path.join('Validacao', fname))
    for i, im in enumerate(Xte): fname = f"{i}_{Yte[i]}.png"; fpath = os.path.join(dir_teste, fname); cv2.imwrite(fpath, garantir_imagem_valida_bgr(im)); nomes_salvos.append(os.path.join('Teste', fname))
    print(f"\n[OK] Imagens salvas em: {base}")
    _salvar_manifesto_csv(manifesto_rows, base, nomes_salvos)
    df = _gerar_tabela_de_manifesto(base, num_classes)
    return df

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == '__main__':
    for n in [2,3,6]:
        print('\n' + '='*80)
        print(f"INICIANDO PIPELINE PARA {n} CLASSES - {datetime.now().isoformat()}")
        print('='*80)
        df = salvar_BalanceamentoDividido(n, force=False)
        print(df)

