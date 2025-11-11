# fusion_pipeline_final_integrado.py
# Integração: extração de features de textura + pipeline PyTorch + ensemble
# Requer: torch, timm, torchvision, sklearn, mahotas, scikit-image, opencv-python, tqdm

import os
import sys
import numpy as np
import cv2
import mahotas
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import recall_score, multilabel_confusion_matrix
from tqdm import tqdm
sys.path.append("/Users/xr4good/Desktop/Ingrid/DIFF/")
try:
    from balanceamento import dividirEBalancearPorClasse, ler_BalanceamentoDividido, salvar_BalanceamentoDividido
except Exception as e:
    print("Aviso: não foi possível importar módulo 'balance'. Certifique-se do caminho. Erro:", e)
   

# ---------------------------------------------------------
# Configs gerais
# ---------------------------------------------------------
num_classes = 2
USE_FEATURES = True
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
epochs = 10
pasta = './Modelos_{}_classes'.format(num_classes)
os.makedirs(pasta, exist_ok=True)

# ---------------------------------------------------------
# Transform 
# ---------------------------------------------------------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((336,336)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------
# 1) Extração de features de textura 
# ---------------------------------------------------------
def extract_texture_features_bgr(img_bgr):
    """Recebe imagem BGR uint8 e retorna vetor de features (haralick + lbp + hist)."""
    if img_bgr is None:
        # vetor de zeros (tamanho conhecido: haralick(13) + lbp bins (n_points+2=10) + hist(32) = 55)
        return np.zeros(13 + (8*1 + 2) + 32, dtype=np.float32)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Haralick
    har = mahotas.features.haralick(img_gray).mean(axis=0)  # shape (13,)

    # LBP
    radius = 1
    n_points = 8 * radius
    METHOD = 'uniform'
    lbp = local_binary_pattern(img_gray, n_points, radius, METHOD)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                               range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float32") / (lbp_hist.sum() + 1e-6)  # shape (n_points+2,)

    # Histograma intensidade
    hist = cv2.calcHist([img_gray], [0], None, [32], [0, 256]).flatten().astype("float32")
    hist = hist / (hist.sum() + 1e-6)

    return np.concatenate([har.astype("float32"), lbp_hist.astype("float32"), hist.astype("float32")])

def extrair_e_salvar_features(X_train, X_val, X_test, force_reextract=False):
    """Extrai features para conjuntos (arrays de imagens BGR uint8) e salva .npy
       Se já existirem arquivos e force_reextract=False, apenas carrega.
    """
    fn_train = "features_train.npy"
    fn_val   = "features_val.npy"
    fn_test  = "features_test.npy"
    if (not force_reextract) and os.path.exists(fn_train) and os.path.exists(fn_val) and os.path.exists(fn_test):
        print("Features já existem em disco — carregando.")
        return np.load(fn_train), np.load(fn_val), np.load(fn_test)

    print("Extraindo features de textura (isso pode demorar)...")
    feats_train = np.array([extract_texture_features_bgr(img) for img in tqdm(X_train, desc="train")])
    feats_val   = np.array([extract_texture_features_bgr(img) for img in tqdm(X_val, desc="val")])
    feats_test  = np.array([extract_texture_features_bgr(img) for img in tqdm(X_test, desc="test")])

    # Normalização global 
    scaler = StandardScaler()
    feats_train = scaler.fit_transform(feats_train)
    feats_val   = scaler.transform(feats_val)
    feats_test  = scaler.transform(feats_test)

    np.save(fn_train, feats_train)
    np.save(fn_val, feats_val)
    np.save(fn_test, feats_test)


    print("Features salvas:", fn_train, fn_val, fn_test)
    return feats_train, feats_val, feats_test

# ---------------------------------------------------------
# 2) Dataset de fusão (imagem + features de textura)
# ---------------------------------------------------------
class FusionDataset(Dataset):
    def __init__(self, imagens, rotulos, texturas, transform=None):
        self.imagens = imagens
        self.rotulos = np.array(rotulos).astype(np.int64)
        self.texturas = np.array(texturas).astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, idx):
        img = self.imagens[idx]
        tex = self.texturas[idx]
        y = self.rotulos[idx]

        if self.transform:
            img = self.transform(img)

        tex = torch.tensor(tex, dtype=torch.float32)
        return img, tex, torch.tensor(y, dtype=torch.long)

def criar_dataloaders_fusion(X_tr, y_tr, F_tr, X_val, y_val, F_val, X_te, y_te, F_te, batch_size=16):
    train = FusionDataset(X_tr, y_tr, F_tr, transform=transform)
    val   = FusionDataset(X_val, y_val, F_val, transform=transform)
    test  = FusionDataset(X_te, y_te, F_te, transform=transform)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False)
    )

# ---------------------------------------------------------
# 3) Construir backbone (timm) para fusion
# ---------------------------------------------------------
MODELOS_MAP = {
    1: "efficientvit_b2.r224_in1k",
    2: "edgenext_base",
    3: "eva02_small_patch14_336.mim_in22k_ft_in1k",
    4: "efficientnet_b0",
    5: "efficientnet_b1",
    6: "efficientnet_b2",
    7: "efficientnet_b3",
    8: "efficientnet_b4",
    9: "efficientnet_b5",
    10: "efficientnet_b6",
    11: "mobilenetv2_100",
    12: "xception",
    13: "inception_v3",
    14: "tf_efficientnetv2_s.in21k_ft_in1k"
}

def construir_backbone(num_classificador):
    if num_classificador not in MODELOS_MAP:
        raise ValueError(f"Classificador {num_classificador} não mapeado.")
    name = MODELOS_MAP[num_classificador]
    try:
        backbone = timm.create_model(name, pretrained=True, num_classes=0)  # sem cabeça
    except Exception as e:
        print(f"warning: timm.create_model({name}) falhou: {e}. Tentando sem num_classes.")
        backbone = timm.create_model(name, pretrained=True)
        # tentar remover classifier se existir
        try:
            if hasattr(backbone, 'fc'):
                backbone.fc = nn.Identity()
            elif hasattr(backbone, 'classifier'):
                backbone.classifier = nn.Identity()
        except Exception:
            pass

    # tenta inferir num_features
    if hasattr(backbone, 'num_features') and backbone.num_features is not None:
        num_features = backbone.num_features
    else:
        # heurística: passa um batch dummy para obter dimensão de saída
        backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros((1,3,336,336))
            try:
                out = backbone.forward_features(dummy) if hasattr(backbone, 'forward_features') else backbone(dummy)
            except Exception:
                # fallback: tentar forward e flatten
                out = backbone(dummy)
            if out.ndim > 2:
                out = out.mean(dim=(2,3))
            num_features = out.shape[1]
    return backbone, num_features

# ---------------------------------------------------------
# 4) Modelo de fusão
# ---------------------------------------------------------
class FusionModel(nn.Module):
    def __init__(self, backbone, backbone_feat_dim, feat_dim, num_classes, use_features=True):
        super().__init__()
        self.backbone = backbone
        self.use_features = use_features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # descobrir dim. do backbone
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 336, 336)
            if hasattr(self.backbone, "forward_features"):
                x = self.backbone.forward_features(dummy)
            else:
                x = self.backbone(dummy)
            if x.ndim > 2:
                x = self.pool(x).flatten(1)
            backbone_feat_dim = x.shape[1]

        # classificador
        if self.use_features:
            self.classifier = nn.Linear(backbone_feat_dim + feat_dim, num_classes)
        else:
            self.classifier = nn.Linear(backbone_feat_dim, num_classes)

    def forward(self, x, tex):
        if hasattr(self.backbone, 'forward_features'):
            x = self.backbone.forward_features(x)
        else:
            x = self.backbone(x)

        if x.ndim > 2:
            x = self.pool(x).flatten(1)

        if self.use_features and tex is not None:
            x = torch.cat([x, tex], dim=1)

        return self.classifier(x)



# ---------------------------------------------------------
# 5) Funções de treino/val/predict (adaptadas para fusion)
# ---------------------------------------------------------
def treinar(model, train_loader, val_loader, epochs, lr, peso_path=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    melhor_recall = -1.0
    melhor_peso_path = None

    for epoch in range(epochs):
        model.train()
        tot_loss = 0.0
        n_samples = 0
        for x_img, x_tex, y in train_loader:
            x_img = x_img.to(device)
            x_tex = x_tex.to(device)
            y = y.to(device).long()
            optimizer.zero_grad()
            logits = model(x_img, x_tex if USE_FEATURES else None)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bs = x_img.size(0)
            tot_loss += loss.item() * bs
            n_samples += bs
        if n_samples > 0:
            tot_loss /= n_samples

        # avaliação
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for x_img, x_tex, y in val_loader:
                x_img = x_img.to(device)
                x_tex = x_tex.to(device)
                logits = model(x_img, x_tex if USE_FEATURES else None)
                preds_batch = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(preds_batch)
                trues.extend(y.numpy().astype(int))

        if len(trues) == 0:
            recall_macro = 0.0
        else:
            recall_macro = recall_score(trues, preds, average='macro')

        print(f"Epoch {epoch+1}/{epochs} - loss: {tot_loss:.4f} - val_recall_macro: {recall_macro:.4f}")

        # salvar melhor
        if recall_macro > melhor_recall:
            melhor_recall = recall_macro
            if peso_path:
                torch.save(model.state_dict(), peso_path)
                melhor_peso_path = peso_path

    return melhor_recall, melhor_peso_path

def prever(model, loader):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for x_img, x_tex, _ in loader:
            x_img = x_img.to(device)
            x_tex = x_tex.to(device)
            logits = model(x_img, x_tex if USE_FEATURES else None)
            preds_batch = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(preds_batch)
    return np.array(preds, dtype=int)

# ---------------------------------------------------------
# 6) Métricas / impressão 
# ---------------------------------------------------------
def imprimirResultado(mcm):
    tn = np.mean(mcm[:,0,0])
    tp = np.mean(mcm[:,1,1])
    fn = np.mean(mcm[:,0,1])
    fp = np.mean(mcm[:,1,0])

    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12) if (prec + rec) > 0 else 0.0
    acuracia = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    especificidade = tn / (tn + fp + 1e-12)

    print(f"Precisão:        {prec*100:.6f}%")
    print(f"Revocação:       {rec*100:.6f}%")
    print(f"F1:              {f1*100:.6f}%")
    print(f"Acurácia:        {acuracia*100:.6f}%")
    print(f"Especificidade:  {especificidade*100:.6f}%")


# ---------------------------------------------------------
# 7) Fluxo principal
# ---------------------------------------------------------
if __name__ == "__main__":
    # garante balance
    try:
        salvar_BalanceamentoDividido(num_classes)
    except Exception as e:
        print("Aviso ao chamar salvar_BalanceamentoDividido:", e)

    # Ler dados: espera que ler_BalanceamentoDividido retorne X_train, y_train, X_val, y_val, X_test, y_test
    try:
        X_treinamento, y_treinamento, X_validacao, y_validacao, X_teste, y_teste = ler_BalanceamentoDividido(num_classes)
    except Exception as e:
        raise RuntimeError("Não foi possível carregar dados com ler_BalanceamentoDividido. Erro: " + str(e))

    # Padroniza imagens (garante 3 canais e tamanho base; seus dados originais eram 90x90 talvez)
    def padronizar_imagens(lista_imgs, target_shape=(90,90,3)):
        lista_corrigida = []
        for i, img in enumerate(lista_imgs):
            if img is None:
                print(f"Imagem {i} era None, substituindo por zeros")
                img = np.zeros(target_shape, dtype=np.uint8)
            else:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] != 3:
                    img = img[:, :, :3]
                if img.shape != target_shape:
                    img = cv2.resize(img, (target_shape[1], target_shape[0]))
            lista_corrigida.append(img.astype(np.uint8))
        return np.array(lista_corrigida, dtype=np.uint8)

    X_treinamento = padronizar_imagens(X_treinamento)
    X_validacao   = padronizar_imagens(X_validacao)
    X_teste       = padronizar_imagens(X_teste)

    # Extrair e salvar features (ou carregar se já existir)
    F_train, F_val, F_test = extrair_e_salvar_features(X_treinamento, X_validacao, X_teste, force_reextract=False)

    # Criar dataloaders de fusão
    train_loader, val_loader, test_loader = criar_dataloaders_fusion(
        X_treinamento, y_treinamento, F_train,
        X_validacao, y_validacao, F_val,
        X_teste, y_teste, F_test,
        batch_size=batch_size
    )

    # ---- Construção e treinamento/carregamento dos 3 modelos do ensemble ----
    # Para cada modelo, construiremos backbone + FusionModel (mesma lógica de seleção de modelos do seu script original)
    y_pred_list = []
    recalls = []

    # escolha dos IDs de classificador para cada slot
    if num_classes == 6:
        slot_map = [14, 2, 3]  # primeiro, segundo, terceiro
        pesos = [
            os.path.join(pasta, 'best_model_efficientnetv2_6classes_fusion.pth'),
            os.path.join(pasta, 'best_model_edgenext_6classes_fusion.pth'),
            os.path.join(pasta, 'best_model_eva02_6classes_fusion.pth'),
        ]
    elif num_classes == 3:
        slot_map = [14, 2, 3]
        pesos = [
            os.path.join(pasta, 'best_model_efficientnetv2_3classes_fusion.pth'),
            os.path.join(pasta, 'best_model_edgenext_3classes_fusion.pth'),
            os.path.join(pasta, 'best_model_eva02_3classes_fusion.pth'),
        ]
    elif num_classes == 2:
        slot_map = [14, 2, 3]
        pesos = [
            os.path.join(pasta, 'best_model_efficientnetv2_2classes_fusion.pth'),
            os.path.join(pasta, 'best_model_edgenext_2classes_fusion.pth'),
            os.path.join(pasta, 'best_model_eva02_2classes_fusion.pth'),
        ]
    else:
        raise NotImplementedError("Fluxo só implementado para num_classes == 2, 3 ou 6")

    feat_dim = F_train.shape[1]

    for idx, model_id in enumerate(slot_map):
        print(f"\n--- Preparando modelo {idx+1} (id timm {model_id}) ---")
        backbone, backbone_feat_dim = construir_backbone(model_id)
        model = FusionModel(backbone, backbone_feat_dim, feat_dim, num_classes, use_features=USE_FEATURES)
        peso_path = pesos[idx]

        if os.path.exists(peso_path):
            try:
                model.load_state_dict(torch.load(peso_path, map_location=device))
                print("Pesos carregados de", peso_path)
            except Exception as e:
                print("Falha ao carregar pesos:", e)
                print("Será treinado do zero.")
                treinar(model, train_loader, val_loader, epochs=epochs, lr=1e-4, peso_path=peso_path)
        else:
            print("Treinando modelo e salvando em", peso_path)
            treinar(model, train_loader, val_loader, epochs=epochs, lr=1e-4, peso_path=peso_path)

        preds = prever(model, test_loader)
        y_pred_list.append(preds)

        # calcula recall para logging
        # ajusta true labels
        true_labels = np.array([int(x) for x in np.array([np.argmax(lbl) if hasattr(lbl, "__len__") and len(lbl) > 1 else lbl for lbl in y_teste])])
        if true_labels.ndim > 1:
            true_labels = np.argmax(true_labels, axis=1)
        rec = recall_score(true_labels, preds, average='macro')
        recalls.append(rec)
        print(f"Recall do modelo {idx+1}: {rec:.4f}")

    # Desempate usa o melhor recall entre os 3
    melhor = int(np.argmax(recalls))
    print("\nRecalls por modelo:", recalls)
    print("Modelo com maior recall (índice):", melhor)

    # Votação por amostra
    true_labels = np.array([int(x) for x in np.array([np.argmax(lbl) if hasattr(lbl, "__len__") and len(lbl) > 1 else lbl for lbl in y_teste])])
    if true_labels.ndim > 1:
        true_labels = np.argmax(true_labels, axis=1)

    n_samples = len(true_labels)
    y_pred_ensemble = []
    for i in range(n_samples):
        votos = np.zeros(num_classes, dtype=int)
        for preds in y_pred_list:
            votos[preds[i]] += 1
        candidatos = np.where(votos == votos.max())[0]
        if len(candidatos) == 1:
            y_pred_ensemble.append(candidatos[0])
        else:
            # desempate pelo modelo com maior recall
            desempate = y_pred_list[melhor][i]
            y_pred_ensemble.append(int(desempate))

    y_pred_ensemble = np.array(y_pred_ensemble, dtype=int)

    # Matriz de confusão multilabel e métricas
    mcm = multilabel_confusion_matrix(true_labels, y_pred_ensemble)
    imprimirResultado(mcm)

    # Salva predições do ensemble
    np.save(os.path.join(pasta, "y_pred_ensemble.npy"), y_pred_ensemble)
    print("Predições do ensemble salvas em", os.path.join(pasta, "y_pred_ensemble.npy"))

    # ==== CONFUSION MATRIX COM LABELS ====
    if num_classes == 2:
        class_names = ["Positivo", "Negativo"] # label 0, label 1
    
    elif num_classes == 3:
        class_names = [
            "ASCH/CA/HSIL", # doenca label 0
            "ASCUS/LSIL", # limitrofe label 1
            "Normal" # label 2
        ]
    
    elif num_classes == 6:
        class_names = ["ASCH", "ASCUS", "CA", "HSIL", "LSIL", "Normal"]

    cm = confusion_matrix(true_labels, y_pred_ensemble)

    plt.figure(figsize=(7,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(values_format='d', cmap="Blues")
    plt.title("Matriz de Confusão - Ensemble")

    # Salvar
    plt.savefig(os.path.join(pasta, "matriz_confusao.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # ==== ROC CURVE ====
    y_true_bin = label_binarize(true_labels, classes=list(range(num_classes)))
    y_pred_bin = label_binarize(y_pred_ensemble, classes=list(range(num_classes)))
    
    plt.figure(figsize=(8,6))
    
    if num_classes == 2:
        # Em binário, só há uma coluna → usar ela
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[0]} (AUC = {roc_auc:.3f})")  # classe positiva
    else:
        # Multi-classe
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")


    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC - Ensemble")
    plt.legend()
    plt.grid(True)

    #  Salvar
    plt.savefig(os.path.join(pasta, "curva_ROC.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # ==== PRECISION–RECALL CURVE ====
    plt.figure(figsize=(8,6))
    
    if num_classes == 2:
        precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_pred_bin.ravel())
        plt.plot(recall, precision, label=f"{class_names[0]}") # classe positiva
    else:
        # Multi-classe
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_pred_bin[:, i]
            )
            plt.plot(recall, precision, label=f"{class_names[i]}")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision–Recall (PR) - Ensemble")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(pasta, "curva_PR.png"), dpi=300, bbox_inches='tight')
    plt.show()



    
