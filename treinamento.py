# ============================================================
# CRIC – Pap Smear Classification
# Fine-tuning Progressivo + EMA
# Aprimorado: LLRD, Mixup/CutMix, Cosine+Warmup, TTA, Steps matching
# ============================================================

import os, sys, gc, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from typing import Iterable


# ===== geracao de matriz de confusao e curvas ROC/PR
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ===== timm =====
import timm
from timm.utils import ModelEmaV2
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

# ============================================================
# ---------------- CONFIGURAÇÕES GERAIS -----------------------
# ============================================================
num_classes = 2
if num_classes == 2:
    class_names = ['Alterada','Normal']
elif num_classes == 3:
    class_names = ['Low','High','Normal']
else:
    class_names = ['ASCH','ASCUS','CA','HSIL','LSIL','Normal']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    torch.backends.cudnn.benchmark = True

# ---------------- MODELOS PARA TESTE ---------------- itera todos os modelos durante execução
BACKBONES = [
    # CNNs puras
    'tf_efficientnet_b4.ns_jft_in1k',
    'tf_efficientnetv2_s.in21k_ft_in1k',
    # CNN moderna
    'convnextv2_large',
    'edgenext_base',
    # Híbridos
    'coatnet_3_rw_224',
    'maxxvitv2_rmlp_base_rw_224',
    # Transformers puros
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    'deit_base_patch16_224.fb_in1k',
    'eva02_small_patch14_224',
    'swin_base_patch4_window7_224',
]
PRETRAINED = True

# ---------- Seeds / determinismo ----------
def set_seeds(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds(42, deterministic=False)

# ---------- Hiperparâmetros globais ----------
WEIGHT_DECAY = 1e-4

# Fases do FT progressivo
LR_HEAD_PHASE1 = 1e-4
LR_BACKBONE_PHASE1 = 0.0  # backbone totalmente congelado
LR_HEAD_PHASE2 = 5e-5
LR_BACKBONE_PHASE2 = 5e-6

# Ciclo / paradaf
EPOCHS_MAX = 100                  # valor teto, mas usaremos equalização por steps
EARLY_STOP_PATIENCE = 7
MIN_EPOCHS_BEFORE_STOP = 20      # evita parar cedo demais na fase 2

# Loss / regularização
LABEL_SMOOTHING = 0.05
USE_EMA = True
EMA_DECAY = 0.999
GRAD_CLIP_NORM = 1.0

# Mixup/CutMix (leve)
USE_MIXUP = True
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 0.1
MIXUP_PROB = 0.5

# Scheduler
USE_WARMUP_COSINE = True
WARMUP_EPOCHS = 2

# LLRD para ViTs na fase 2
USE_LLRD = True
LAYER_DECAY = 0.75  # 0.65–0.85 típico

# Amostragem balanceada (opcional)
USE_WEIGHTED_SAMPLER = False

# TTA no teste
USE_TTA = True

# Equalização de orçamento: mesmo nº de optimizer steps por fase
TARGET_STEPS_PER_PHASE = 16000

# Tamanho único para comparabilidade (originais 90x90 -> 160 para todos)
DEFAULT_IMG = 160

USE_TORCH_COMPILE = False  # desliga o torch.compile no Windows
# ============================================================
# ---------------- FUNÇÕES AUXILIARES -------------------------
# ============================================================
def get_backbone_type(backbone_id: str):
    name = backbone_id.lower()
    is_vit_like = any(k in name for k in ['vit', 'deit', 'eva'])
    is_swin = 'swin' in name
    is_efficient = 'efficientnet' in name
    is_staged = any(k in name for k in ['convnext', 'coatnet', 'maxxvit', 'edgenext', 'swin'])
    return is_vit_like, is_swin, is_efficient, is_staged

def get_batch_size(backbone_id: str):
    name = backbone_id.lower()
    if any(k in name for k in ['vit','deit','eva','swin','coat']):
        return 16
    return 32


#Para comparação justa, usamos o mesmo tamanho para todos, excecao de arquiteturas com atenção 2D com viés posicional relativo
def get_img_size(backbone_id: str):
    name = backbone_id.lower()
    if any(k in name for k in ['vit', 'deit', 'eva', 'swin', 'coatnet', 'maxxvit']):
        return 224
    return DEFAULT_IMG

# ============================================================
# ---------------- TRANSFORMS --------------------------------
# ============================================================
def build_transforms(img_size: int):
    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        # Mantém célula no quadro; variações leves:
        T.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.95, 1.05),
                       shear=5, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.15, 0.15, 0.10, 0.02),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_transform, val_transform

# ============================================================
# ---------------- DATASET -----------------------------------
# ============================================================
class FusionDataset(Dataset):
    def __init__(self, imagens, rotulos, transform=None):
        self.imagens = imagens
        self.rotulos = np.array(rotulos).astype(np.int64)
        self.transform = transform
    def __len__(self):
        return len(self.imagens)
    def __getitem__(self, idx):
        img = self.imagens[idx]
        # BGR -> RGB
        if img is None:
            img = np.zeros((90,90,3), np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y = self.rotulos[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y)

# ============================================================
# ---------------- MODEL / UNFREEZE ---------------------------
# ============================================================
def build_model(backbone_id: str):
    model = timm.create_model(
        backbone_id,
        pretrained=PRETRAINED,
        num_classes=num_classes
    )
    return model

def set_trainable(
    model: nn.Module,
    phase: int,
    vit_last_blocks: int = 1,
    unfreeze_more_stages: bool = False,
    freeze_norm_eval: bool = True,
    verbose: bool = False,
):
    def _is_iterable_modules(x):
        return isinstance(x, (list, tuple, nn.ModuleList, nn.Sequential))
    def _unfreeze_any(x):
        if x is None: return
        if isinstance(x, nn.Module):
            for p in x.parameters(): p.requires_grad = True
            return
        if _is_iterable_modules(x):
            for m in x: _unfreeze_any(m)
            return
        if isinstance(x, dict):
            for m in x.values(): _unfreeze_any(m)
    def _freeze_norms(m: nn.Module):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.LayerNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
    def _collect_trainable(m: nn.Module):
        return [n for n,p in m.named_parameters() if p.requires_grad]

    # 1) Congela tudo
    for p in model.parameters():
        p.requires_grad = False

    # 2) Head sempre treinável
    head_keys = ('head', 'classifier', 'fc', 'heads', 'pre_logits')
    for name, p in model.named_parameters():
        if any(k in name for k in head_keys):
            p.requires_grad = True

    # Phase 1: apenas head
    if phase != 2:
        if verbose:
            print("[set_trainable] phase=1 → apenas head/classifier.")
            print("Trainable:", len(_collect_trainable(model)))
        return

    # Detectar tipo
    cname = model.__class__.__name__.lower()
    has_blocks = hasattr(model, 'blocks')
    has_stages = hasattr(model, 'stages')
    has_features = hasattr(model, 'features')
    is_vit_like = has_blocks and (hasattr(model, 'cls_token') or 'vit' in cname or 'deit' in cname or 'eva' in cname)
    is_coatnet = ('coatnet' in cname) and has_stages
    is_staged = has_stages
    is_efficientnet_like = ('efficientnet' in cname) or (has_blocks and ('conv_head' in dir(model) or 'bn1' in dir(model) or 'bn2' in dir(model)))

    # Norm policy
    if is_vit_like:
        freeze_norm_eval = True
    elif is_efficientnet_like:
        freeze_norm_eval = False
    else:
        freeze_norm_eval = True
    if freeze_norm_eval:
        model.apply(_freeze_norms)

    # Unfreeze por tipo
    if is_vit_like and not is_staged:
        blocks = model.blocks
        if vit_last_blocks > 0 and hasattr(blocks, '__getitem__'):
            try:
                last = blocks[-vit_last_blocks:]
            except Exception:
                last = blocks
            _unfreeze_any(last)
        else:
            _unfreeze_any(blocks)
        if hasattr(model, 'norm'):
            _unfreeze_any(model.norm)
        if verbose:
            print(f"[set_trainable] ViT/EVA: últimos {vit_last_blocks} blocks + norm + head.")
            print("Trainable:", len(_collect_trainable(model)))
        return

    if is_coatnet:
        stages = model.stages
        if _is_iterable_modules(stages) and len(stages) >= 1:
            if unfreeze_more_stages and len(stages) >= 2:
                _unfreeze_any(stages[-2])
            _unfreeze_any(stages[-1])
        if hasattr(model, 'norm'):
            _unfreeze_any(model.norm)
        if verbose:
            print("[set_trainable] CoAtNet: último stage (+ opcional penúltimo) + norm + head.")
            print("Trainable:", len(_collect_trainable(model)))
        return

    if is_staged:
        stages = model.stages
        if _is_iterable_modules(stages) and len(stages) >= 1:
            if unfreeze_more_stages and len(stages) >= 2:
                _unfreeze_any(stages[-2])
            _unfreeze_any(stages[-1])
        if hasattr(model, 'norm'):
            _unfreeze_any(model.norm)
        if verbose:
            print("[set_trainable] Staged backbone: último stage (+ opcional penúltimo) + norm + head.")
            print("Trainable:", len(_collect_trainable(model)))
        return

    if is_efficientnet_like:
        # BN deve adaptar; liberar 2-3 últimos blocos
        if hasattr(model, 'blocks') and _is_iterable_modules(model.blocks):
            blocks = model.blocks
            if len(blocks) >= 3:
                to_free = blocks[-3:]
            elif len(blocks) >= 2:
                to_free = blocks[-2:]
            else:
                to_free = [blocks[-1]]
            for b in to_free: _unfreeze_any(b)
        if hasattr(model, 'conv_head'): _unfreeze_any(model.conv_head)
        if hasattr(model, 'bn1'): _unfreeze_any(model.bn1)
        if hasattr(model, 'bn2'): _unfreeze_any(model.bn2)
        if hasattr(model, 'features') and _is_iterable_modules(model.features):
            feats = model.features
            if len(feats) >= 2:
                _unfreeze_any(feats[-2]); _unfreeze_any(feats[-1])
            else:
                _unfreeze_any(feats[-1])
        if verbose:
            print("[set_trainable] EfficientNet: últimos 2-3 blocos + conv_head + BNs + head.")
            print("Trainable:", len(_collect_trainable(model)))
        return

    # fallback ~20% finais
    head_keys = ('head', 'classifier', 'fc', 'heads', 'pre_logits')
    backbone_params = [p for n,p in model.named_parameters() if not any(k in n for k in head_keys)]
    n_unfreeze = max(1, int(0.2 * len(backbone_params)))
    for p in backbone_params[-n_unfreeze:]:
        p.requires_grad = True
    if verbose:
        print("[set_trainable] Fallback: liberando ~20% finais.")
        print("Trainable:", len(_collect_trainable(model)))

# ============================================================
# ---------------- OPTIMIZER / SCHEDULER ----------------------
# ============================================================
def make_optimizer(model, phase):
    # detecta ViT-like
    is_vit_like = hasattr(model, 'blocks') and (hasattr(model, 'cls_token') or hasattr(model, 'pos_embed'))

    # fase 1: head LR, backbone LR ~ 0
    if phase == 1 or not (is_vit_like and USE_LLRD and phase == 2):
        head, backbone = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            (head if ('head' in name or 'classifier' in name) else backbone).append(p)
        if phase == 1:
            lr_h, lr_b = LR_HEAD_PHASE1, LR_BACKBONE_PHASE1
        else:
            lr_h, lr_b = LR_HEAD_PHASE2, LR_BACKBONE_PHASE2
        return torch.optim.AdamW(
            [{'params': head, 'lr': lr_h}, {'params': backbone, 'lr': lr_b}],
            weight_decay=WEIGHT_DECAY
        )

    # fase 2 + ViT-like + LLRD
    param_groups = []
    # head
    head_params = [p for n,p in model.named_parameters() if p.requires_grad and any(k in n for k in ('head','classifier'))]
    if head_params:
        param_groups.append({'params': head_params, 'lr': LR_HEAD_PHASE2, 'weight_decay': WEIGHT_DECAY})

    # norm final
    if hasattr(model, 'norm'):
        norm_params = [p for p in model.norm.parameters() if p.requires_grad]
        if norm_params:
            param_groups.append({'params': norm_params, 'lr': LR_BACKBONE_PHASE2, 'weight_decay': WEIGHT_DECAY})

    # blocks com layer decay
    blocks = model.blocks
    num_blocks = len(blocks)
    for i, blk in enumerate(blocks):
        decay_i = (LAYER_DECAY ** (num_blocks - 1 - i))
        lr_i = LR_BACKBONE_PHASE2 * decay_i
        params_i = [p for p in blk.parameters() if p.requires_grad]
        if params_i:
            param_groups.append({'params': params_i, 'lr': lr_i, 'weight_decay': WEIGHT_DECAY})

    return torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

def make_scheduler(optimizer, steps_per_epoch, max_epochs=EPOCHS_MAX, warmup_epochs=WARMUP_EPOCHS):
    if not USE_WARMUP_COSINE:
        return None
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================================
# ---------------- EVAL / TTA --------------------------------
# ============================================================
def predict_with_tta(model, x):
    """
    TTA simples: original + flip horizontal.
    Retorna probabilidade média (softmax) sobre as TTA views.
    """
    logits_list = []
    with torch.no_grad():
        # original
        logits_list.append(model(x))
        # flip horizontal
        logits_list.append(model(torch.flip(x, dims=[3])))
    probs = torch.stack([F.softmax(l, dim=1) for l in logits_list], dim=0).mean(0)
    return probs

def evaluate(model, loader):
    model.eval()
    preds, labels, probs_all = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            if device == 'cuda':
                x = x.to(memory_format=torch.channels_last)

            if USE_TTA:
                probs = predict_with_tta(model, x)
            else:
                logits = model(x)
                probs = F.softmax(logits, dim=1)

            pred = probs.argmax(1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.numpy())
            probs_all.extend(probs.cpu().numpy())

    labels = np.array(labels)
    preds = np.array(preds)
    probs_all = np.array(probs_all)

    return (
        f1_score(labels, preds, average='micro'),
        f1_score(labels, preds, average='macro'),
        labels,
        preds,
        probs_all
    )

# ============================================================
# ---------------- TRAIN -------------------------------------
# ============================================================
def train(model, train_loader, val_loader, save_path):
    scaler = GradScaler('cuda', enabled=(device == 'cuda'))
    # mixup
    mixup_fn = Mixup(
        mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA, prob=MIXUP_PROB,
        switch_prob=0.0, mode='batch', label_smoothing=LABEL_SMOOTHING,
        num_classes=num_classes
    ) if USE_MIXUP else None
    criterion = SoftTargetCrossEntropy() if mixup_fn is not None else nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    for phase in [1, 2]:
        print(f"\n===== PHASE {phase} =====")
        best_macro = -1
        no_improve = 0
        model_ema = ModelEmaV2(model, decay=EMA_DECAY) if USE_EMA else None

        set_trainable(model, phase, verbose=True)
        optimizer = make_optimizer(model, phase)

        steps_per_epoch = len(train_loader)
        # equaliza orçamento: calcula epochs para atingir ~TARGET_STEPS_PER_PHASE
        dyn_epochs = max(1, TARGET_STEPS_PER_PHASE // max(1, steps_per_epoch))
        dyn_epochs = min(dyn_epochs, EPOCHS_MAX)
        print(f"Treinando até {dyn_epochs} épocas (≈{TARGET_STEPS_PER_PHASE} steps/fase).")

        scheduler = make_scheduler(optimizer, steps_per_epoch, max_epochs=dyn_epochs, warmup_epochs=WARMUP_EPOCHS)

        for epoch in range(dyn_epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"P{phase} Epoch {epoch+1}/{dyn_epochs}")
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                if device == 'cuda' and 'coatnet' not in BACKBONE_ID.lower():
                    x = x.to(memory_format=torch.channels_last)

                targets = y
                if mixup_fn is not None:
                    x, targets = mixup_fn(x, y)

                with autocast('cuda', enabled=(device == 'cuda')):
                    logits = model(x)
                    loss = criterion(logits, targets)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                # grad clipping
                scaler.unscale_(optimizer)
                if GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()
                if model_ema:
                    model_ema.update(model)

                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # ---- validação ao final da época ----
            eval_model = model_ema.module if model_ema else model
            micro, macro, _, _, _ = evaluate(eval_model, val_loader)
            print(f"Val → microF1={micro:.4f} | macroF1={macro:.4f}")

            if macro > best_macro:
                best_macro = macro
                torch.save(eval_model.state_dict(), save_path)
                no_improve = 0
                print("✓ Melhor modelo salvo")
            else:
                no_improve += 1

            if (no_improve >= EARLY_STOP_PATIENCE) and ((epoch + 1) >= MIN_EPOCHS_BEFORE_STOP):
                print(f"Early stopping na fase {phase}")
                break

        gc.collect()

# ============================================================
# ---------------- MAIN (MULTI-BACKBONE) ---------------------
# ============================================================
if __name__ == '__main__':
    # ajuste seu caminho conforme seu projeto
    sys.path.append('/Users/xr4good/Desktop/Ingrid/DIFF/')
    from balance import ler_BalanceamentoDividido

    X_train, y_train, X_val, y_val, X_test, y_test = ler_BalanceamentoDividido(num_classes)

    def padronizar(lista):
        out = []
        for img in lista:
            if img is None:
                img = np.zeros((90,90,3), np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            out.append(cv2.resize(img, (90,90)))  # mantemos 90 aqui; transform fará o upscale
        return np.array(out)

    X_train, X_val, X_test = map(padronizar, [X_train, X_val, X_test])

    RESULTS_FILE = "results_backbones.txt"
    with open(RESULTS_FILE, "w") as f:
        f.write("CRIC – Pap Smear Classification Results\n")
        f.write("=" * 60 + "\n\n")

    for BACKBONE_ID in BACKBONES:
        print("\n" + "=" * 80)
        print(f" TREINANDO BACKBONE: {BACKBONE_ID}")
        print("=" * 80)

        BATCH_SIZE = get_batch_size(BACKBONE_ID)
        IMG_SIZE = get_img_size(BACKBONE_ID)
        print(f"IMG_SIZE={IMG_SIZE} | BATCH_SIZE={BATCH_SIZE}")

        # transforms
        train_transform, val_transform = build_transforms(IMG_SIZE)

        # datasets/loaders
        train_ds = FusionDataset(X_train, y_train, train_transform)
        val_ds = FusionDataset(X_val, y_val, val_transform)
        test_ds = FusionDataset(X_test, y_test, val_transform)

        if USE_WEIGHTED_SAMPLER:
            class_counts = np.bincount(y_train, minlength=num_classes)
            class_weights = 1.0 / (class_counts + 1e-6)
            sample_weights = class_weights[y_train]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=4 if device == 'cuda' else 0,
                drop_last=True,
                pin_memory=(device == 'cuda')
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=4 if device == 'cuda' else 0,
                drop_last=True,
                pin_memory=(device == 'cuda')
            )

        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # model
        model = build_model(BACKBONE_ID).to(device)

        # opcional: torch.compile (PyTorch 2+)
        if USE_TORCH_COMPILE:
            try:
                model = torch.compile(model)
                print("torch.compile ativado.")
            except Exception as e:
                print(f"torch.compile indisponível/skip: {e}")

        save_path = f"best_{BACKBONE_ID.replace('.','_')}.pth"

        # train
        train(model, train_loader, val_loader, save_path)

        # test
        state = torch.load(save_path, map_location=device)
        model.load_state_dict(state)
        micro, macro, y_true, y_pred, y_prob = evaluate(model, test_loader)
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

        # ================= CONFUSION MATRIX =================
        
        cm_counts = confusion_matrix(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        
        def plot_confusion_matrix(cm, fmt, title, path):
        
            plt.figure(figsize=(7,6))
        
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title(title)
            plt.colorbar()
        
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
        
            thresh = cm.max() / 2.0
        
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j, i,
                        format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black"
                    )
        
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
        
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
        
        
        # matriz absoluta
        plot_confusion_matrix(
            cm_counts,
            "d",
            f"Confusion Matrix (Counts) - {BACKBONE_ID}",
            f"cm_counts_{BACKBONE_ID.replace('.','_')}.png"
        )
        
        # matriz normalizada
        plot_confusion_matrix(
            cm,
            ".2f",
            f"Confusion Matrix (Normalized) - {BACKBONE_ID}",
            f"cm_norm_{BACKBONE_ID.replace('.','_')}.png"
        )
        
        
        # ================= ROC e PR Curves =================
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        precision = dict()
        recall = dict()
        pr_auc = dict()
        

        if num_classes == 2:
            # ===== ROC =====
            # Classe 1 → Normal
            fpr[1], tpr[1], _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc[1] = auc(fpr[1], tpr[1])
        
            # Classe 0 → Alterada
            fpr[0], tpr[0], _ = roc_curve(1 - y_true, y_prob[:, 0])
            roc_auc[0] = auc(fpr[0], tpr[0])
        
            # ===== PR =====
            # Classe 1 → Normal
            precision[1], recall[1], _ = precision_recall_curve(
                y_true, y_prob[:, 1]
            )
            pr_auc[1] = auc(recall[1], precision[1])
        
            # Classe 0 → Alterada
            precision[0], recall[0], _ = precision_recall_curve(
                1 - y_true, y_prob[:, 0]
            )
            pr_auc[0] = auc(recall[0], precision[0])
        
        else:
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        
            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
                precision[i], recall[i], _ = precision_recall_curve(
                    y_true_bin[:, i], y_prob[:, i]
                )
                pr_auc[i] = auc(recall[i], precision[i])
        
        
        # ================= MICRO / MACRO ROC =================
        
        # micro ROC
        if num_classes == 2:
            fpr["micro"], tpr["micro"], _ = roc_curve(
                y_true,
                y_prob[:, 1]
            )
        else:
            fpr["micro"], tpr["micro"], _ = roc_curve(
                y_true_bin.ravel(),
                y_prob.ravel()
            )
        
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        
        # macro ROC
        if num_classes == 2:
            # junta FPR das duas classes
            all_fpr = np.unique(np.concatenate([fpr[0], fpr[1]]))
        
            mean_tpr = np.zeros_like(all_fpr)
        
            for i in [0, 1]:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
            mean_tpr /= 2  # média das duas classes
        
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(all_fpr, mean_tpr)
        else:
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        
            mean_tpr = np.zeros_like(all_fpr)
        
            classes_to_plot = [1] if num_classes == 2 else range(num_classes)

            for i in classes_to_plot:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
            mean_tpr /= len(classes_to_plot)
        
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        
        # ================= ROC PLOT =================
        
        plt.figure(figsize=(7,7))
        
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label=f"micro-average (AUC={roc_auc['micro']:.3f})",
            linewidth=3
        )
        
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label=f"macro-average (AUC={roc_auc['macro']:.3f})",
            linewidth=3
        )
        
        classes_to_plot = [0, 1] if num_classes == 2 else range(num_classes)
        for i in classes_to_plot:
            plt.plot(
                fpr[i],
                tpr[i],
                linewidth=2,
                label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})"
            )
        
        plt.plot([0,1],[0,1],'k--')
        
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        
        plt.title(f"ROC Curve - {BACKBONE_ID}")
        plt.legend(loc="lower right")
        
        roc_path = f"roc_{BACKBONE_ID.replace('.','_')}.png"
        
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        
        # ================= MICRO PR =================
        if num_classes == 2:
            precision["micro"], recall["micro"], _ = precision_recall_curve(
                y_true,
                y_prob[:, 1]
            )
        else:
            precision["micro"], recall["micro"], _ = precision_recall_curve(
                y_true_bin.ravel(),
                y_prob.ravel()
            )
        
        pr_auc["micro"] = auc(recall["micro"], precision["micro"])
        
        
        # ================= PR PLOT =================
        
        plt.figure(figsize=(7,7))
        
        plt.plot(
            recall["micro"],
            precision["micro"],
            linewidth=3,
            label=f"micro-average (AUC={pr_auc['micro']:.3f})"
        )
        
        classes_to_plot = [0, 1] if num_classes == 2 else range(num_classes)

        for i in classes_to_plot:
            plt.plot(
                recall[i],
                precision[i],
                linewidth=2,
                label=f"{class_names[i]} (AUC={pr_auc[i]:.3f})"
            )
        
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        
        plt.title(f"Precision-Recall Curve - {BACKBONE_ID}")
        
        plt.legend(fontsize=9)
        
        pr_path = f"pr_{BACKBONE_ID.replace('.','_')}.png"
        
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        
        print(f"\n TESTE FINAL → microF1={micro:.4f} | macroF1={macro:.4f}")
        print(report)
        
        print("Matriz de confusão normalizada:\n", cm)
        print("Matriz de confusão absoluta:\n", cm_counts)

        # salvar resultados
        with open(RESULTS_FILE, "a") as f:
            f.write(f"\nBACKBONE: {BACKBONE_ID}\n")
            f.write(f"IMG_SIZE: {IMG_SIZE} | BATCH_SIZE: {BATCH_SIZE}\n")
            f.write(f"TEST microF1: {micro:.4f}\n")
            f.write(f"TEST macroF1: {macro:.4f}\n")
            f.write(report + "\n")
            f.write("Confusion Matrix (Normalized):\n")
            np.savetxt(f, cm, fmt='%.4f')
            
            f.write("\nConfusion Matrix (Counts):\n")
            np.savetxt(f, cm_counts, fmt='%d')
            f.write("\n" + "-" * 60 + "\n")

        # limpeza
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    print("\n EXPERIMENTOS FINALIZADOS")
    print(f" Resultados salvos em: {RESULTS_FILE}")
