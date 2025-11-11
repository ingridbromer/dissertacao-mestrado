import os
import json
from PIL import Image

# ================== CAMINHOS ==================
base_dir = "E:/datasets/imagens/base"
json_path = os.path.join(base_dir, "classifications_6classes.json")

out_6 = "E:/datasets/imagens/6classes/"
os.makedirs(out_6, exist_ok=True)

# ================== PASTAS JÁ EXISTENTES ==================
classes_6 = ["Normal", "ASCH", "ASCUS", "LSIL", "HSIL", "CA"]

for cls in classes_6:
    os.makedirs(os.path.join(out_6, cls), exist_ok=True)


# ================== DE–PARA: LABEL → PASTA 6 CLASSES ==================
mapa_6 = {
    "NEGATIVE": "Normal",
    "ASC-H":    "ASCH",
    "ASC-US":   "ASCUS",
    "LSIL":     "LSIL",
    "HSIL":     "HSIL",
    "SCC":      "CA"
}

# ================== CARREGAR JSON ==================
with open(json_path, "r") as f:
    data = json.load(f)

# ================== EXTRAIR TODAS AS CÉLULAS ==================
all_cells = []
for img_data in data:
    image_name = img_data["image_name"]
    for cell in img_data["classifications"]:
        all_cells.append({
            "image_name": image_name,
            "cell_id": cell["cell_id"],
            "x": cell["nucleus_x"],
            "y": cell["nucleus_y"],
            "label": cell["bethesda_system"]
        })

# ================== RECORTAR 90x90 pixels E SALVAR ==================
descartadas = 0
for cell in all_cells:
    image_path = os.path.join(base_dir, cell["image_name"])
    if not os.path.exists(image_path):
        descartadas += 1
        continue

    try:
        img = Image.open(image_path)
    except:
        descartadas += 1
        continue

    x, y = cell["x"], cell["y"]
    half_crop = 45
    if x - half_crop < 0 or y - half_crop < 0 or x + half_crop > img.width or y + half_crop > img.height:
        descartadas += 1
        continue

    crop = img.crop((x - half_crop, y - half_crop, x + half_crop, y + half_crop))
    base_name = f"{os.path.splitext(cell['image_name'])[0]}_celula_{cell['cell_id']}.png"

    label = cell["label"]

    # Conversão via de–para
    classe_6 = mapa_6[label]

    crop.save(os.path.join(out_6, classe_6, base_name))
print(f"Células descartadas: {descartadas}")
print("Divisão em 6 classes concluída com sucesso!")
