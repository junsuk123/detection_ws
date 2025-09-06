import os
import sys
import shutil
from glob import glob
from pathlib import Path
from PIL import Image
import subprocess
from ultralytics import YOLO
import matplotlib.pyplot as plt
import math

# ====================== [ 설정 파라미터 ] ======================
DATASET_ROOT = 'dataset'           # 여러 개의 데이터 폴더가 있는 상위 디렉토리
OUTPUT_DIR   = 'merged_dataset'    # 변환된 YOLO 데이터셋 저장 디렉토리
MODEL_NAME   = 'yolo11n.pt'        # 사용할 사전학습된 모델 파일
EPOCHS       = 400
DEVICE       = 0                   # GPU 번호 또는 'cpu'
VAL_SPLIT    = 0.1                 # ✅ 검증 분할 비율(0~1)
MAX_IMGSZ    = 1280                # 너무 큰 해상도 방지용 상한
SEED         = 42
# ============================================================

# 0. 예제 이미지 자동 탐색 (크기 추출용)
sample_paths = glob(os.path.join(DATASET_ROOT, '*', '*.png'))
if len(sample_paths) == 0:
    raise FileNotFoundError(f"No .png files found under '{DATASET_ROOT}'")
IMG_PATH_EXAMPLE = sample_paths[0]
print(f"[INFO] Using example image for size detection: {IMG_PATH_EXAMPLE}")

# 1. 이미지 크기 추출 -> 32 배수로 스냅 + 상한
with Image.open(IMG_PATH_EXAMPLE) as im:
    width, height = im.size
    base = max(width, height)
imgsz_raw = min(base, MAX_IMGSZ)
imgsz = int(math.ceil(imgsz_raw / 32) * 32)
print(f"[INFO] Detected image size: ({width}x{height}) → imgsz={imgsz} (snapped to 32)")

# 2. YOLO 포맷으로 변환할 통합 대상 폴더 구조 생성
merged_image_dir = os.path.abspath(os.path.join(OUTPUT_DIR, 'image'))
os.makedirs(merged_image_dir, exist_ok=True)

# 3. dataset/ 하위 폴더들 순회하며 .json, .png 복사 (중복 방지용 접두사 포함)
print(f"[INFO] Merging all dataset folders under '{DATASET_ROOT}'...")
dataset_dirs = [f.path for f in os.scandir(DATASET_ROOT) if f.is_dir()]
image_idx = 0
for ds_dir in dataset_dirs:
    prefix = os.path.basename(ds_dir)
    pngs = sorted(glob(os.path.join(ds_dir, '*.png')))
    for png_path in pngs:
        json_path = png_path.replace('.png', '.json')
        if not os.path.exists(json_path):
            continue
        new_img = f'{prefix}_{image_idx:06d}.png'
        new_json = f'{prefix}_{image_idx:06d}.json'
        shutil.copy(png_path, os.path.join(merged_image_dir, new_img))
        shutil.copy(json_path, os.path.join(merged_image_dir, new_json))
        image_idx += 1
print(f"[INFO] Copied {image_idx} image/json pairs to '{merged_image_dir}'")

# 4. labelme2yolo 변환 (✅ 검증 셋 분할 활성화)
print("[INFO] Converting LabelMe JSON → YOLO format...")
labelme2yolo_cmd = shutil.which('labelme2yolo')
common_args = ['--json_dir', merged_image_dir,
               '--output_format', 'polygon',
               '--val_size', str(VAL_SPLIT)]
if labelme2yolo_cmd:
    cmd = [labelme2yolo_cmd] + common_args
else:
    cmd = [sys.executable, '-m', 'labelme2yolo.labelme2yolo'] + common_args
subprocess.run(cmd, check=True)

# 4.5. 변환 결과 경로
yolo_dataset_dir = os.path.abspath(os.path.join(merged_image_dir, 'YOLODataset'))
imgs_train_dir = os.path.join(yolo_dataset_dir, 'images', 'train')
imgs_val_dir   = os.path.join(yolo_dataset_dir, 'images', 'val')
lbls_train_dir = os.path.join(yolo_dataset_dir, 'labels', 'train')
lbls_val_dir   = os.path.join(yolo_dataset_dir, 'labels', 'val')

# (안전장치) val 폴더가 비어 있으면 train에서 한 장만 옮기기
os.makedirs(imgs_val_dir, exist_ok=True)
os.makedirs(lbls_val_dir, exist_ok=True)
if not glob(os.path.join(imgs_val_dir, '*.png')):
    train_imgs = sorted(glob(os.path.join(imgs_train_dir, '*.png')))
    if train_imgs:
        img_to_move = train_imgs[0]
        lbl_to_move = os.path.join(lbls_train_dir, os.path.basename(img_to_move).replace('.png', '.txt'))
        shutil.move(img_to_move, imgs_val_dir)
        if os.path.exists(lbl_to_move):
            shutil.move(lbl_to_move, lbls_val_dir)
        print(f"[INFO] Moved 1 sample from train → val: {os.path.basename(img_to_move)}")

# 5. YOLO 학습 시작 (✅ 매 epoch 검증/플롯 저장)
os.chdir(yolo_dataset_dir)
print("[INFO] Starting YOLO training with validation enabled...")
model = YOLO(MODEL_NAME)
train_results = model.train(
    data='dataset.yaml',
    epochs=EPOCHS,
    imgsz=imgsz,
    batch=4,
    device=DEVICE,
    name='run',
    workers=8,          # 환경에 맞게 조정
    seed=SEED,
    cache=True,        # 첫 epoch 로딩 가속
    patience=50,       # early-stopping 여유
    verbose=True,
    plots=True,        # ✅ 결과 플롯 저장
)
print(f"[INFO] ✅ Training complete. See: {yolo_dataset_dir}/runs/detect/run")

# 6. 최종 모델로 예시 이미지 추론 및 시각화
inference_candidates = sorted(glob(os.path.join(merged_image_dir, '*.png')))
if len(inference_candidates) == 0:
    # 합본 폴더 루트에 .png가 없을 수 있어 train에서 하나 사용
    inference_candidates = sorted(glob(os.path.join(imgs_train_dir, '*.png'))) or \
                           sorted(glob(os.path.join(imgs_val_dir, '*.png')))
inference_image = inference_candidates[0]
print(f"[INFO] Running inference on: {inference_image}")

best_path = os.path.join('runs/detect/run/weights/best.pt')
model = YOLO(best_path)
results = model.predict(source=inference_image, save=False, conf=0.25)

# 결과 이미지 시각화
result_img = results[0].plot()[..., ::-1]  # BGR → RGB
plt.figure(figsize=(8, 8))
plt.imshow(result_img)
plt.axis('off')
plt.title("YOLOv11n Inference Result")
plt.show()

