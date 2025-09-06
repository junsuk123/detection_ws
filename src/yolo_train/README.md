# YOLOv11n Local Training with LabelMe Dataset

이 프로젝트는 다양한 하위 폴더에 흩어진 LabelMe 형식의 데이터셋을 자동으로 병합하고, YOLO 포맷으로 변환하여 **YOLOv11n 모델을 GPU로 학습**하는 파이썬 스크립트입니다.

---

## 📁 프로젝트 구조

```
파이썬 파일포트/
├── train_yolo_local.py     # 메인 학습 스크립트
├── requirements.txt        # 설치 의존성 목록
├── README.md
└── dataset/                # 여러 폴더 존재 (crawling 대상)
    ├── demo/
    │   ├── image_000001.png
    │   ├── image_000001.json
    ├── jiphyeon/
    │   └── ...
```

---

## ✅ 기능 요약

* `dataset/` 포더 내 모든 하위 폴더 자료 자본 탑재
* `.png` / `.json` 세트 직접 정렬 및 추입
* `labelme2yolo` 이용해 polygon 기본 YOLO 형식으로 변환
* 이미지 크기에 따라 YOLO 입력 해상도 자동 설정
* GPU 기반 `YOLOv11n` 모델 학습

---

## ⚙️ 설치 (CUDA 12.1 기준)

CUDA **12.1** 기능 GPU 환경에서 다음 명령으로 설치하세요:

```bash
pip install -r requirements.txt
```

가까운 수동 설치:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics labelme2yolo Pillow
```

---

## 🚀 실행 방법

```bash
python3 train_yolo_local.py
```

학습이 완료되면 모델은 다음 위치에 저장됩니다:

```
merged_dataset/image/YOLODataset/runs/detect/run/
```

---

## 🧠 주의사항

* `train_yolo_local.py` 상단의 파래메터를 프로젝트 구조에 맞게 조정하세요.
* `dataset/` 내부 폴더 구조가 변경되면 `IMG_PATH_EXAMPLE` 경로도 같이 수정해야 합니다.
* `labelme2yolo`는 polygon 기본 YOLO 변환만 지원합니다.

---

## 📍 참고

* [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* [https://github.com/defcom17/labelme2yolo](https://github.com/defcom17/labelme2yolo)

