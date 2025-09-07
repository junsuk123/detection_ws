#!/bin/bash

# Git LFS 설치 확인
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS가 설치되어 있지 않습니다. 설치를 시작합니다..."
    
    # 우분투 기준 설치 명령어
    sudo apt-get update
    sudo apt-get install git-lfs
    
    echo "Git LFS 설치 완료"
else
    echo "Git LFS가 이미 설치되어 있습니다."
fi

# Git LFS 초기화
git lfs install

echo "저장소에서 Git LFS 초기화 완료"

# models 디렉토리 존재 확인 및 생성
if [ ! -d "models" ]; then
    echo "models 디렉토리가 없습니다. 생성합니다..."
    mkdir -p models
    echo "models 디렉토리 생성 완료"
fi

# .gitattributes 파일 생성 또는 업데이트
cat > .gitattributes << 'EOL'
# 모든 .pt 파일을 Git LFS로 관리
models/**/*.pt filter=lfs diff=lfs merge=lfs -text

# 모델 폴더의 다른 대용량 파일도 함께 관리
models/**/*.pth filter=lfs diff=lfs merge=lfs -text
models/**/*.bin filter=lfs diff=lfs merge=lfs -text
models/**/*.onnx filter=lfs diff=lfs merge=lfs -text
models/**/*.engine filter=lfs diff=lfs merge=lfs -text
models/**/*.weights filter=lfs diff=lfs merge=lfs -text

# 모델 메타데이터 파일은 일반 텍스트로 관리
models/**/*.yaml -filter=lfs -diff=lfs -merge=lfs text
models/**/*.json -filter=lfs -diff=lfs -merge=lfs text
EOL

echo ".gitattributes 파일이 업데이트되었습니다."

# Git LFS 추적 설정
git lfs track "models/**/*.pt"
git lfs track "models/**/*.pth"
git lfs track "models/**/*.bin"
git lfs track "models/**/*.onnx"
git lfs track "models/**/*.engine"
git lfs track "models/**/*.weights"

echo "Git LFS 설정이 완료되었습니다."
echo "다음 명령어를 실행하세요:"
echo "  cd /home/j/detection_ws/src/ultralytics_ros"
echo "  git add .gitattributes"
echo "  git add models/"
echo "  git add setup_git_lfs.sh"
echo "  git commit -m \"모델 파일에 Git LFS 적용\""
