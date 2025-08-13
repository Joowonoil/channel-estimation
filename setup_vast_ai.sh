#!/bin/bash

echo "🚀 Vast AI 환경 자동 설정 시작..."

# 1. 시스템 업데이트
echo "📦 시스템 패키지 업데이트..."
apt update -y > /dev/null 2>&1

# 2. Git LFS 설치
echo "🔧 Git LFS 설치 및 활성화..."
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash > /dev/null 2>&1
apt install -y git-lfs > /dev/null 2>&1
git lfs install

# 3. WandB 자동 로그인
echo "📊 WandB 자동 로그인 설정..."
export WANDB_API_KEY=82a660d4aa45976a1a47c13396f07c22c81bf414
echo "export WANDB_API_KEY=82a660d4aa45976a1a47c13396f07c22c81bf414" >> ~/.bashrc
wandb login $WANDB_API_KEY > /dev/null 2>&1

# 4. GPU 환경 확인
echo "🎮 GPU 환경 확인..."
nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ GPU 사용 가능"
    python -c "import torch; print(f'✅ PyTorch CUDA: {torch.cuda.is_available()}')" 2>/dev/null
else
    echo "❌ GPU 확인 실패"
fi

# 5. 채널 추정 프로젝트 클론
echo "📥 채널 추정 프로젝트 클론..."
if [ -d "channel-estimation" ]; then
    rm -rf channel-estimation
fi

git clone https://github.com/Joowonoil/channel-estimation.git > /dev/null 2>&1
cd channel-estimation

# 6. Git LFS 설정 (현재는 100MB+ 파일 없어서 스킵)
echo "💾 Git LFS 설정 확인..."
git lfs install

# 7. 환경 변수 설정
echo "⚙️  환경 변수 설정..."
export CUDA_VISIBLE_DEVICES=0
echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc

# 8. 완료 메시지
echo ""
echo "🎉 설정 완료!"
echo "📁 작업 디렉토리: $(pwd)"
echo "🚀 사용 가능한 실행 파일들:"
echo "   - 베이스 모델: python engine_v3.py 또는 python engine_v4.py"
echo "   - 전이학습: python Transfer_v3_InF.py, Transfer_v3_RMa.py"
echo "   -           python Transfer_v4_InF.py, Transfer_v4_RMa.py"
echo ""
echo "📊 WandB 프로젝트는 실행하는 코드에 따라 자동 설정됩니다"
echo ""

# 9. 최종 상태 확인
echo "🔍 최종 환경 확인:"
echo "   - Git LFS: $(git lfs --version | head -1)"
echo "   - WandB: 자동 로그인 완료"
echo "   - PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "   - CUDA: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
echo "   - 데이터셋: $(ls -la dataset/PDP_processed/ | wc -l) 파일"
echo ""
echo "✅ 모든 설정이 완료되었습니다!"
echo ""
echo "📂 현재 위치: channel-estimation 폴더"
echo "🚀 원하는 스크립트를 실행하세요!"