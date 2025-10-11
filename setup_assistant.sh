#!/bin/bash
# ==========================================================
# CoIRL-AD Environment Setup Script
# This script stops immediately if any step fails.
# ==========================================================

set -e  # ⛔ Stop on first error

GREEN="\033[1;32m"
RED="\033[1;31m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
NC="\033[0m"

echo -e "${BLUE}=============================="
echo -e "  CoIRL-AD Environment Setup  "
echo -e "==============================${NC}"

# Helper to print step info
step() {
    echo -e "\n${YELLOW}=== Step $1: $2 ===${NC}"
}

# Helper to print success message
ok() {
    echo -e "${GREEN}✅ Step $1 completed successfully.${NC}\n"
}

# ----------------------------------------------------------
step 0 "Install dependencies from requirements.txt"
# (Don’t include torch/mmcv/mmdet3d in this file!)
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo -e "${RED}requirements.txt not found, skipping.${NC}"
fi
ok 0

# ----------------------------------------------------------
step 1 "Install PyTorch 1.9.0 + cu111"
echo -e "Make sure CUDA 11.1 is installed and CUDA_HOME is set correctly."
echo -e "Example: export CUDA_HOME=/usr/local/cuda-11.1\n"
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
ok 1

# ----------------------------------------------------------
step 2 "Install mmcv-full 1.4.0"
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
python -c "from mmcv.ops import nms; print('✅ mmcv ops imported OK')" || {
    echo -e "${RED}❌ mmcv import test failed!${NC}"
    exit 1
}
ok 2

# ----------------------------------------------------------
step 3 "Install mmengine / mmdet / mmseg"
pip install mmengine==0.10.7 mmdet==2.14.0 mmsegmentation==0.14.1
ok 3

# ----------------------------------------------------------
step 4 "Install mmdet3d (build locally)"
echo -e "${BLUE}Ensure GCC ≤ 9 (current: $(gcc -dumpversion))${NC}"
echo -e "If not, use: export CC=/usr/bin/gcc-9 && export CXX=/usr/bin/g++-9\n"
pip install -e git+https://github.com/open-mmlab/mmdetection3d.git@f1107977dfd26155fc1f83779ee6535d2468f449#egg=mmdet3d
ok 4

# ----------------------------------------------------------
step 5 "Install timm 1.0.20"
pip install timm==1.0.20
ok 5

# ----------------------------------------------------------
step 6 "Pin setuptools version"
pip install setuptools==59.5.0
ok 6

# ----------------------------------------------------------
step 7 "Verify installation"
python - <<'EOF'
import torch, mmcv, mmdet, mmdet3d
print(f"✅ torch: {torch.__version__}")
print(f"✅ mmcv: {mmcv.__version__}")
print(f"✅ mmdet: {mmdet.__version__}")
print(f"✅ mmdet3d: {mmdet3d.__version__}")
EOF
ok 7

echo -e "${GREEN}🎉 Environment setup completed successfully!${NC}"
