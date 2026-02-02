#!/bin/bash
# ===========================================
# LOGIC-HALT Setup Script
# RTX 5070 (CUDA 12.x) için
# ===========================================

echo "========================================"
echo "🚀 LOGIC-HALT RTX 5070 Setup"
echo "========================================"

# 1. Check NVIDIA driver
echo ""
echo "📊 Checking NVIDIA GPU..."
nvidia-smi || { echo "❌ NVIDIA driver not found!"; exit 1; }

# 2. Create virtual environment
echo ""
echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA
echo ""
echo "⏳ Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install other requirements
echo ""
echo "⏳ Installing other dependencies..."
pip install -r requirements.txt

# 5. Verify CUDA
echo ""
echo "🔍 Verifying CUDA installation..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 6. Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p config/optimization_results
mkdir -p data/processed/results
mkdir -p logs

# 7. Done
echo ""
echo "========================================"
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Copy your .env file with API keys"
echo "   2. Activate venv: source venv/bin/activate"
echo "   3. Run: python scripts/truthfulqa_optimization.py --trials 100"
echo "========================================"
