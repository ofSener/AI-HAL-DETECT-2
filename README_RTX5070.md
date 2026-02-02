# LOGIC-HALT × TruthfulQA Bayesian Optimization

**RTX 5070 (12GB VRAM) için optimize edildi**

## Hızlı Başlangıç

### 1. Kurulum
```bash
# Projeyi aç
cd bitirm2

# Setup scriptini çalıştır
chmod +x setup_rtx5070.sh
./setup_rtx5070.sh

# Veya manuel kurulum:
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. API Anahtarları
`.env` dosyasını oluştur:
```bash
OPENAI_API_KEY=sk-your-key-here
```

### 3. GPU Kontrolü
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. Bayesian Optimization Çalıştır

```bash
# Aktif et
source venv/bin/activate

# Hızlı test (50 soru, 30 trial) - ~15 dakika
python scripts/truthfulqa_optimization.py --trials 30 --sample 50

# Orta (200 soru, 50 trial) - ~45 dakika
python scripts/truthfulqa_optimization.py --trials 50 --sample 200

# Full (817 soru, 100 trial) - ~2 saat
python scripts/truthfulqa_optimization.py --trials 100

# Uzun optimizasyon (200 trial, 6 saat timeout)
python scripts/truthfulqa_optimization.py --trials 200 --timeout 21600
```

## Parametreler

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--trials` | Optuna deneme sayısı | 100 |
| `--sample` | Kullanılacak soru sayısı | None (817) |
| `--folds` | Cross-validation fold | 5 |
| `--timeout` | Maksimum süre (saniye) | None |
| `--seed` | Random seed | 42 |
| `--name` | Sonuç dosyası adı | truthfulqa_rtx5070 |

## Çıktılar

Sonuçlar `config/optimization_results/` klasörüne kaydedilir:
- `truthfulqa_rtx5070_best_params.json` - En iyi parametreler
- `truthfulqa_rtx5070_trials.csv` - Tüm denemeler
- `truthfulqa_rtx5070_importance.json` - Parametre önem sırası

## GPU Memory Hatası (OOM)

Eğer memory hatası alırsan, `config/config.yaml`'da batch_size'ı düşür:
```yaml
consistency:
  batch_size: 64  # 128'den düşür
```

## Proje Yapısı
```
bitirm2/
├── config/
│   ├── config.yaml         # RTX 5070 için optimize
│   └── optimization_results/
├── data/raw/
│   └── truthfulqa_dataset.json  # 817 soru
├── scripts/
│   └── truthfulqa_optimization.py
├── src/                    # Core modüller
├── web/                    # Web arayüzü
└── .env                    # API keys
```
