# LOGIC-HALT

**LLM Hallucination Detection System using Multi-Signal Bayesian Optimization**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LOGIC-HALT, büyük dil modellerinin (LLM) ürettiği yanıtlardaki **halüsinasyonları** ve **mantık hatalarını** tespit etmek için geliştirilmiş bir sistemdir. Aynı soruya verilen birden fazla yanıtı analiz ederek tutarsızlıkları ortaya çıkarır.

## Temel Özellikler

- **Çoklu Sinyal Analizi**: Tutarlılık, entropi ve sıkıştırma mesafesi metriklerini birleştirir
- **NLI Tabanlı Çelişki Tespiti**: DeBERTa-v3 modeli ile anlamsal çelişkileri bulur
- **Bayesian Optimizasyon**: Optuna ile 13 hiperparametreyi otomatik optimize eder
- **Web Arayüzü**: Flask tabanlı kullanıcı dostu arayüz
- **GPU Hızlandırma**: RTX 5070 (12GB VRAM) için optimize edilmiş

## Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────┐
│                         LOGIC-HALT Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│   │  INPUT   │───▶│   MORPHER    │───▶│    INTERROGATOR       │ │
│   │  Soru    │    │  (Modül A)   │    │      (Modül B)        │ │
│   └──────────┘    │  5 Varyant   │    │  LLM API Sorgulama    │ │
│                   └──────────────┘    └───────────┬───────────┘ │
│                                                   │              │
│                   ┌───────────────────────────────┼─────────┐   │
│                   │                               ▼         │   │
│   ┌───────────────┴───────────┐  ┌──────────────────────┐  │   │
│   │   CONSISTENCY ENGINE      │  │  COMPLEXITY ENGINE   │  │   │
│   │       (Modül C)           │  │     (Modül D)        │  │   │
│   │  • NLI Çelişki Analizi    │  │  • Token Entropi     │  │   │
│   │  • Graf Tabanlı Metrikler │  │  • NCD Hesaplama     │  │   │
│   └───────────────┬───────────┘  └──────────┬───────────┘  │   │
│                   │                          │              │   │
│                   └──────────┬───────────────┘              │   │
│                              ▼                              │   │
│                   ┌──────────────────────┐                  │   │
│                   │    FUSION LAYER      │                  │   │
│                   │      (Modül E)       │                  │   │
│                   │  Risk = α×Cons +     │                  │   │
│                   │  β×Entropy + γ×NCD   │                  │   │
│                   └──────────┬───────────┘                  │   │
│                              ▼                              │   │
│                   ┌──────────────────────┐                  │   │
│                   │   ANSWER VALIDATOR   │                  │   │
│                   │      (Modül F)       │                  │   │
│                   │  Çoğunluk Oylama     │                  │   │
│                   └──────────┬───────────┘                  │   │
│                              ▼                              │   │
│                   ┌──────────────────────┐                  │   │
│                   │       OUTPUT         │                  │   │
│                   │  ✓ Safe / ✗ Halluc.  │                  │   │
│                   └──────────────────────┘                  │   │
└─────────────────────────────────────────────────────────────────┘
```

## Kurulum

### Gereksinimler

- Python 3.10+
- NVIDIA GPU (CUDA 12.1+) - Önerilen: RTX 3060+ (12GB VRAM)
- 32GB RAM (önerilen)

### Adım 1: Repository'yi Klonlayın

```bash
git clone https://github.com/yourusername/logic-halt.git
cd logic-halt
```

### Adım 2: Sanal Ortam Oluşturun

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Adım 3: Bağımlılıkları Yükleyin

```bash
# PyTorch (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Diğer bağımlılıklar
pip install -r requirements.txt
```

### Adım 4: API Anahtarlarını Ayarlayın

```bash
cp .env.example .env
```

`.env` dosyasını düzenleyin:
```env
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

## Kullanım

### Web Arayüzü

```bash
cd web
python app.py
```

Tarayıcıda `http://localhost:5000` adresine gidin.

**Özellikler:**
- Soru girin ve birden fazla LLM yanıtı üretin
- Her yanıt için risk skoru görün
- Manuel etiketleme yapın
- Sonuçları dışa aktarın

### Komut Satırı

#### Hızlı Test
```bash
python scripts/truthfulqa_optimization.py --trials 30 --sample 50
```

#### Tam Optimizasyon
```bash
python scripts/truthfulqa_optimization.py --trials 100
```

#### Batch API ile Maliyet Tasarrufu
```bash
# Batch hazırla
python scripts/batch_api_helper.py --prepare-variants --sample 100

# Gönder
python scripts/batch_api_helper.py --submit data/batch_results/batch_variants.jsonl

# Sonuçları indir
python scripts/batch_api_helper.py --download <BATCH_ID>
```

## Proje Yapısı

```
logic_halt_rtx5070/
│
├── src/                          # Kaynak kod modülleri
│   ├── morpher.py                # Modül A: Soru varyantları oluşturma
│   ├── interrogator.py           # Modül B: LLM sorgulama
│   ├── consistency.py            # Modül C: NLI tabanlı tutarlılık
│   ├── consistency_lite.py       # Modül C: Hafif versiyon
│   ├── complexity.py             # Modül D: Entropi ve NCD
│   ├── fusion.py                 # Modül E: Karar birleşimi
│   ├── answer_validator.py       # Modül F: Çoğunluk oylama
│   ├── detector.py               # Ana pipeline
│   ├── pipeline_demo.py          # Entegre demo
│   └── morpher_demo.py           # Mock demo
│
├── scripts/                      # Çalıştırma scriptleri
│   ├── truthfulqa_optimization.py    # TruthfulQA optimizasyonu
│   ├── comprehensive_optimization.py # Kapsamlı optimizasyon
│   ├── batch_optimization.py         # Batch API optimizasyonu
│   ├── batch_api_helper.py           # OpenAI Batch API wrapper
│   └── generate_visualizations.py    # Görselleştirme
│
├── web/                          # Web arayüzü
│   ├── app.py                    # Flask uygulaması
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       └── js/main.js
│
├── config/                       # Yapılandırma
│   ├── config.yaml               # Ana konfigürasyon
│   ├── prompts.yaml              # LLM promptları
│   └── optimization_results/     # Optimizasyon sonuçları
│
├── data/                         # Veri setleri
│   └── raw/
│       ├── truthfulqa_dataset.json       # 817 soru
│       ├── hallucination_test_dataset.json # 20 soru
│       └── pilot_dataset.json            # 20 mantık sorusu
│
├── requirements.txt
├── setup_rtx5070.sh
└── .env.example
```

## Modüller

### Modül A: Morpher
Orijinal soruyu 5 farklı dönüşüm ile varyantlara ayırır:
- **Paraphrase**: Farklı kelimelerle yeniden yazma
- **Negation**: Çift olumsuzlama
- **Variable Substitution**: Değişken isimlerini değiştirme
- **Premise Reordering**: Önerme sırasını değiştirme
- **Redundant Context**: İlgisiz bağlam ekleme

### Modül B: Interrogator
Hedef LLM'leri sorgular ve yanıtları toplar:
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude)
- Log-probability verileri ile birlikte

### Modül C: Consistency Engine
DeBERTa-v3 NLI modeli ile çelişki analizi:
- Yanıt çiftleri arasında entailment/neutral/contradiction tahmini
- Graf tabanlı metrikler (yoğunluk, kümeleme katsayısı)
- HYBRID FORMAT: Sadece ANSWER kısmı karşılaştırılır

### Modül D: Complexity Engine
Bilgi-teorik metrikler:
- **Token Entropy**: Model belirsizliği ölçümü
- **NCD (Normalized Compression Distance)**: Yanıt benzerliği

### Modül E: Fusion Layer
Tüm sinyalleri birleştirir:
```
Risk = α × (1 - Consistency) + β × Entropy + γ × NCD + δ × GT_Contradiction
```

Optimizasyon sonucu ağırlıklar (1000 trial):
| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| α (alpha) | 0.64 | Ground Truth Contradiction |
| β (beta) | 0.06 | Inconsistency |
| γ (gamma) | 0.08 | Entropy |
| δ (delta) | 0.23 | NCD |
| Threshold | 0.49 | Karar eşiği |

### Modül F: Answer Validator
Çoğunluk oylama ile son doğrulama:
- Yanıtlardan nihai cevabı çıkarır
- Azınlık cevaplarına ceza uygular

## Performans

TruthfulQA veri seti üzerinde (817 soru, 1000 trial Bayesian optimizasyon):

| Metrik | Değer |
|--------|-------|
| **F1 Score** | 0.77 |
| **Precision** | 0.98 |
| **Recall** | 0.64 |
| **Accuracy** | 0.79 |

## Veri Setleri

| Veri Seti | Soru Sayısı | Açıklama |
|-----------|-------------|----------|
| TruthfulQA | 817 | Yanlış inançlar, yanıltıcı sorular |
| Hallucination Test | 20 | Var olmayan kitaplar, kişiler, olaylar |
| Pilot Dataset | 20 | Silojizm, mantık bulmacaları |

## Teknolojiler

- **Deep Learning**: PyTorch, Transformers, Sentence-BERT
- **NLI Model**: `cross-encoder/nli-deberta-v3-large`
- **Optimization**: Optuna (TPE Sampler)
- **API**: OpenAI, Anthropic
- **Web**: Flask, HTML/CSS/JS
- **Visualization**: Matplotlib, Seaborn, NetworkX

## Lisans

MIT License

## Yazar

**Ömer Faruk Şener**
Gebze Teknik Üniversitesi
Danışman: Dr. Tülay AYYILDIZ

---

*Bu proje, 2024-2025 akademik yılı bitirme projesi kapsamında geliştirilmiştir.*
