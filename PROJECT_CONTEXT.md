# CDS1092 NLP — Proje bilgi devri

İnceleme tarihi: 17 Kasım 2026. Bu belge mevcut yerel workspace'in kod, notebook, veri ve sonuç dosyalarına dayanır. Training, inference, evaluation veya veri hazırlama scriptleri çalıştırılmadı; veri dosyaları ve kayıtlı predictions salt okunur biçimde incelendi. İnternet kaynaklarının güncel içeriği veya Mercury'nin mevcut durumu doğrulanmadı. “Doğrulanmış” ifadesi yerel dosyalardan doğrulamayı; “öneri” gelecekte yapılabilecek işi anlatır. Bilinmeyen ayrıntılar açıkça belirtilir. Tüm yollar aksi belirtilmedikçe bu proje köküne göredir.

## 1. Araştırma amacı ve kapsam

**Doğrulanmış tasarım:** Matematiksel word problem metninin tek bir cevabı belirlemeye yeterli bilgi içerip içermediğinin binary classification ile saptanması. Amaç matematik problemini çözmek değildir. `ANSWERABLE` yeterli ve tekil cevap sağlayan bilgi; `UNANSWERABLE` eksik, belirsiz, çelişkili veya yetersiz bilgi anlamındadır. Positive class `UNANSWERABLE`; bütün training scriptlerinin `LABEL_TO_ID` sabiti `{"ANSWERABLE": 0, "UNANSWERABLE": 1}` biçimindedir. Kaynak: `README.md`, `src/05_train_bert_run02.py::compute_metrics`.

SUM üzerinde fine-tuned encoder modelleri ile aynı sorular üzerinde general-purpose zero-shot LLM yaklaşımı karşılaştırılacaktır. Planlanan encoder'lar BERT-base, MathBERT, DeBERTa-v3-base ve ModernBERT-base'dir. SUM internal test, UMWP external generalization içindir; private/fresh değerlendirme kümesi README'de olasılık olarak geçer. Uygulanmış encoder yalnızca BERT'tir; MathBERT/DeBERTa/ModernBERT/LLM script veya sonuçları yoktur (`src/`, `results/` envanteri).

Proje dışındaki ilgili araştırma taslağı `../dokumentation_01/term-paper/documentation.tex` üç soru tanımlar: FF1 encoder–zero-shot LLM performansı; FF2 SUM→UMWP generalization ve LLM karşılaştırması; FF3 hata örüntüleri ve veri/fine-tuning/prompting iyileştirmeleri. Buradaki literatür boşluğu iddiası taslağın beyanıdır; bu incelemede bağımsız literatür taraması yapılmadı. Resmî, yönlü H1/H2 hipotezleri veya istatistiksel kabul ölçütleri bulunmadı. Domain specialization'ın faydası ve cross-dataset performans kaybı araştırılabilecek beklentilerdir; kanıtlanmış hipotez olarak sunulmamalıdır.

Taslakta beş LLM adı plan olarak verilmiştir (GPT-6.1 Terra, Gemini 3.8 Flash, Llama 4 Scout, Qwen3.6-27B, Apertus v1.5-8B); bunlar çalıştırılmış veya erişilebilirliği doğrulanmış model kimlikleri değildir. Nihai model/API sürümleri, prompt metni, generation parametreleri ve parsing kuralları bilinmiyor. Tasarım yalnızca örneksiz ortak prompt ile `ANSWERABLE`/`UNANSWERABLE` çıktısı istiyor. Kaynak: `../dokumentation_01/term-paper/documentation.tex`, Methodik bölümü.

Ekip: Murat Arikan ve Sinan Özdere; danışman Prof. Corsin Capol (`README.md`). Encoder ve LLM yaklaşımının farklı eğitim verisi, boyut ve pretraining koşulları vardır; sonucu yalnız mimari farkına bağlamamak taslağın metodolojik yaklaşımıdır.

## 2. Workspace ve önemli dosyalar

Yerel proje: `C:/Users/Murat Arikan/dev/school/03_hs_26/NLP/CDS1092_NLP_Project`. Ayrı Git repository; inceleme başlangıcında HEAD `f8d5e2d` (`Add BERT Run 05 and Run 06 validation results`). Başlangıçta `notebooks/03_bert_training_analysis.ipynb` zaten modified durumundaydı; mevcut yerel sürümü incelendi ve korundu. Local origin fetch/push URL'si `https://github.com/Janissary09/CDS1092_NLP_Project.git`; README Mercury için SSH remote tarif ediyor. Bunlar farklı makinelerdeki remote ayarları olabilir.

```text
CDS1092_NLP_Project/
├── .git/                         Git metadata; araştırma çıktısı değildir
├── .gitignore
├── README.md
├── requirements.txt
├── PROJECT_CONTEXT.md            Bu bilgi devri
├── data/
│   ├── .gitkeep
│   ├── raw/umwp/StandardDataset.jsonl
│   └── processed/
│       ├── sum/{train,validation,test}.csv
│       └── umwp/test.csv
├── notebooks/
│   ├── 01_sum_dataset_analysis.ipynb
│   ├── 02_umwp_dataset_analysis.ipynb
│   └── 03_bert_training_analysis.ipynb
├── src/
│   ├── 01_inspect_sum.py
│   ├── 02_analyze_sum_duplicates.py
│   ├── 03_prepare_sum.py
│   ├── 04_train_bert_run01.py
│   ├── 05_train_bert_run02.py
│   ├── 06_train_bert_run03.py
│   ├── 07_train_bert_run04.py
│   ├── 08_prepare_umwp.py
│   ├── 09_evaluate_bert.py
│   ├── 10_train_bert_run05.py
│   └── 11_train_bert_run06.py
└── results/
    ├── .gitkeep
    └── bert/
        ├── run_01/{training_history.csv,validation_metrics.json}
        ├── run_02/
        │   ├── config.json
        │   ├── training_history.csv
        │   ├── validation_metrics.json
        │   ├── sum_test/{metrics.json,confusion_matrix.csv,predictions.csv}
        │   └── umwp/{metrics.json,confusion_matrix.csv,predictions.csv,duplicate_analysis.csv}
        └── run_03 … run_06/       Her birinde config.json,
                                  training_history.csv,validation_metrics.json
```

`models/` bu yerel workspace'te **yoktur**. SLURM/sbatch/shell job scriptleri, `.out`/`.log` job kayıtları, environment lockfile veya ayrı test paketi bulunmadı. Yukarıdaki ağaç `.git/` iç yapısı hariç mevcut araştırma dosyalarını kapsar. `../dokumentation_01/` ayrı dokümantasyon repository'sidir: `README.md`, `.gitignore`, `LICENSE`, `term-paper/documentation.tex`, `title.tex`, `packages_and_configuration.tex`, `library.bib`, `documentation.pdf`. Bu belgedeki taslak bilgileri `.tex`/`.bib` kaynaklarından alınmıştır; PDF görsel olarak incelenmedi.

| Dosya | Görev ve önemli giriş noktaları |
| --- | --- |
| `src/01_inspect_sum.py` | `main`: publisher Parquet dosyalarını ayrı ayrı okur; farklı schema'ların tek loader'da birleşmesini önler; örnek/schema basar, ayrı dataset yazmaz. |
| `src/02_analyze_sum_duplicates.py` | `load_occurrences`, `has_conflict`, `main`: exact-text overlap ve label conflict analizi; salt okunur. |
| `src/03_prepare_sum.py` | `build_groups`, `assign_groups`, `expand`, `validate`, `main`: leakage-safe SUM CSV üretimi. |
| Altı `src/*train_bert*.py` | Her birinde `CONFIG`, `load_split`, `compute_metrics`, `main`; ayrı deney/output adresleri. Run 01 adres istisnası aşağıdadır. |
| `src/08_prepare_umwp.py` | `question_statistics`, `validate_text`, `main`: raw JSONL→değiştirilmemiş benchmark CSV; `require` kontrolleri. |
| `src/09_evaluate_bert.py` | `load_dataset`, `calculate_metrics`, `main`: yalnız frozen BERT Run 02; `--dataset sum|umwp`. |
| `results/bert/run_*/config.json` | Run 02–06 training öncesi kaydedilmiş ayarlar ve Mercury absolute data paths. |
| `results/bert/run_*/training_history.csv` | Epoch training loss, gradient norm, LR; validation metrikleri; training özeti; sonunda best checkpoint tekrar evaluation satırı. |
| `results/bert/run_*/validation_metrics.json` | Checkpoint path/F1, yeniden evaluation sonucu, configuration, torch/transformers sürümleri. |
| `README.md` | Amaç, eski durum özeti, setup ve Mercury/Git kullanım tarifi; güncellik sınırlamaları aşağıdadır. |

## 3. SUM veri kaynağı, preprocessing ve leakage

Kaynak kimliği `lime-nlp/Synthetic_Unanswerable_Math`; kodda kullanılan URL `https://huggingface.co/datasets/lime-nlp/Synthetic_Unanswerable_Math/resolve/main/synthetic_unanswerable_math.parquet` (`src/03_prepare_sum.py::SOURCE_URL`). Kaynak 36.480 pair; alanlar `answerable_question`, `unanswerable_question`, `ground_truth`. `ground_truth` model girdisi değildir ve final CSV'de tutulmaz. Publisher'ın ayrı `test.parquet` dosyası inceleme scriptinde okunur, ancak deney split'i üretiminde kullanılmaz. README bu publisher test dosyasını 284 satırlık farklı setup olarak tarif eder; yerel kopyası yoktur, sayısı yeniden doğrulanmadı.

`src/03_prepare_sum.py` algoritması:

1. Generic `datasets.load_dataset("parquet", data_files={"train": SOURCE_URL}, split="train")`; 36.480 pair ve beklenen alanlar doğrulanır.
2. `build_groups`: original row index `pair_id` olur. Aynı exact question text'ini paylaşan pair'ler union-find ile, transitif bağlantılar dahil birleştirilir. Her iki question alanı da bu indekse girer.
3. `assign_groups`: `random.Random(42).shuffle`, sonra stable descending group-size sort; her group kalan pair açığı en yüksek split'e gider. Bu greedy heuristic group bütünlüğünü önceler; hedef sayılara ulaşmayı genel durumda garanti etmez.
4. `expand`: split atandıktan sonra her pair iki örneğe açılır; cevaplanabilir metin `ANSWERABLE`, diğer metin `UNANSWERABLE`. Her split seed 42 ile shuffle edilir. Text normalization, trim, lowercase veya duplicate silme yapılmaz.
5. `validate`: her pair için tam iki kayıt ve birer label, özgün text eşitliği, group bütünlüğü, tüm pair'lerin korunması, her split'te 50/50 sınıf dengesi, split'ler arası pair ve exact-text disjointness doğrulanır. Başarıdan sonra CSV yazılır. Script mevcut CSV'yi overwrite edebilir; deneyler başladıktan sonra gelişigüzel yeniden çalıştırılmamalıdır.

Mevcut CSV'lerde salt okunur sayımla doğrulanan durum:

| Yol | Pair | Örnek | ANSWERABLE | UNANSWERABLE | Unique exact text | Tekrarlanan distinct text | Conflicting distinct text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `data/processed/sum/train.csv` | 25.536 | 51.072 | 25.536 | 25.536 | 50.019 | 733 | 1 |
| `data/processed/sum/validation.csv` | 5.472 | 10.944 | 5.472 | 5.472 | 10.943 | 1 | 1 |
| `data/processed/sum/test.csv` | 5.472 | 10.944 | 5.472 | 5.472 | 10.944 | 0 | 0 |

Her pair tam iki kayıttır; bütün split çiftlerinde pair ve exact-text overlap sıfırdır. Duplicate/conflicting text'in **split içinde kalması**, cross-split leakage bulunmasıyla aynı şey değildir. Mevcut conflicting label örnekleri korunmuştur. Near-duplicate, semantic/template overlap veya pretraining contamination bulunmadığı kanıtlanmış değildir. Kaynak Parquet revision/hash'i sabitlenmemiştir; yerel raw SUM dosyası yoktur. README eski ilk split denemesinde leakage bulunduğunu söyler; o eski split dosyaları ve sayısal leakage logları mevcut değildir.

## 4. UMWP veri kaynağı ve hazırlama

`data/raw/umwp/StandardDataset.jsonl` yerel benchmark kaynağıdır. Kaynak makale, komşu `../dokumentation_01/term-paper/library.bib::Sun2024` kaydında *Benchmarking Hallucination in Large Language Models Based on Unanswerable Math Word Problem*, LREC-COLING 2024 olarak geçer. Raw dosyanın hangi download URL/commit/revision ile edinildiği bu workspace'te kayıtlı değildir; tahmin edilmemelidir.

`src/08_prepare_umwp.py::main` UTF-8-sig JSONL okur; `id`, `question`, `answerable`, `category`, `source` ister; 5.200 satır, unique/non-null ID, string/nonempty text ve boolean label kontrol eder. Raw'da ayrıca `answer`, `relevant_ids` bulunur. `question`→`text`, `True`→`ANSWERABLE`, `False`→`UNANSWERABLE`; çıktı sırası `id,text,label,category,source`. Text, ID, category, source ve sıra değiştirilmez. `answer`/`relevant_ids` final CSV'ye veya modele girmez.

`data/processed/umwp/test.csv`: 5.200 kayıt, 2.600/2.600 label dengesi, 5.190 unique text; 10 duplicate text grubu, bunlara ait 20 satır ve 5 conflicting label grubu. Hepsi tutulur; label düzeltme veya dedup yapılmaz. UMWP train/validation split'i yoktur ve oluşturulmamalıdır; tuning/early stopping/checkpoint selection için kullanılmaz (`src/08_prepare_umwp.py`, `notebooks/02_umwp_dataset_analysis.ipynb`, Dataset Preparation Decision).

| Source metadata | Toplam | Her sınıfta |
| --- | ---: | ---: |
| GSM8K | 3.400 | 1.700 |
| SVAMP | 1.000 | 500 |
| MultiArith | 600 | 300 |
| ASDiv | 200 | 100 |

Category metadata: answerable 2.600 satırda null; unanswerable code 1:840, 2:1.263, 3:275, 4:103, 5:119. Processed CSV'de kodlar `1.0`…`5.0`, null boş alan olarak görünür. Sayısal kod→anlamsal category eşlemesi kod/notebook'ta tanımlı değildir; bilinmiyor. Taslak eksik/belirsiz bilgi, gerçekçi olmayan koşullar, referanssız nesne, eksik soru türlerini anar ama bunları kodlara eşlemez. Kaynak: `notebooks/02_umwp_dataset_analysis.ipynb`, category/source analysis hücreleri; mevcut CSV sayımları.

Raw notebook çıktısında `answer` 2.598 kez null; iki unanswerable kayıtta non-null integer vardır; bütün answerable answer'ları list'tir. `relevant_ids` bütün answerable satırlarda null, bütün unanswerable satırlarda list'tir. Bunların modele verilmesi doğrudan label ipucu yaratabilir; mevcut pipeline yalnız `text` kullanır. SUM–UMWP cross-dataset exact/semantic overlap için kaydedilmiş bir kontrol bulunmadı; external değerlendirme ifadesi tek başına overlap yokluğunu kanıtlamaz.

## 5. Ortak BERT training pipeline

Esas referans `src/05_train_bert_run02.py::main`, `load_split`, `compute_metrics`. Run 03/04/05/06 scriptleri aynı akışı kullanır; farkları run kimliği/output path, LR ve epoch budget'tır.

- `load_split` sadece `train` ve `validation` kabul eder. `pair_id,text,label` zorunlu; empty/missing değer, non-string text veya bilinmeyen label reddedilir. Model dataset'ine yalnız `text` ve numeric `labels` taşınır. `pair_id` ve matematik cevabı input değildir.
- `WORLD_SIZE != 1` reddedilir; mevcut `CUDA_VISIBLE_DEVICES` korunur, yoksa `0` atanır; görünür GPU sayısı birden fazlaysa hata verir. Ordinary `python` beklenir, `torchrun` değil. CUDA yoksa CPU'ya izin verir; GPU zorunluluğu teknik olarak enforce edilmez.
- `set_seed(42)` model yüklenmeden önce çağrılır. `BertTokenizerFast.from_pretrained("bert-base-uncased")`; tokenizer çağrısı `truncation=True,max_length=256`. `Dataset.from_pandas(...,preserve_index=False).map(...,batched=True,remove_columns=["text"],load_from_cache_file=False)`; dynamic padding `DataCollatorWithPadding` ile batch içinde yapılır.
- `AutoModelForSequenceClassification.from_pretrained` aynı model kimliğinden `num_labels=2`, explicit `label2id/id2label` ile yükler. Classification task için fine-tuning uygulanır; encoder freeze eden kod yoktur.
- `Trainer` ve `TrainingArguments`: train/eval per-device batch 32, gradient accumulation 1, `adamw_torch`, linear scheduler, weight decay 0.01, epoch evaluation/save/logging, `report_to="none"`, seed/data_seed 42.
- Her epoch `ceil(51072/32)=1596` optimizer step. `planned_optimizer_steps=1596*max_epochs`; `warmup_steps=ceil(planned_steps*0.05)`. Dolayısıyla 5 epoch: 7.980 step/399 warmup; 7 epoch: 11.172 step/559 warmup. Max epoch değişince ilk epoch'lardaki LR schedule da değişir; uzun run kısa run'ın sonuna iki epoch eklemek değildir.
- `load_best_model_at_end=True`, `metric_for_best_model="f1"`, `greater_is_better=True`, `save_total_limit=2`. `EarlyStoppingCallback` patience 2, threshold 0.0; validation binary F1 iyileşmeyen iki evaluation sonrası durabilir. Loss seçim kriteri değildir.
- `compute_metrics`: logits argmax; Accuracy; positive label 1 için binary Precision/Recall/F1; `zero_division=0`. `best_model_metric="validation_f1"` açıklayıcı CONFIG kaydıdır; Trainer'ın kullandığı gerçek anahtar `f1`'dır.
- `trainer.train()` sonrası `trainer.evaluate(validation_data)`, `trainer.save_model`, `tokenizer.save_pretrained`; history `trainer.state.log_history` üzerinden CSV'ye, best checkpoint ve environment bilgileri JSON'a yazılır. Run 02–06 ayrıca training öncesi `config.json` kaydeder. Training scriptlerinde mevcut output'u koruyan overwrite guard veya resume CLI'si yoktur; tekrar çalıştırma mevcut deneyi bozabilir.

**Reproducibility sınırları:** Bütün run raporlarında `torch=2.11.0+cu128`, `transformers=5.22.2`, `device=cuda:0` kayıtlıdır. Python/CUDA driver/GPU modeli, `datasets`/`numpy`/`pandas`/`scikit-learn`/`accelerate` sürümleri ve model/tokenizer commit hash'leri raporlarda yoktur. `requirements.txt` yalnız `torch,transformers,datasets,pandas,numpy,scikit-learn` içerir; version pin yoktur, scriptlerin belirttiği `accelerate` listede eksiktir. Notebook ortamı Python 3.13.7 olarak görünür; bu Mercury training Python sürümünü kanıtlamaz. Training'de explicit deterministic algorithms veya precision (`fp16`/`bf16`) ayarı yoktur; library default'larını sabit deney ayarı gibi varsaymamak gerekir. Evaluation scripti daha güçlü determinism ayarları kullanır.

## 6. Altı BERT deneyi ve kayıtların yorumlanması

Sonraki tablolardaki sayılar doğrudan `results/bert/run_XX/validation_metrics.json`, `config.json` (varsa) ve `training_history.csv` kaynaklıdır; türetilen best epoch history'nin maksimum F1 satırından alınmıştır. Decimal noktası teknik formatı korur. Ayrıntılı epoch metrikleri ekte yer alır.

Run 01 özel durumu: `src/04_train_bert_run01.py::MODEL_DIR` → `models/bert/best_model`, `CHECKPOINT_DIR` → `models/bert/checkpoints`, `RESULTS_DIR` → `results/bert`. Mevcut compact sonuçlar ise `results/bert/run_01/` altındadır. Sonuçların ne zaman/nasıl bu alt dizine taşındığı bilinmiyor. Bu script değiştirilmeden tekrar çalıştırılırsa `run_01/` altına yazmaz; ayrı `config.json` üretmez. Rapor configuration'ı mevcuttur.

Run 02 validation F1 açısından altı run'ın en iyisidir. Run 04 Accuracy'si daha yüksek olsa da seçim kriteri F1'dır. Run 03, Run 02'ye göre yalnız LR 2e-5; Run 04 LR 4e-5; Run 05 5e-5 ve 5 epoch; Run 06 5e-5 ve 7 epoch. Run 01→03 ve Run 05→06 karşılaştırmalarında epoch budget ile birlikte warmup ve decay schedule da değişir. `notebooks/03_bert_training_analysis.ipynb` Run 05/06'yı daha yüksek LR için exploratory ek deneyler olarak niteler. Git history'de Run 02 test sonuçlarının commit'i `89c9fde`, Run 05/06 kodu `bf04ddb`, sonuçları `f8d5e2d` sırasındadır. Bu commit sırası gerçek training zamanlarını tek başına kanıtlamaz; ancak ek deneylerin test sonuçları repository'ye girdikten sonra eklendiği açıktır. “Bütün altı run test görülmeden planlandı” iddiası desteklenmez.

### Run 06 tutarsızlığı

`results/bert/run_06/training_history.csv`: epoch 5 F1 **0.8509703405346027**, loss 0.4683046042919159; best checkpoint `checkpoint-7980`. Epoch 6 ve 7 F1 daha düşük; iki iyileşmeme patience koşulu max epoch sınırıyla aynı noktaya denk gelir. Dolayısıyla yalnız logdan hangisinin bitişi tetiklediği kesin ayrıştırılamaz.

`results/bert/run_06/validation_metrics.json::best_checkpoint_validation_f1` aynı 0.8509703405346027 değeridir; fakat `best_validation_metrics` yeniden evaluation sonucu F1 **0.8506903172716467**, Accuracy 0.8507858187134503, Precision 0.851235132662397, Recall 0.8501461988304093 ve loss 0.4714003801345825 verir. Bu son evaluation'ın `epoch=7` metadata'sı checkpoint'in epoch 7 olduğunu göstermez; training state'in son epoch'udur. Gerçek best epoch checkpoint step / 1596 = 5'tir. Sebep bilinmiyor; best model yükleme, numerical determinism ve serialization/restore gelecekte incelenmelidir. Değerlerden biri sessizce diğerinin yerine konmamalıdır.

Run 02'de training loss epoch 1→7 0.523006→0.103833 düşerken validation loss epoch 3'te 0.374826'dan epoch 7'de 0.629921'e çıkar; F1 aynı aralıkta iyileşir (`results/bert/run_02/training_history.csv`). Bu overfitting veya confidence/calibration sorunu için bir belirtidir; F1 seçim kuralının hatalı olduğunu veya belirli bir sebebi tek başına kanıtlamaz. Tek seed kullanıldığı için küçük run farklarının istatistiksel anlamlılığı bilinmiyor.

## 7. SUM test ve UMWP evaluation

`src/09_evaluate_bert.py` frozen Run 02'ye hardcoded'dur: `MODEL_PATH=models/bert/run_02/best_model`, `CONFIG_PATH=results/bert/run_02/config.json`. Dataset CLI zorunludur. `load_dataset` yalnız seçilen `data/processed/{sum|umwp}/test.csv` dosyasını okur; `keep_default_na=False` literal text/metadata korur; boyut, label ve text kontrolü; UMWP için unique ID ve sınıf dengesi kontrolü yapar.

`main` config'te Run 02 ve label mapping doğrular; max_length ve batch'i kayıtlı config'ten alır (256,32). Model/tokenizer sadece yerel best_model içinden `local_files_only=True` ile yüklenir. Seed 42; cuDNN benchmark kapalı, deterministic açık, `torch.use_deterministic_algorithms(True)`; entry point'te torch importundan önce `CUBLAS_WORKSPACE_CONFIG=:4096:8`. CUDA varsa GPU, yoksa CPU seçer. `model.eval()` ve `torch.no_grad()` altında yalnız `text` tokenize edilir; batch padding, truncation 256; predictions logits argmax, probabilities softmax. Threshold tuning/retraining yoktur.

`calculate_metrics` Accuracy, positive-class Precision/Recall/F1, iki label üzerinden macro-F1, correct/incorrect counts hesaplar. Confusion matrix sıra `[ANSWERABLE, UNANSWERABLE]`, rows=true/columns=predicted; `[[TN,FP],[FN,TP]]`.

| Dataset | N | Accuracy | Precision (U) | Recall (U) | F1 (U) | Macro-F1 | Correct / Incorrect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SUM test | 10944 | 0.847953216374269 | 0.8545623836126629 | 0.8386330409356725 | 0.8465227817745803 | 0.847940007581116 | 9280 / 1664 |
| UMWP | 5200 | 0.6761538461538461 | 0.7301507537688442 | 0.5588461538461539 | 0.6331154684095861 | 0.6716351868726071 | 3516 / 1684 |

Kaynaklar: `results/bert/run_02/sum_test/metrics.json`, `results/bert/run_02/umwp/metrics.json`. Predictions dosyaları salt okunur sayıldığında satır ve correct sayıları bu raporlarla uyuşur.

| Dataset | TN | FP: A→U | FN: U→A | TP |
| --- | ---: | ---: | ---: | ---: |
| SUM test | 4691 | 781 | 883 | 4589 |
| UMWP | 2063 | 537 | 1147 | 1453 |

UMWP'deki 1.147 FN, eksik bilgiyi fark edememe yönündeki temel araştırma hatasıdır. SUM→UMWP F1 düşüşü yaklaşık 21.34 yüzde puan; recall düşüşü yaklaşık 27.98 yüzde puandır. Bu veriler distribution değişiminde zayıflamayı gösterir; sebebini izole etmez.

Output'lar `results/bert/run_02/sum_test/` ve `results/bert/run_02/umwp/`: `metrics.json`, `confusion_matrix.csv`, `predictions.csv`; yalnız UMWP'de ayrıca `duplicate_analysis.csv`. Predictions ortak kolonları `text,true_label,predicted_label,correct,probability_answerable,probability_unanswerable`; SUM'da `pair_id`; UMWP'de `id,category,source` bulunur. Bunlar metadata'dır, model girdisi değildir. UMWP duplicate raporu: 10 repeated text, 5 conflicting ground-truth text, inconsistent prediction group 0. Aynı text aynı prediction almıştır; conflicting label çiftlerinde en az bir kaçınılmaz hata vardır. Official ana sonuçlar dedup edilmemiş 5.200 kayıt üzerinden kalmalıdır.

Script output dizini zaten varsa `FileExistsError` ile overwrite'ı reddeder. İki output dizini de mevcut; mevcut komutlar yeniden çalıştırılırsa (model bulunsa bile) overwrite guard'a takılır. Yerelde `models/` olmadığı için önce missing-model hatası beklenir. Diğer BERT run'larının SUM test/UMWP sonuçları bulunmuyor. Category/source bazlı evaluation sonuçları ayrı artifact olarak henüz yoktur; mevcut predictions metadata'sı buna olanak sağlar.

## 8. Mercury HPC ve Git iş akışı

**README'de belgelenmiş, bu oturumda uzaktan doğrulanmamış:** Mercury repository `~/projects/CDS1092_NLP_Project`, Conda environment `cds1092`; GitHub SSH ayarı `git@github.com:Janissary09/CDS1092_NLP_Project.git`. Sonuç JSON'larında absolute kök `/home/arikanmurat@edu.local/projects/CDS1092_NLP_Project` olarak kayıtlıdır. README A100 erişimi ve CUDA doğrulamasının yapıldığını söyler; run JSON'ları yalnız `cuda:0` kanıtlar, GPU modelini kaydetmez.

README'deki interactive allocation ve environment komutları (burada çalıştırılmadı):

```bash
git pull
srun --partition=students --gres=gpu:a100:1 --time=02:00:00 --pty bash
conda activate cds1092
cd ~/projects/CDS1092_NLP_Project
python -c "import torch; print(torch.cuda.get_device_name(0)); print('CUDA:', torch.cuda.is_available())"
```

Mevcut scriptlerin doğrudan çalıştırma komutları:

```bash
python src/04_train_bert_run01.py
python src/05_train_bert_run02.py
python src/06_train_bert_run03.py
python src/07_train_bert_run04.py
python src/10_train_bert_run05.py
python src/11_train_bert_run06.py
python src/09_evaluate_bert.py --dataset sum
python src/09_evaluate_bert.py --dataset umwp
```

Bunlar mevcut kodun kullanımını kaydeder; tamamlanmış run'ları tekrar çalıştırma talimatı değildir. Run-specific yeni yollar belirlenmeden training tekrar edilmemelidir. GPU allocation içinden tek GPU ile ordinary `python` kullanılmalıdır (`src/*train_bert*.py::main`). Login node üzerinde GPU işi çalıştırmama kuralı README'de Slurm zorunluluğu olarak belirtilir.

**Bilinmeyenler:** Gerçekte kullanılan `sbatch` dosyaları, job ID, stdout/stderr, CPU/RAM isteği, CUDA module setup, hostname/login komutu, Conda oluşturma komutu ve tam environment export mevcut değildir. Bu yüzden var olmayan bir SLURM scripti mevcutmuş gibi gösterilmez. A100 memory türü ve queue'nun güncel kullanılabilirliği bilinmiyor.

**Öneri — gelecekte oluşturulabilecek batch şablonu:** Aşağıdaki içerik mevcut artifact değildir; partition/GPU/time README örneğinden türetilmiştir. Yeni MathBERT scripti hazırlanmadan kullanılamaz; CPU/RAM değerleri cluster politikası ve ölçümle ayrıca belirlenmelidir.

```bash
#!/bin/bash
#SBATCH --job-name=mathbert01
#SBATCH --partition=students
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
# Conda shell initialization: Mercury kurulumuna göre doğrulanmalı.
conda activate cds1092
cd ~/projects/CDS1092_NLP_Project
python -u src/12_train_mathbert_run01.py  # Önerilen dosya; henüz yok.
```

Önerilen kullanım: submit öncesi `mkdir -p logs`, ardından hazırlanmış script için `sbatch <job-script.sh>`; takip `squeue -u "$USER"`, çıktı `tail -f logs/<job-output>`; bunlar cluster üzerinde doğrulanmış geçmiş komutlar değildir. Slurm GPU mask'ini script korur; allocation mask'ini gelişigüzel overwrite etmemek gerekir.

**Sonuçları geri alma:** README'nin var olan modeli compact sonuçları GitHub'a commit/push etmek, yerelde pull etmektir. Mercury'de önce `git status`, sonra yalnız yeni run'ın source/config/history/metrics/notebook dosyalarını açık yollarla `git add`, `git commit`, `git push`; diğer makinede `git pull`. Model weights bu yolla taşınmaz. `.gitignore` `/models/`, checkpoints, ağırlık uzantıları, HF cache ve environment'ları dışlar; SUM fixed CSV ve raw/processed UMWP için açık istisnalar vardır. Büyük weights için Mercury'de backup konumu/retention ve gerçekleşmiş transfer komutları bilinmiyor. Öneri: gerçek SSH hedefi öğrenildiğinde `scp -r <user@host>:<remote-project>/models/<model>/<run>/best_model <local-destination>` veya kurumun onaylı transfer yöntemi; placeholder'lar gerçek bağlantı bilgisi değildir. Compact sonuçların Git'te bulunması modelin halen Mercury'de korunduğunu kanıtlamaz.

## 9. Notebook'lar ve dokümantasyon sorunları

### `notebooks/01_sum_dataset_analysis.ipynb`

Final CSV boyut/label/missing checks, word count (`length_statistics`), truncation olmadan BERT token count, 128/256/512 truncation oranları, kısa/uzun örnekler ve histogramlar. Kaydedilmiş kernel `fhgr (3.13.7)`. Dataset yazmaz/training yapmaz; ilk tokenizer yüklemesi network/cache kullanabilir. Kayıtlı çıktıda train token mean 69.43, validation 70.61, test 70.95; max 2186/1029/1047. 256 sınırını aşan örnekler train 582 (%1.14), validation 138 (%1.26), test 119 (%1.09). Bunlar notebook'un kayıtlı çıktısıdır; tokenizer yeniden çalıştırılmadı.

Notebook test text'lerini, label'larını, örneklerini ve uzunluk dağılımını inceler. Dolayısıyla “test'e hiç bakılmadı” demek doğru değildir; training scriptlerinin test kullanmaması ve test metriklerinin seçimde kullanılmaması daha dar iddialardır. Token limitinin test istatistiklerine dayanarak seçilip seçilmediği bilinmiyor. HF unauthenticated/symlink uyarıları ve full sequence >512 uyarısı kayıtlıdır; uzunluk ölçümü model inference olmadığı için bu tek başına hata değildir. Matplotlib/Jupyter/IPython dependencies `requirements.txt` içinde yoktur.

### `notebooks/02_umwp_dataset_analysis.ipynb`

Raw structure, null fields, bool labels, category/source dağılımları, text quality, duplicate/conflict, answer/relevant_ids tipleri ve final validation; `check` fonksiyonu PASS/WARNING basar. Başlangıç hücreleri import ve root discovery içerir; kayıtlı execution 1–12. İşlem read-only'dir; önerilen processed schema ve official benchmark'ı değiştirmeme kararı vardır. Processed dosya artık bulunduğundan “will contain” türü gelecek zaman ifadeleri tarihsel kalmıştır. Mevcut çıktılarda Python error output'u bulunmadı.

### `notebooks/03_bert_training_analysis.ipynb`

Altı run setup/validation, Run 02 seçimi, SUM/UMWP metrikleri ve yorumlar. **Temiz kernel'de self-contained değildir:** ilk code cell doğrudan `pd.DataFrame`, `runs`, `RUN_NAMES` kullanır; bunları import/initialize eden hücreler mevcut notebook'ta yoktur. Kayıtlı execution 7/8/9 ve çıktıların varlığı dış/kernel state'e bağlı geçmiş çalışmayı gösterir; temiz çalıştırılabilirliği kanıtlamaz. Kaydedilmiş hata output'u yoktur, fakat static code bu bağımlılığı açıkça gösterir.

`epoch_metrics(history)` `groupby("epoch").last()` kullanır. Son best-model re-evaluation son training epoch'una yazıldığı için özellikle Run 06'nın gerçek epoch 7 metriklerinin yerine restored checkpoint metriklerini koyar. En yüksek F1 epoch 5 olarak yine bulunur; epoch 7 trajectory'si bu işlemle çarpıtılır. Öneri: epoch evaluation ile post-training restored-model evaluation event'lerini ayırmak. Test metrikleri ve confusion matrix code cell'de hardcoded; JSON'lardan otomatik okunmuyor, ileride drift riski var. Notebook çalışma ağacında zaten modified idi; bilgi devri sırasında düzeltilmedi.

### Güncelliğini yitirmiş beyanlar

`README.md` yalnız Run 01/02 completed gösterir, Run 03 için “not yet trained” der; gerçekte Run 03–06 sonuçları mevcut. SUM test/UMWP değerlendirmesi future work gösterilse de Run 02 sonuçları vardır. Repository tree eksiktir; `accelerate` açıklamada bulunurken requirements listesinde yoktur. Ayrıca `../dokumentation_01/term-paper/documentation.tex` official SUM test split'in aynen kullanılacağını söyler; uygulanmış `src/03_prepare_sum.py` publisher'ın training Parquet'inden yeni 70/15/15 split üretir. Rapor yazılırken uygulanan deney düzeni esas alınmalı, taslak bu konuda gelecekte düzeltilmelidir.

## 10. Tamamlanan işler, açık işler ve önerilen sıra

**Artifact ile doğrulanmış tamamlananlar:** SUM inspection/duplicate analizi ve final grouped split kodu; sabit SUM CSV'leri; raw ve processed UMWP; üç analiz notebook'u (üçüncünün clean-kernel sorunu saklı kalmak üzere); altı BERT run configuration/history/validation raporu; Run 02 SUM ve UMWP predictions/metrics/confusion; duplicate consistency analizi. Mercury environment ve SSH kurulumunun tamamlandığı README beyanıdır, uzaktan yeniden doğrulanmadı.

**Henüz artifact bulunmayan işler:** MathBERT, DeBERTa-v3, ModernBERT training/evaluation; zero-shot LLM ortak prompt/API runner ve sonuçları; private test; category/source bazlı detaylı error analysis; multi-seed belirsizlik analizi; version lock ve model/data revision manifest; gerçek SLURM script/log arşivi; final araştırma sonuç raporu.

**Önerilen öncelik sırası:**

1. Mercury'de Run 02 best_model ve diğer checkpoint'lerin varlığını, backup/retention durumunu kontrol et; immutable model hash/environment manifest oluştur. Yerelde model olmadan frozen sonuç yeniden üretilemez.
2. Run 06 restore/evaluation farkını aynı checkpoint ve aynı validation verisiyle daha sonra incele; training'i tekrar etmek ilk adım değildir. İnceleme sonucu gelmeden mevcut kayıtları “düzeltme”.
3. Notebook 03 import/load hücrelerini ve event ayrımını ayrı bir değişiklikte düzelt; README/araştırma taslağını actual split ve completed deneylerle uyumlu hale getir.
4. MathBERT'i aşağıdaki önceden belirlenmiş comparable koşullarla başlat; BERT sonuçlarını yeniden overwrite etme. Test sonuçları zaten bilindiği için sonraki model protokolünü önceden yazılı dondur.
5. Aynı frozen test/UMWP örnekleri için diğer encoder ve LLM pipeline'larını tamamla. UMWP'yi tuning set'e dönüştürme. Yeni prompt/threshold tasarımını test outcome'a göre optimize etme.
6. Mevcut predictions üzerinden category/source ve FN/FP analizi, qualitative örnekler, truncation ile hata ilişkisi ve benchmark conflicts etkisini raporla. Secondary dedup/conflict-sensitive analiz varsa official primary sonucu ayrı tut. Multi-seed/paired uncertainty analizini kaynak bütçesi ve araştırma protokolüne göre planla.

## 11. MathBERT'e başlamak için uygulama planı

Bu bölüm **öneridir**; MathBERT scripti/model kimliği/tokenizer seçimi henüz mevcut projede belirlenmemiştir. “MathBERT” adı tek başına tam Hugging Face repository/revision veya vocab variant'ını belirlemez. Belirli bir model ID'si uydurulmamalıdır. Deneye başlamadan model card, license, pretrained checkpoint, tokenizer/vocab, casing ve sequence length uyumu resmi kaynaktan doğrulanmalı; exact repository ID/revision rapora kaydedilmelidir. Bu inceleme sırasında model indirilmedi.

### Karşılaştırılabilir ana deney

Başlangıç scripti olarak `src/05_train_bert_run02.py` akışını **yeni bir dosyaya** uyarlamak uygun olur: önerilen ad `src/12_train_mathbert_run01.py`, yeni `models/mathbert/run_01/{best_model,checkpoints}` ve `results/mathbert/run_01/`. Bu yollar henüz yoktur. BERT dosyalarını değiştirme; `load_split`, positive label, `compute_metrics`, best-F1 selection, dynamic padding ve artifact şemasını koru.

| Ayar | Önerilen comparable değer |
| --- | --- |
| Data | Aynı `data/processed/sum/train.csv` ve `validation.csv`; split yeniden üretme |
| Model/tokenizer | Doğrulanmış MathBERT checkpoint ve **ona ait** tokenizer/revision |
| Labels | ANSWERABLE=0, UNANSWERABLE=1; binary UNANSWERABLE F1 |
| Max length | 256; model limit uyumu doğrulanmalı |
| Train/eval batch | Tek GPU üzerinde 32/32; accumulation 1; effective batch 32 |
| LR / max epoch | 3e-5 / 7 (seçilmiş BERT Run 02 ile aynı) |
| Weight decay / warmup | 0.01 / 0.05 → 559 warmup step |
| Optimizer / scheduler | adamw_torch / linear; planned steps 11172 |
| Seed/data_seed | 42/42 |
| Selection | Her epoch validation; max binary F1, patience 2, threshold 0 |
| Checkpoints | load_best_model_at_end=True; save_total_limit=2 |
| Precision/runtime | BERT koşullarını explicit kaydet; aynı environment/GPU mümkünse koru |

Model/tokenizer değişimi beklenen ana deney farkıdır; farklı vocab nedeniyle aynı 256 token aynı metin miktarını kapsamayabilir. Öneri: train/validation token uzunluklarını ve truncation oranlarını kaydet; bu analizle test'e göre limit seçme. `BertTokenizerFast`'ı yalnız BERT ailesi adı benziyor diye zorla kullanma; seçilen checkpoint'in tokenizer class'ını model card/config ile doğrula, uygun olduğunda `AutoTokenizer` kullan. `AutoModelForSequenceClassification(...,num_labels=2,label2id=...,id2label=...)` desteğini ve classification head initialization'ını kontrol et. Bilinmeyen checkpoint için `trust_remote_code` gerekliliği varsayılmamalıdır.

Bir kez çalıştırılacak yeni run için training öncesi config ve immutable source revision kaydet; output varsa duracak koruma eklemek önerilir. Aynı library sürümlerini kullanmak amaçlanmalı; uyumsuzluk nedeniyle değişirse exact sürüm ve neden kayıt altına alınmalı. Başka GPU/batch/precision gerekiyorsa sapmayı belirt; effective batch'i korumak için accumulation kullanılması bile BERT'in gerçek step/scheduler koşullarıyla tamamen aynı değildir. OOM durumunda sessiz batch küçültme comparable koşulu bozar.

### MathBERT evaluation uyarlaması

`src/09_evaluate_bert.py` doğrudan MathBERT'e uygulanamaz: model/config/output adresleri, Run 02 assertion ve tokenizer class'ı hardcoded'dur. Öneri: yeni `src/13_evaluate_mathbert.py` veya model/run parametreli ayrı evaluation sürümü; BERT mevcut artifacts'ını koru. Yeni evaluator aynı `load_dataset`, `calculate_metrics`, argmax, label order, max_length/batch config okuma, local saved tokenizer/model ve output overwrite guard'ını korumalıdır. Önce SUM validation ile yalnız teknik uyum doğrulanır; model seçiminden sonra frozen SUM test ve UMWP birer kez değerlendirilir; training/tuning UMWP'ye bakmaz. `models/mathbert/run_01/best_model` içine tokenizer'ın da kaydedilmesi zorunlu operasyonel adımdır.

MathBERT için henüz çalıştırılabilir hazır komut yoktur. Yeni scriptler hazırlanıp incelendikten sonra allocation içinde önerilen komut `python src/12_train_mathbert_run01.py`; yeni evaluator tamamlandıktan sonra `python src/13_evaluate_mathbert.py --dataset sum` ve `--dataset umwp` olacaktır. Bunlar gelecekteki dosya adlarıdır. Config, history, best checkpoint metric, restored-model metric, versions, epoch/step ve model/tokenizer/data hash'lerini beraber saklamak Run 06 benzeri farkları görünür kılar. Sonuç farklarını “MathBERT daha iyi” diye yorumlamadan seed ve tuning budget farklarını açıkla: BERT altı exploratory deneye sahipken tek MathBERT run'ı eşit search budget değildir. Ana comparison matched configuration olarak, eşit budget search yapılırsa ayrı experiment olarak raporlanabilir.

## 12. Sayısal ekler ve veri parmak izleri

Aşağıdaki tablolar mevcut dosyalardan programatik olarak aktarılmıştır. Her run için exact değerlerin nihai kaynağı ilgili JSON/CSV dosyasıdır. History'nin post-training evaluation satırı epoch trajectory tablosunda ayrıca ayrılmıştır.

### 12.1 Run configuration ve best epoch

| Run / script | LR | Max / tamamlanan epoch | Warmup / planned steps | Best epoch / step | Best checkpoint F1 |
| --- | --- | --- | --- | --- | --- |
| run_01 / `src/04_train_bert_run01.py` | 2e-05 | 5 / 5.0 | 399 / 7980 | 5.0 / 7980 | 0.8431372549019608 |
| run_02 / `src/05_train_bert_run02.py` | 3e-05 | 7 / 7.0 | 559 / 11172 | 7.0 / 11172 | 0.8517332845513583 |
| run_03 / `src/06_train_bert_run03.py` | 2e-05 | 7 / 7.0 | 559 / 11172 | 7.0 / 11172 | 0.8497940503432494 |
| run_04 / `src/07_train_bert_run04.py` | 4e-05 | 7 / 7.0 | 559 / 11172 | 7.0 / 11172 | 0.8513302034428795 |
| run_05 / `src/10_train_bert_run05.py` | 5e-05 | 5 / 5.0 | 399 / 7980 | 5.0 / 7980 | 0.8458781362007168 |
| run_06 / `src/11_train_bert_run06.py` | 5e-05 | 7 / 7.0 | 559 / 11172 | 5.0 / 7980 | 0.8509703405346027 |

### 12.2 History i?inde maksimum F1 veren validation event

| Kaynak: run training_history.csv | Loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| `results/bert/run_01/training_history.csv` | 0.4437739849090576 | 0.8450292397660819 | 0.8535580524344569 | 0.83296783625731 | 0.8431372549019608 |
| `results/bert/run_02/training_history.csv` | 0.629921019077301 | 0.8518823099415205 | 0.8525911005310383 | 0.8508771929824561 | 0.8517332845513583 |
| `results/bert/run_03/training_history.csv` | 0.5452165007591248 | 0.8500548245614035 | 0.8512745277828718 | 0.8483187134502924 | 0.8497940503432494 |
| `results/bert/run_04/training_history.csv` | 0.6973326206207275 | 0.8524305555555556 | 0.8577258393618995 | 0.8450292397660819 | 0.8513302034428795 |
| `results/bert/run_05/training_history.csv` | 0.5088651180267334 | 0.846765350877193 | 0.8508042151968941 | 0.8410087719298246 | 0.8458781362007168 |
| `results/bert/run_06/training_history.csv` | 0.4683046042919159 | 0.8512426900584795 | 0.8525311812179017 | 0.8494152046783626 | 0.8509703405346027 |

### 12.3 Best model restore sonras? JSON validation metrikleri

| Kaynak: run validation_metrics.json | Loss | Accuracy | Precision | Recall | F1 | epoch metadata |
| --- | --- | --- | --- | --- | --- | --- |
| `results/bert/run_01/validation_metrics.json` | 0.4437739849090576 | 0.8450292397660819 | 0.8535580524344569 | 0.83296783625731 | 0.8431372549019608 | 5.0 |
| `results/bert/run_02/validation_metrics.json` | 0.629921019077301 | 0.8518823099415205 | 0.8525911005310383 | 0.8508771929824561 | 0.8517332845513583 | 7.0 |
| `results/bert/run_03/validation_metrics.json` | 0.5452165007591248 | 0.8500548245614035 | 0.8512745277828718 | 0.8483187134502924 | 0.8497940503432494 | 7.0 |
| `results/bert/run_04/validation_metrics.json` | 0.6973326206207275 | 0.8524305555555556 | 0.8577258393618995 | 0.8450292397660819 | 0.8513302034428795 | 7.0 |
| `results/bert/run_05/validation_metrics.json` | 0.5088651180267334 | 0.846765350877193 | 0.8508042151968941 | 0.8410087719298246 | 0.8458781362007168 | 5.0 |
| `results/bert/run_06/validation_metrics.json` | 0.4714003801345825 | 0.8507858187134503 | 0.851235132662397 | 0.8501461988304093 | 0.8506903172716467 | 7.0 |

### 12.4 Checkpoint adresleri ve training s?releri

Absolute adresler sonu? raporlar?n?n kayd?d?r; bu makinede bu checkpoint dosyalar? yoktur. S?reler history i?indeki `train_runtime` saniyeleridir; queue bekleme/download/setup s?resi olarak yorumlanmamal?d?r.

| Run | best_checkpoint | train_runtime (s) |
| --- | --- | --- |
| run_01 | `/home/arikanmurat@edu.local/projects/CDS1092_NLP_Project/models/bert/checkpoints/checkpoint-7980` | 2116.1642 |
| run_02 | `/home/arikanmurat@edu.local/projects/CDS1092_NLP_Project/models/bert/run_02/checkpoints/checkpoint-11172` | 2967.4375 |
| run_03 | `/home/arikanmurat@edu.local/projects/CDS1092_NLP_Project/models/bert/run_03/checkpoints/checkpoint-11172` | 2932.4643 |
| run_04 | `/home/arikanmurat@edu.local/projects/CDS1092_NLP_Project/models/bert/run_04/checkpoints/checkpoint-11172` | 2933.9415 |
| run_05 | `/home/arikanmurat@edu.local/projects/CDS1092_NLP_Project/models/bert/run_05/checkpoints/checkpoint-7980` | 2129.8538 |
| run_06 | `/home/arikanmurat@edu.local/projects/CDS1092_NLP_Project/models/bert/run_06/checkpoints/checkpoint-7980` | 2982.7804 |

### 12.5 Epoch history

Her epoch i?in ilk validation event g?sterilir; son restored-model event 12.3 tablosundad?r. Train loss epoch log de?eridir. Kaynaklar her tablonun ba?l???nda belirtilir.

#### `results/bert/run_01/training_history.csv`

| Epoch | Step | Train loss | Validation loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1596 | 0.5257077324659305 | 0.43079593777656555 | 0.8107638888888888 | 0.838407960199005 | 0.7699195906432749 | 0.8027055349147375 |
| 2.0 | 3192 | 0.39099182281876566 | 0.3751548230648041 | 0.8300438596491229 | 0.864553895841744 | 0.7827119883040936 | 0.8215998465375024 |
| 3.0 | 4788 | 0.3166973525121397 | 0.3837580978870392 | 0.8390899122807017 | 0.8406462272810722 | 0.8368055555555556 | 0.8387214946423666 |
| 4.0 | 6384 | 0.2578537828641428 | 0.3965548276901245 | 0.8463084795321637 | 0.8637236084452975 | 0.8223684210526315 | 0.8425388504025464 |
| 5.0 | 7980 | 0.20904292438860825 | 0.4437739849090576 | 0.8450292397660819 | 0.8535580524344569 | 0.83296783625731 | 0.8431372549019608 |

#### `results/bert/run_02/training_history.csv`

| Epoch | Step | Train loss | Validation loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1596 | 0.5230060854651276 | 0.4434451162815094 | 0.8078399122807017 | 0.8156267566048342 | 0.7955043859649122 | 0.8054399111851235 |
| 2.0 | 3192 | 0.3879495491658835 | 0.380142480134964 | 0.8332419590643275 | 0.8552503409312293 | 0.8022660818713451 | 0.8279113625648279 |
| 3.0 | 4788 | 0.3062116388689008 | 0.37482649087905884 | 0.8422880116959064 | 0.8463387573964497 | 0.8364400584795322 | 0.841360294117647 |
| 4.0 | 6384 | 0.2369722256385593 | 0.3859826326370239 | 0.8490497076023392 | 0.8598342125094197 | 0.8340643274853801 | 0.8467532467532467 |
| 5.0 | 7980 | 0.1790014472521636 | 0.4656360447406769 | 0.8516081871345029 | 0.8601647323099962 | 0.8397295321637427 | 0.8498243018309599 |
| 6.0 | 9576 | 0.13814738758823328 | 0.5620740652084351 | 0.8516081871345029 | 0.8561643835616438 | 0.8452119883040936 | 0.8506529336030899 |
| 7.0 | 11172 | 0.10383264642012746 | 0.629921019077301 | 0.8518823099415205 | 0.8525911005310383 | 0.8508771929824561 | 0.8517332845513583 |

#### `results/bert/run_03/training_history.csv`

| Epoch | Step | Train loss | Validation loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1596 | 0.5392013205621475 | 0.4315396547317505 | 0.8028143274853801 | 0.8231279251170047 | 0.7713815789473685 | 0.7964150943396227 |
| 2.0 | 3192 | 0.39917516110833723 | 0.3732753396034241 | 0.8309576023391813 | 0.8664508296236342 | 0.7825292397660819 | 0.8223545227578261 |
| 3.0 | 4788 | 0.32327415829613093 | 0.4045828580856323 | 0.8292214912280702 | 0.8106570098292809 | 0.8591008771929824 | 0.834176204418419 |
| 4.0 | 6384 | 0.2623334420952283 | 0.38507401943206787 | 0.8462171052631579 | 0.8721272834413671 | 0.8114035087719298 | 0.8406702641295086 |
| 5.0 | 7980 | 0.20910923702077458 | 0.4428161382675171 | 0.8469480994152047 | 0.8462520517964618 | 0.847953216374269 | 0.8471017800091283 |
| 6.0 | 9576 | 0.16914140132435582 | 0.49591901898384094 | 0.8497807017543859 | 0.8573562359970127 | 0.8391812865497076 | 0.8481714074621352 |
| 7.0 | 11172 | 0.1407810201620996 | 0.5452165007591248 | 0.8500548245614035 | 0.8512745277828718 | 0.8483187134502924 | 0.8497940503432494 |

#### `results/bert/run_04/training_history.csv`

| Epoch | Step | Train loss | Validation loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1596 | 0.5229045131750274 | 0.43882375955581665 | 0.8135051169590644 | 0.8462159434914228 | 0.766264619883041 | 0.8042581758895176 |
| 2.0 | 3192 | 0.3910188603221922 | 0.37640559673309326 | 0.829952485380117 | 0.8709677419354839 | 0.774671052631579 | 0.8200019344230584 |
| 3.0 | 4788 | 0.3064057199578536 | 0.38364866375923157 | 0.8401864035087719 | 0.8412465627864345 | 0.8386330409356725 | 0.8399377688295049 |
| 4.0 | 6384 | 0.23366963474971608 | 0.3653043210506439 | 0.8516081871345029 | 0.8843387934478626 | 0.8090277777777778 | 0.8450085894254629 |
| 5.0 | 7980 | 0.16915801772497652 | 0.46096131205558777 | 0.8497807017543859 | 0.858830146231721 | 0.837171052631579 | 0.8478622987229317 |
| 6.0 | 9576 | 0.12470941376267818 | 0.569610595703125 | 0.8496893274853801 | 0.852329221137912 | 0.8459429824561403 | 0.8491240942859763 |
| 7.0 | 11172 | 0.08785945729803023 | 0.6973326206207275 | 0.8524305555555556 | 0.8577258393618995 | 0.8450292397660819 | 0.8513302034428795 |

#### `results/bert/run_05/training_history.csv`

| Epoch | Step | Train loss | Validation loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1596 | 0.5162239935165062 | 0.4268994629383087 | 0.8097587719298246 | 0.8471937730438345 | 0.7558479532163743 | 0.7989182924473633 |
| 2.0 | 3192 | 0.38510713123139884 | 0.3701725900173187 | 0.8350694444444444 | 0.8689877238881063 | 0.7891081871345029 | 0.827123838712767 |
| 3.0 | 4788 | 0.29667996284656956 | 0.3823418915271759 | 0.8376279239766082 | 0.8391775289149991 | 0.835343567251462 | 0.8372561589889184 |
| 4.0 | 6384 | 0.21546532575946703 | 0.4108377993106842 | 0.8460343567251462 | 0.8561218732367877 | 0.8318713450292398 | 0.8438224117156363 |
| 5.0 | 7980 | 0.1432509840580455 | 0.5088651180267334 | 0.846765350877193 | 0.8508042151968941 | 0.8410087719298246 | 0.8458781362007168 |

#### `results/bert/run_06/training_history.csv`

| Epoch | Step | Train loss | Validation loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1596 | 0.5267701113134399 | 0.4188584089279175 | 0.8114948830409356 | 0.8581634797226309 | 0.7463450292397661 | 0.798357931775975 |
| 2.0 | 3192 | 0.39814215256157975 | 0.38544726371765137 | 0.8278508771929824 | 0.8530106257378984 | 0.7922149122807017 | 0.8214894826606026 |
| 3.0 | 4788 | 0.31397298941934915 | 0.3909871578216553 | 0.8384502923976608 | 0.8414454277286135 | 0.8340643274853801 | 0.8377386196769456 |
| 4.0 | 6384 | 0.24151481303356045 | 0.38359466195106506 | 0.8494152046783626 | 0.878913991280222 | 0.8104897660818714 | 0.8433162198136528 |
| 5.0 | 7980 | 0.17342627197877505 | 0.4683046042919159 | 0.8512426900584795 | 0.8525311812179017 | 0.8494152046783626 | 0.8509703405346027 |
| 6.0 | 9576 | 0.12431351702314869 | 0.575013279914856 | 0.8489583333333334 | 0.8500458295142072 | 0.8474049707602339 | 0.8487233458405784 |
| 7.0 | 11172 | 0.08669894799253994 | 0.7083661556243896 | 0.8496893274853801 | 0.8494977168949772 | 0.8499634502923976 | 0.8497305197771079 |

### 12.6 Sabit veri dosyalar?n?n SHA-256 de?erleri

Bu de?erler 17 Kas?m 2026 yerel dosya byte i?eriklerinden hesapland?. Ayn? logical CSV farkl? line ending ile kaydedilirse hash de?i?ebilir. Kar??la?t?rma i?in dosyalar? aynen koruyun.

| Dosya | SHA-256 |
| --- | --- |
| `data/processed/sum/test.csv` | `7fda226cb432d6cda00dce0a6ac7a0e9aebcf4a8f874b703d2e3428c788549c2` |
| `data/processed/sum/train.csv` | `c91dd31945eb857a1528ce69f495f36582689271a5a7f5129a1f2e05f0482ed8` |
| `data/processed/sum/validation.csv` | `5478039d9c39ef5f658b532ec09ffdde41bdd06bddb63a17651f2e7e927339d0` |
| `data/processed/umwp/test.csv` | `c03cfaeff114eba47454dc09dc925117c38e8a321288c3e5e359c6cf28b568c0` |
| `data/raw/umwp/StandardDataset.jsonl` | `13adeb41e16d68cea2fcdde55b0f608b604cde4d3f0a0951fa5b89f1919189ba` |
