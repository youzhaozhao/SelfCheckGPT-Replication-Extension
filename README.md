# Reproducing and Extending SelfCheckGPT
### Cross-Lingual and Multi-View Hallucination Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" alt="Python version">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge" alt="PRs welcome">
  <img src="https://img.shields.io/badge/LLM-Hallucination%20Detection-red?style=for-the-badge" alt="Hallucination Detection">
</p>

<p align="center">
  <b>🔍 Reproducing and Extending SelfCheckGPT for Zero-Resource Black-Box Hallucination Detection</b><br>
  <i>Cross-lingual adaptation, ensemble methods, and multi-view feature fusion</i>
</p>

---

## 📖 Overview

This project reproduces and extends the EMNLP 2023 paper:

**SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs**  
Manakul et al., EMNLP 2023  
🔗 [https://aclanthology.org/2023.emnlp-main.557/](https://aclanthology.org/2023.emnlp-main.557/)

SelfCheckGPT proposes a sampling-based black-box framework for detecting hallucinations in LLM-generated text without requiring external knowledge bases or additional supervision.

In this repository, we:

- Reproduce all five SelfCheckGPT variants (BERTScore, NLI, MQAG, Ngram, Prompt) on the WikiBio benchmark.
- Investigate ensemble fusion of the strongest methods (NLI + Prompt) to examine whether near-ceiling performance can be further improved.
- Conduct cross-lingual experiments on a translated Chinese version of WikiBio, analyzing linguistic challenges such as entity sensitivity and hallucination propagation under Chain-of-Thought prompting.
- Propose **Learning-to-Check (L2C)**, a multi-view fusion framework that integrates black-box consistency signals with white-box uncertainty metrics using Random Forest.

Our empirical findings suggest that:

- External semantic consistency remains the dominant signal for hallucination detection.
- Internal uncertainty alone is insufficient but can provide complementary information under non-linear fusion.
- Cross-lingual transfer exposes structural limitations of symbolic methods such as N-gram overlap.

Overall, this project provides a systematic reproduction, analysis, and extension of SelfCheckGPT under English and Chinese settings.

---

## 🧱 Project Structure

```bash
SelfCheckGPT-Replication-Extension/
│
├── README.md
├── requirements.txt
├── SelfCheckGPT_Final_Report.pdf
├── SelfCheckGPT_Midterm_Presentation.pdf
│
├── configs/
│   └── config.py
│
├── data/
│   ├── raw/
│   │   └── wikibio_gpt3_hallucination.json
│   └── processed/
│       └── wikibio_processed_final.json
│
├── docs/
│   ├── SelfCheckGPT_Final_Report.pdf
│   └── SelfCheckGPT_Midterm_Presentation.pdf
│
├── experiments/
│   ├── run_reproduction.py        # Reproduce original SelfCheckGPT results
│   ├── run_cross_lingual.py       # Cross-lingual evaluation (Chinese WikiBio)
│   ├── run_fusion.py              # Ensemble fusion experiments
│   └── run_l2c.py                 # Learning-to-Check (L2C) framework
│
├── src/
│   ├── data/
│   │   ├── download.py            # Dataset download utilities
│   │   ├── loader.py              # Data loading functions
│   │   └── preprocessing.py       # Data preprocessing pipeline
│   │
│   └── selfcheckgpt/
│       ├── modeling_mqag.py
│       ├── modeling_ngram.py
│       ├── modeling_selfcheck.py
│       ├── modeling_selfcheck_apiprompt.py
│       └── utils.py
│
└── results/
    ├── reproduction/
    │   ├── reproduction_results_bertscore.json
    │   ├── reproduction_results_mqag.json
    │   ├── reproduction_results_ngram.json
    │   ├── reproduction_results_NLI.json
    │   └── reproduction_results_prompt.json
    └── fusion/
        └── ensemble_analysis.png

```

---

## ✨ Key Contributions

### 1. 📊 Full Reproduction of SelfCheckGPT

We replicated all five methods (BERTScore, NLI, MQAG, Ngram, Prompt) on the original WikiBio dataset. Results closely match the original paper, with NLI and MQAG achieving slightly higher scores than those reported in the original paper under our experimental setup.

| Method          | Reproduced (NonFact AUC‑PR) | Original (NonFact AUC‑PR) |
|-----------------|-----------------------------|---------------------------|
| BERTScore       | 0.8120                      | 0.8196                    |
| **NLI**         | **0.9250**                  | **0.9250**                |
| **MQAG**        | **0.8517**                  | 0.8426                    |
| Ngram           | 0.8562                      | 0.8563                    |
| Prompt          | 0.9133                      | 0.9342                    |

> *Full implementation details are provided in [`experiments/run_reproduction.py`](experiments/run_reproduction.py)*
---

### 2. 🔗 Ensemble: NLI + Prompt

By combining the two best‑performing methods (NLI & Prompt) via **grid‑search weighting**, we achieved:

- **NonFact AUC‑PR = 0.9299** – surpassing the top single model (NLI: 0.9250)
- Optimal weight: **80% NLI, 20% Prompt**

This small but significant gain demonstrates that even near‑ceiling performance can be improved through complementary fusion.

---

### 3. 🌏 Cross‑Lingual Extension (Chinese)

We constructed a **Chinese WikiBio dataset** by translating the original data.  
Only Ngram and Prompt are directly transferable (others rely on English‑specific models).

| Language | Method          | NonFact AUC‑PR | Passage‑level Pearson |
|----------|-----------------|----------------|-----------------------|
| Chinese  | Ngram           | 0.6090         | 0.0463                |
| Chinese  | **Prompt**      | **0.8069**     | **0.3522**            |
| English  | Ngram (repr.)   | 0.8562         | 0.6472                |
| English  | Prompt (repr.)  | 0.9133         | 0.7196                |

> Prompt retains robustness; Ngram collapses due to Chinese linguistic characteristics.

We explored **four enhancement paths** (semantic, symbolic, reasoning, translation) – all failed, revealing the *Entity Paradox* (embedding‑based methods blur entity distinctions) and *hallucination propagation* in CoT prompting.

---

### 4. 🧠 Learning‑to‑Check (L2C) – Multi‑View Fusion

When internal LLM states are accessible (grey‑box setting), can cheap white‑box metrics replace expensive sampling?  
We designed **L2C** – a Random Forest model that fuses:

- **Black‑box features**: Baichuan‑Prompt, Baichuan‑Ngram
- **White‑box features**: token‑level entropy, perplexity (PPL)

**Results (5‑fold CV on Chinese data):**

| Model               | Average NonFact AUC‑PR | Gain vs. Baseline |
|---------------------|------------------------|-------------------|
| Baichuan‑Prompt     | 0.8107                 | –                 |
| Logistic Regression | 0.8111                 | +0.22%            |
| **Random Forest**   | **0.8206**             | **+1.22%**        |

> ✅ Random Forest achieves **consistent improvement across all folds**, reviving weak signals (Ngram, PPL) that linear models discard.

**Feature importance** (RF vs. LR):

| Feature         | LR Coefficient | RF Importance |
|-----------------|----------------|---------------|
| Baichuan‑Prompt | 0.8549         | 0.5032        |
| Baichuan‑Ngram  | -0.1691        | 0.1727        |
| PPL             | -0.2080        | 0.1292        |
| Entropy         | 0.0675         | 0.0892        |

> Non‑linear models capture **conditional relevance** – e.g., when Prompt is ambiguous (score ~0.5), Ngram acts as a tie‑breaker.

**Conclusion**: Results indicate that external sampling remains the dominant signal, while multi-view non-linear fusion yields additional improvements.

---

## 🚀 Quick Start

### Requirements

- Python 3.8+
- PyTorch, Transformers, scikit‑learn, pandas, numpy
- (For Prompt) Access to Qwen / Baichuan models (or any chat LLM)

Install dependencies:

```bash
pip install -r requirements.txt
```

### Data

The English evaluation data follows the preprocessing described in the original SelfCheckGPT paper.

We use the official WikiBio hallucination benchmark released by the authors:

🔗 https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination

Please refer to the official repository for dataset details, licensing, and usage terms.

In addition, we construct a translated Chinese version of WikiBio (`chinese_wikibio.json`) for cross-lingual experiments.  
The Chinese data is generated via automatic translation and is intended solely for research and analysis purposes.

### Run Experiments

| Experiment               | Command                                         |
| ------------------------ | ----------------------------------------------- |
| Reproduction (full)      | `python experiments/run_reproduction.py --full` |
| Ensemble (Fusion)        | `python experiments/run_fusion.py`              |
| Cross-lingual Evaluation | `python experiments/run_cross_lingual.py`       |
| Learning-to-Check (L2C)  | `python experiments/run_l2c.py`                 |

---

## 📈 Key Results Summary

| Experiment               | Setting                              | NonFact AUC‑PR | Key Insight                          |
|--------------------------|--------------------------------------|----------------|--------------------------------------|
| Reproduction (English)   | SelfCheck‑NLI                        | 0.9250         | Best single model                    |
| Ensemble                 | NLI+Prompt (grid‑search, α=0.8)      | **0.9299**     | +0.53% over NLI                      |
| Cross‑lingual (Chinese)  | Prompt (Baichuan2)                   | 0.8069         | Robust; Ngram fails                   |
| L2C (Chinese)            | Random Forest (4 features)           | **0.8206** (avg)| +1.22% over baseline; non‑linear fusion essential |

---
## ⚠️ Limitations

- Chinese dataset constructed via automatic translation; may contain noise.
- L2C evaluated only on WikiBio-style biography generation.
- White-box uncertainty analysis limited to entropy and perplexity.
- External sampling cost remains high in practical deployment.

Future work may explore:
- Multilingual pretrained NLI transfer
- Calibration-aware uncertainty modeling
- Retrieval-augmented consistency signals
---

## 📖 Citation

If you find this work useful, please cite both our extension and the original SelfCheckGPT paper.

```bibtex
@misc{yuan2025selfcheckgpt,
  title={Reproducing and Extending SelfCheckGPT: Cross-Lingual Analysis and Learning-to-Check},
  author={Yuan, Zhouyan and Huang, Jiarui},
  year={2025},
  note={Undergraduate research project}
}


@inproceedings{manakul2023selfcheckgpt,
  title={SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models},
  author={Manakul, Potsawee and Liusie, Adian and Gales, Mark J. F.},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={9004--9017},
  year={2023}
}
```

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>For more details, please refer to the <a href="SelfCheckGPT_Final_Report.pdf">full paper</a> in the <code>docs/</code> directory.

</p>
