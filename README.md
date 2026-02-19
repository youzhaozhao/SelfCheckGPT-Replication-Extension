# SelfCheckGPT Reproduction and Extension

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
🔗 [https://arxiv.org/abs/2303.08896](https://aclanthology.org/2023.emnlp-main.557/)

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
├── reproduction_experiment.py          # Run reproduction of all 5 SelfCheckGPT variants
├── final_fusion_experiment.py           # Ensemble (NLI+Prompt) with grid search
├── cross_lingual_experiment.py          # Chinese evaluation (Ngram & Prompt)
├── comprehensive_analysis.py             # L2C framework with Random Forest
├── verify_internal_analysis.py           # Correlation analysis of internal metrics
│
├── data/
│   ├── reproduction_results.json         # Reproduced scores (English)
│   └── chinese_wikibio.json               # Chinese translated WikiBio dataset
│
├── results/
│   └── ensemble_analysis.png              # Weight analysis plot
│
├── paper/
│   └── final_paper.pdf                     # Full project paper
│
└── README.md
```

---

## ✨ Key Contributions

### 1. 📊 Full Reproduction of SelfCheckGPT

We replicated all five methods (BERTScore, NLI, MQAG, Ngram, Prompt) on the original WikiBio dataset. Results closely match the original paper, with **NLI and MQAG even outperforming** the reported numbers.

| Method          | Reproduced (NonFact AUC‑PR) | Original (NonFact AUC‑PR) |
|-----------------|-----------------------------|---------------------------|
| BERTScore       | 0.8120                      | 0.8196                    |
| **NLI**         | **0.9250**                  | **0.9250**                |
| **MQAG**        | **0.8517**                  | 0.8426                    |
| Ngram           | 0.8562                      | 0.8563                    |
| Prompt          | 0.9133                      | 0.9342                    |

> *Full details in [`reproduction_experiment.py`](reproduction_experiment.py)*

---

### 2. 🔗 Ensemble: NLI + Prompt

By combining the two best‑performing methods (NLI & Prompt) via **grid‑search weighting**, we achieved:

- **NonFact AUC‑PR = 0.9299** – surpassing the top single model (NLI: 0.9250)
- Optimal weight: **80% NLI, 20% Prompt**

This small but significant gain demonstrates that even near‑ceiling performance can be improved through complementary fusion.

![Ensemble Weight Analysis](results/ensemble_analysis.png)

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

**Conclusion**: External sampling is irreplaceable, but multi‑view non‑linear fusion delivers the best performance.

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

The original WikiBio dataset is included in `data/` (courtesy of SelfCheckGPT authors).  
Chinese translation is provided as `chinese_wikibio.json`.

### Run Experiments

| Experiment               | Command                                      |
|--------------------------|----------------------------------------------|
| Reproduction (full)      | `python reproduction_experiment.py --full`   |
| Ensemble (NLI+Prompt)    | `python final_fusion_experiment.py`          |
| Cross‑lingual            | `python cross_lingual_experiment.py`         |
| L2C Random Forest        | `python comprehensive_analysis.py`           |
| Internal metrics analysis| `python verify_internal_analysis.py`         |

---

## 📈 Key Results Summary

| Experiment               | Setting                              | NonFact AUC‑PR | Key Insight                          |
|--------------------------|--------------------------------------|----------------|--------------------------------------|
| Reproduction (English)   | SelfCheck‑NLI                        | 0.9250         | Best single model                    |
| Ensemble                 | NLI+Prompt (grid‑search, α=0.8)      | **0.9299**     | +0.53% over NLI                      |
| Cross‑lingual (Chinese)  | Prompt (Baichuan2)                   | 0.8069         | Robust; Ngram fails                   |
| L2C (Chinese)            | Random Forest (4 features)           | **0.8206** (avg)| +1.22% over baseline; non‑linear fusion essential |

---

## 📖 Citation

If you find this work useful, please cite both our extension and the original SelfCheckGPT paper.

```bibtex
@article{yuan2025selfcheckgpt,
  title={SelfCheckGPT Reproduction and Extension: Cross-Lingual Empirical Research and Multi-Dimensional Feature Fusion for LLM Hallucination Detection},
  author={Yuan, Zhouyan and Huang, Jiarui},
  year={2025}
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
  <i>For more details, please refer to the <a href="paper/final_paper.pdf">full paper</a> in the <code>paper/</code> directory.</i>
</p>
