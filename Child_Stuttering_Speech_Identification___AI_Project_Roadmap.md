# Pediatric Stuttering Speech Identification & AI Project Roadmap

---

## 1. Executive Summary & Context

Historically, digital signal processing (DSP) toolboxes in environments like MATLAB were used to process speech signals, extract acoustic features (such as MFCCs, fundamental frequency $F_0$, and formants), and apply basic statistical classifiers to analyze speech disfluencies. 

Modern deep learning (DL) and AI architectures—specifically self-supervised audio transformers like **Wav2Vec 2.0**, **HuBERT**, and **Whisper**—revolutionize this process. By processing raw waveforms end-to-end, deep learning systems can automatically detect, classify, and score pediatric speech disfluencies with significantly higher accuracy, context awareness, and resilience to real-world acoustic noise.

---

## 2. Available Children & Disfluent Speech Datasets

To train and fine-tune deep learning models for stuttering speech identification, several public and research datasets are available:

| Dataset Name | Primary Content & Focus | Access Requirements |
| :--- | :--- | :--- |
| **UCLASS** (Univ. College London Archive of Stuttered Speech) | Audio recordings and transcripts of developmental stuttering in children across various age groups. | Academic / Research Request |
| **FluencyBank** (Carnegie Mellon / TalkBank) | Clinical interviews, conversational speech, and reading passages from pediatric and adult fluency research. | Free Research Registration ([FluencyBank Website](https://fluency.talkbank.org/)) |
| **SEP-28k** (Apple Research) | 28,000 podcast speech clips with fine-grained disfluency labels (blocks, prolongations, repetitions, interjections); includes annotations for FluencyBank. | Open Access (GitHub) |
| **MyST Child Speech Corpus** / **CU Kids' Speech** | Thousands of hours of pediatric conversational speech; ideal for pre-training models to handle child vocal acoustics. | Research Access |

---

## 3. Automation Tools Capability: Cursor vs. Devin

When utilizing modern AI coding agents like **Cursor** or **Devin** to build this deep learning pipeline, their capabilities and workflows differ:

```
                          ┌──────────────────────────┐
                          │   Dataset Retrieval      │
                          └────────────┬─────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       [ Open Datasets ]                             [ Clinical / Gated ]
  (SEP-28k, Hugging Face)                         (UCLASS, FluencyBank)
                │                                             │
      Devin / Cursor Script                         User Fills Sign-Up / IRB
    Downloads Automatically                        Downloads Files to Local Folder
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │ Deep Learning Pipeline   │
                          │   (PyTorch / HuggingFace)│
                          └──────────────────────────┘
```

* **Devin:** Operates within an autonomous cloud environment. It can browse open sources, clone repositories, download public datasets (`wget`, `curl`, Hugging Face CLI), write PyTorch code, run training scripts, and debug errors independently. However, it **cannot** bypass clinical IRB or sign research agreements required for gated speech databases.
* **Cursor:** Operates within your local IDE. It writes model architectures, data loaders, and terminal scripts directly for you to execute in one click. You manage local data downloads and authentication manually.

---

## 4. What You Will Achieve (Project Deliverables)

By building this deep learning architecture, you will create a clinical decision support tool capable of:

1. **Frame-Level Disfluency Detection:**
   * **Blocks:** Identification of silent pitch locks and acoustic air-flow stoppages.
   * **Prolongations:** Detection of sustained phonemes (e.g., *"Sssss-un"*).
   * **Repetitions:** Distinction between sound-level (*"b-b-ball"*) and whole-word repetitions (*"my-my-my"*).
   * **Interjections:** Detection of filler speech tokens (*"um"*, *"uh"*, *"like"*).

2. **Automated Stuttering Severity Rating:**
   * Calculation of **Percentage of Stuttered Syllables (%SS)**.
   * Measurement of continuous disfluency duration in milliseconds.
   * Automated mapping to continuous clinical scores equivalent to the **SSI-4 (Stuttering Severity Instrument)**.

3. **Inference API & Interface:**
   * An end-to-end Python pipeline where an unlabelled 10-second pediatric audio file (`.wav`) returns structured JSON timestamps, disfluency counts, and an overall fluency rating.

---

## 5. Limitations of Classic MATLAB Overcome by Deep Learning

| Domain | Traditional MATLAB / DSP Approach | Modern Deep Learning Approach |
| :--- | :--- | :--- |
| **Feature Extraction** | Relied on hand-crafted math features (MFCCs, formants, energy profiles) that miss subtle silent blocks. | **Wav2Vec 2.0 / HuBERT** learn high-level continuous representations directly from raw waveform samples. |
| **Pediatric Acoustics** | Brittle when encountering children's higher fundamental pitch ($F_0$), short vocal tracts, and speech speed variance. | Deep Transformer encoders generalize seamlessly across varied pediatric pitch ranges and speaking speeds. |
| **Context Awareness** | Used static energy thresholds, making it difficult to differentiate intentional speech pauses from silent stuttering blocks. | Multi-head self-attention mechanisms evaluate temporal context across seconds of speech to distinguish intentional pauses from blocks. |
| **Noise Robustness** | Highly sensitive to ambient room noise, requiring clean lab or studio recordings. | Robust to real-world background noise due to dynamic data augmentation (noise injection, speed perturbation). |

---

## 6. Recommended Starter Prompt for Code Generation

Use the following structured prompt when starting your build in Devin or Cursor:

```text
Build a PyTorch deep learning project for automated stuttering speech identification and severity scoring in children.

1. Set up a project structure with a virtual environment, PyTorch, Hugging Face Transformers, Torchaudio, and Librosa.
2. Write a script to fetch publicly available speech disfluency datasets (e.g., loading SEP-28k or public Hugging Face speech audio clips).
3. Create a placeholder dataset loader that can easily be mapped to local audio files from UCLASS or FluencyBank once downloaded manually.
4. Implement a fine-tuning pipeline using pre-trained Wav2Vec 2.0 (or HuBERT) as the audio backbone, attached to a multi-task classification head for identifying stuttering types (blocks, prolongations, repetitions) and regression for severity score.
5. Provide a training loop with evaluation metrics (F1-score, MAE) and a simple inference script to test raw .wav inputs.
```