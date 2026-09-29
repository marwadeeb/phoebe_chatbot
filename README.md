# 😰 Phoebe: the chatbot who is afraid of *everything* (but helps you anyway)

**EECE 503P/798S: Agentic Systems · Assignment C2: Build a Persona-Driven Chat App with an Open-Source LLM**
**Author:** Deeb · **Deliverable:** [`Deeb_LLM_Chatbot.ipynb`](Deeb_LLM_Chatbot.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marwadeeb/phoebe_chatbot/blob/main/Deeb_LLM_Chatbot.ipynb)

Phoebe is a web chat app powered by the open-source **Qwen2.5-7B-Instruct** model. Her twist: she is
scared of whatever you talk about (spiders, soup, numbers, Tuesdays...), reacts with a short burst of
cartoonish panic, and then **pushes through her fear to give you a correct, genuinely useful answer**.
A live *fear meter* shows how scared she is, and an optional **Reasoning Mode** shows her nervous
step-by-step "overthinking" in a separate panel before her final answer.

> 📸 *Screenshot: add `docs/screenshot.png` after the final Colab run.*

---

## Contents
1. [How the assignment requirements are covered](#1-how-the-assignment-requirements-are-covered)
2. [Setup and run: Google Colab (recommended)](#2-setup-and-run-google-colab-recommended)
3. [Setup and run: on your own computer (optional)](#3-setup-and-run-on-your-own-computer-optional)
4. [How to use the app](#4-how-to-use-the-app)
5. [LLM choice and model details](#5-llm-choice-and-model-details)
6. [The persona twist and how it is implemented](#6-the-persona-twist-and-how-it-is-implemented)
7. [Prompting techniques used, and why](#7-prompting-techniques-used-and-why)
8. [How the system and user prompts are crafted](#8-how-the-system-and-user-prompts-are-crafted)
9. [Context handling: window size and what happens near the limit](#9-context-handling-window-size-and-what-happens-near-the-limit)
10. [Bonus: Reasoning Mode toggle](#10-bonus-reasoning-mode-toggle)
11. [Usage examples](#11-usage-examples)
12. [Project structure](#12-project-structure)
13. [Troubleshooting](#13-troubleshooting)
14. [Limitations](#14-limitations)

---

## 1. How the assignment requirements are covered

| Requirement (from the assignment) | Where / how | README section |
|---|---|---|
| Select and deploy an open-source transformer model | Qwen2.5-7B-Instruct (Apache-2.0), loaded with Hugging Face `transformers` | §5 |
| Host it locally or via an API | Hosted **locally on the GPU** of the Colab runtime (or your own PC); no external API | §2, §3, §5 |
| Minimal web UI to send messages and view replies | Gradio app (notebook §11–§12) | §4 |
| Clean, mobile-friendly layout, scrollable history, clear user/bot separation | Responsive two-column layout that stacks on phones; scrollable chat; green bubbles on the right for the user, purple bubbles with Phoebe's avatar on the left for the bot | §4 |
| Distinctive persona twist, implemented through a system prompt | Phoebe, afraid of everything; `PERSONA_SYSTEM_PROMPT` | §6 |
| At least two prompting techniques, stated with reasons | **Few-shot**, **prompt chaining**, **chain-of-thought** (and zero-shot for the summarizer) | §7 |
| Maintain chat history; truncate or summarize older turns | Token-budgeted memory: LLM **summarization**, then **dropping**, then **truncation** | §9 |
| State the context window size and what happens near the limit | **32,768 tokens**; step-by-step behavior explained, with a live memory gauge in the UI | §9 |
| Show how system and user prompts enforce the persona and trigger the techniques | Full prompt anatomy, plus a *Prompt inspector* in both the notebook (§8) and the UI | §8 |
| README with setup, model details, usage examples | This file | §2, §3, §5, §11 |
| **Bonus:** Reasoning Mode toggle, CoT, reasoning shown separately, ≥2 before/after demos | Toggle in the UI; collapsible "📓 panic journal" panel; 4 demo prompts in notebook §10 | §10 |

---

## 2. Setup and run: Google Colab (recommended)

You need only a Google account; nothing is installed on your computer.

1. **Open the notebook in Colab.** Click the **Open in Colab** badge at the top of this page.
   *Alternative:* go to <https://colab.research.google.com>, choose **File → Upload notebook**, and select `Deeb_LLM_Chatbot.ipynb`.
2. **Turn on the GPU.** In the menu choose **Runtime → Change runtime type**, select **T4 GPU**, and click **Save**.
   (The notebook asks for a GPU automatically, but please check. Without a GPU the notebook stops with a clear message.)
3. **Run everything.** Choose **Runtime → Run all**. If Colab warns that the notebook was not authored by Google, click **Run anyway**.
4. **Wait for the model to download.** The first run downloads about 15 GB of model weights, which takes roughly 3–6 minutes. You can follow progress under each cell.
   A warning that `HF_TOKEN` is not set is harmless: the model is public and needs no account or token.
5. **Open the app.** The very last cell prints a line like `Running on public URL: https://xxxxxxxx.gradio.live`. Click it. The link works on your phone too and stays valid for 72 hours while the cell is running.
6. **Stop the app** with the ■ (stop) button next to the last cell, or with **Runtime → Disconnect and delete runtime**.

> Everything between loading and launching (the sanity check, the prompt inspector, the context-handling demo and the Reasoning Mode demos) runs automatically during **Run all**, so the saved outputs double as documentation.

---

## 3. Setup and run: on your own computer (optional)

You need an **NVIDIA GPU with at least 8 GB of memory** and an up-to-date NVIDIA driver. The app was
developed on Windows 11 with an RTX 5070 Laptop GPU (8 GB).

1. **Install Python 3.12** from <https://www.python.org/downloads/>. On Windows, tick **"Add python.exe to PATH"** in the installer.
2. **Download this project.** Either click **Code → Download ZIP** on GitHub and unzip it, or run:
   ```bash
   git clone https://github.com/marwadeeb/phoebe_chatbot.git
   ```
3. **Open a terminal in the project folder** and create a virtual environment:
   ```bash
   python -m venv .venv
   ```
   Then activate it. On Windows (PowerShell): `.venv\Scripts\Activate.ps1`. On macOS/Linux: `source .venv/bin/activate`.
4. **Install PyTorch with CUDA.** RTX 50-series (Blackwell) GPUs need a CUDA 12.8+ build:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```
   (For older GPUs, <https://pytorch.org/get-started/locally/> gives you the right command.)
5. **Install the other dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
6. **Start Jupyter and run the notebook:**
   ```bash
   jupyter notebook Deeb_LLM_Chatbot.ipynb
   ```
   Choose **Run → Run All Cells**. When the last cell prints `Running on local URL: http://127.0.0.1:7860`, open that address in your browser.

---

## 4. How to use the app

![layout](https://img.shields.io/badge/layout-chat%20%2B%20side%20panel-a78bfa)

| Area | What it does |
|---|---|
| **Header** | A trembling Phoebe, and a tagline that cycles through her fears ("afraid of *spiders*... *soup*... *numbers*..."). |
| **Chat** (left) | Type in the box and press **Enter** or **Send 🫣**. Your messages appear in green on the right; Phoebe's appear in purple on the left, next to her avatar. Replies stream in word by word. The history scrolls, and each message has a copy button. |
| **Collapsible panels in the chat** | **😱 Fear scan (chain step 1)** shows the JSON produced by the first model call. **📓 Phoebe's panic journal** shows the Reasoning Mode trace. **🧠 Context manager** appears when old messages are summarized or dropped. |
| **📓 Reasoning Mode** (right) | Toggle it on to make Phoebe think step by step before answering (§10). |
| **Fear meter** (right) | Phobia name, trigger, and a 0–10 meter that updates on every message. It shakes at 7/10 or higher. |
| **🧠 Phoebe's memory** (right) | Live gauge of history tokens against the budget (the pink tick marks the 75% summarization threshold), the size of the last prompt against the 32,768-token window, counters for summaries and dropped messages, and the current memory note. |
| **🧪 Context lab** | A slider for the history budget. Set it to 512 to see summarization kick in after a few messages. |
| **🔬 Prompt inspector** | The exact prompt (system + few-shot + history + user) sent to the model for the last reply. |
| **🧹 Start over** | Clears the chat and Phoebe's memory. |
| **Examples** | One-click example prompts, including logic puzzles for trying Reasoning Mode. |

On a phone, the side panel moves below the chat, and all animations are turned off for users whose
device is set to *reduce motion*.

---

## 5. LLM choice and model details

| | |
|---|---|
| **Model** | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| **Type** | Decoder-only transformer, instruction-tuned chat model (7.6B parameters) |
| **License** | Apache-2.0 (open source; no sign-up or access request needed) |
| **Context window** | **32,768 tokens** (prompt + generated reply) |
| **How it is hosted** | Locally on the runtime's GPU, via Hugging Face `transformers` |
| **Quantization** | 4-bit NF4 with double quantization (`bitsandbytes`): ~15 GB of weights become **~5.5 GB of GPU memory** |
| **Precision** | float16 compute on a T4; bfloat16 on newer GPUs (chosen automatically) |
| **Decoding** | Chat: temperature 0.7, top-p 0.9. Reasoning Mode: temperature 0.3. Internal steps and demos: greedy (deterministic) |

**Why this model?**
- **It fits a free GPU.** In 4-bit it runs comfortably on Colab's free T4 (16 GB) and on an 8 GB laptop GPU.
- **It follows instructions well.** The app depends on the model reliably emitting JSON (fear scan) and `<worry>/<answer>` tags (Reasoning Mode). Qwen2.5 is among the strongest open 7B models at structured output.
- **It needs no gatekeeping.** Unlike Llama 3, it needs no Hugging Face token or license approval, so anyone can run the notebook immediately.
- **It is not a built-in "thinking" model.** Its step-by-step reasoning comes *only* from our chain-of-thought prompt, so the Reasoning Mode comparison really measures the effect of prompting.

---

## 6. The persona twist and how it is implemented

**Phoebe** (a pun on *phobia*) is sweet, knowledgeable, and afraid of almost everything. Every reply has three parts:

1. **Panic opener** (one sentence): she reacts to the scary thing and names the phobia, and her drama scales with the fear level.
2. **The real answer**: accurate, clear, complete.
3. **Nervous sign-off** (one line), e.g. *"Please lower them in with a spoon. For me. 🫣"*

It is implemented in four layers:

| Layer | Implementation |
|---|---|
| **System prompt** | `PERSONA_SYSTEM_PROMPT` defines the character, the three-part reply structure, and hard rules: *correctness beats comedy*, never refuse because of fear, at most two sentences of drama, never mock real anxiety disorders or the user. |
| **Few-shot examples** | Three example exchanges (boiling eggs, capital of Australia, Python tips) demonstrate the voice and structure. |
| **Fear scan (prompt chain)** | Before each answer, a separate model call decides *what* scares Phoebe and *how much*. The answer prompt receives that JSON, so her panic is specific and proportional ("arachnophobia, 9/10" vs "ovophobia, 6/10"), and the UI's fear meter shows it. |
| **Gentle mode (responsible design)** | The fear scan also flags whether the *user* seems genuinely upset. If so, `GENTLE_MODE_INSTRUCTIONS` switch the comedy off for that reply: Phoebe answers warmly and, if there is any sign of danger, points to trusted people or crisis services. The joke never lands on someone who is struggling. |

The UI carries the persona too: the trembling avatar, the cycling list of fears, the fear meter, the
"panic journal" reasoning panel, and placeholder text such as *"Phoebe is hiding under a blanket. Say something... gently."*

---

## 7. Prompting techniques used, and why

| # | Technique | Where in the code | Why we used it | Where you can *see* it |
|---|---|---|---|---|
| 1 | **Few-shot prompting** | `FEW_SHOT_CHAT`, `FEW_SHOT_REASONING`, `FEW_SHOT_SCAN` | A persona described only in words drifts back to a generic assistant voice after a few turns. Showing 2–3 concrete examples locks in the tone, the three-part structure, the phobia-naming habit, and (for the scan) the exact JSON shape. It made the persona noticeably more consistent. | Prompt inspector (UI and notebook §8) |
| 2 | **Prompt chaining** | `fear_scan()` → `build_messages()` in `phoebe_reply_stream()` | Splitting "decide what is scary" from "write the answer" gives each call one simple job. Step 1 is deterministic and outputs structured data (trigger, phobia, fear level, distress flag). Step 2 uses it to scale Phoebe's panic, switch on gentle mode, and drive the UI's fear meter. A single prompt was less consistent about naming a phobia and had no machine-readable output for the UI. The context summarizer is a third chained call. | "😱 Fear scan (chain step 1)" panel under every message, and the fear meter |
| 3 | **Chain-of-thought** (bonus Reasoning Mode) | `REASONING_MODE_INSTRUCTIONS`, `split_reasoning()` | Multi-step problems and trick questions go wrong when the model answers immediately. Asking for numbered steps *before* the answer, including an explicit "double-check / look for tricks" step, improves correctness. Framed as Phoebe's anxious overthinking, it also fits the persona. | "📓 Phoebe's panic journal" panel, and demos in notebook §10 |
| 4 | **Zero-shot instruction** | `SUMMARIZER_PROMPT` | Summarizing old turns is a standard task the model does well without examples, so we kept that prompt short to save tokens. | Memory note in the "🧠 Phoebe's memory" panel |

---

## 8. How the system and user prompts are crafted

**Chain step 1: fear scan** (deterministic, max 80 new tokens)
```
[SYSTEM]    FEAR_SCAN_PROMPT: "reply with ONLY one JSON object with keys trigger, phobia, fear_level, user_distressed"
[USER]      "Latest user message: Can you explain how spiders make their webs?"           ┐
[ASSISTANT] {"trigger": "spiders", "phobia": "arachnophobia", "fear_level": 9, ...}       ├ few-shot (3 pairs)
...                                                                                        ┘
[USER]      "(Phoebe's previous reply, for context: ...)  Latest user message: <your message>"
```
The output is parsed with a regex plus `json.loads`, fields are validated and clamped (fear level 1–10), and a safe default (`phobophobia`, the fear of fear) is used if the JSON is malformed.

**Chain step 2: answer**
```
[SYSTEM]  PERSONA_SYSTEM_PROMPT                        ← who Phoebe is, reply structure, hard rules
          + MODE: QUICK ANSWER  |  MODE: REASONING      ← toggled by the UI (triggers chain-of-thought)
          + MEMORY NOTE: <summary of old turns>         ← only if older turns were summarized (§9)
          + FEAR SCAN: {"trigger": ..., "fear_level": 8, ...}  ← output of chain step 1
          + GENTLE MODE instructions                    ← only if user_distressed is true
[USER] / [ASSISTANT]  few-shot examples                 ← chat examples, or <worry>/<answer> examples in Reasoning Mode
[USER] / [ASSISTANT]  recent conversation history       ← Phoebe's final answers only
[USER]    <your message>                                ← truncated to 1,024 tokens if huge
```

Design notes:
- All dynamic context (mode, memory, fear scan) goes into the **single system message**, so the model treats it as instructions rather than something the user said.
- The **user's message is passed through unchanged**, so the persona is enforced entirely by the system prompt and examples, and users cannot accidentally break the format.
- The few-shot set is **swapped with the mode**: in Reasoning Mode the examples show the `<worry>…</worry><answer>…</answer>` format, which is what makes the model follow it reliably.
- Everything is rendered with the model's own chat template (`tokenizer.apply_chat_template`), which also gives exact token counts.

The notebook's §8 prints a full example prompt, and the **🔬 Prompt inspector** in the UI shows the real prompt for every reply.

---

## 9. Context handling: window size and what happens near the limit

**Context window of the model: 32,768 tokens.** The prompt and the reply share it. The notebook prints
this value from the model's own config (`max_position_embeddings`).

The app does **not** fill the whole window. On a free GPU, long prompts use more memory and slow down
*every* reply, so chat history (memory note + past messages) gets a **history budget of 3,072 tokens**
(adjustable from 256 to 8,192 in the UI's *Context lab*). The persona prompt and few-shot examples
(~1,000–1,300 tokens) come on top, so a typical prompt stays under ~5,000 tokens.

Before every reply, `manage_context()` applies these rules in order:

| History size | What happens |
|---|---|
| **Below 75% of the budget** | All past messages are sent word-for-word. |
| **Above 75%** | **Summarization.** Everything except the last 2 exchanges is folded by the model (a chained, zero-shot call) into a ≤8-bullet **memory note**: the user's name, preferences, topics, important answers. The note goes into the system prompt. Phoebe still "remembers" early details at a fraction of the tokens. A "🧠 Context manager" panel appears in the chat, and the counters update. |
| **Still above 100%** (e.g. very long recent messages) | **Dropping.** The oldest messages are removed one at a time until history fits. The memory note is capped at half the budget. |
| **A single message over 1,024 tokens** | **Truncation.** The message is cut to 1,024 tokens, with a visible note. |
| **Hard safety net** | The reply length is capped at `32,768 − prompt tokens`, so prompt + reply can **never** exceed the window. |

Other details:
- Only Phoebe's **final answers** are stored. Reasoning traces and fear-scan JSON are shown in the UI but never re-sent, which keeps Reasoning Mode from eating the budget.
- Each browser session has its own memory (`gr.State`), so different users never see each other's history.
- **Demonstration:** notebook §9 runs a 5-turn conversation with a 400-token budget. The output shows summarization triggering, the token counts falling, and Phoebe still recalling the user's name and cat from turn 1 through the memory note.

---

## 10. Bonus: Reasoning Mode toggle

Turn on **📓 Reasoning Mode** in the side panel, and three things change:

1. **Step-by-step reasoning is elicited.** The system prompt switches from *QUICK ANSWER* to *REASONING MODE*, and the few-shot examples switch to ones that show numbered steps inside `<worry>…</worry>`, ending with an explicit "double-check / look for trick questions" step, followed by the reply inside `<answer>…</answer>`. Temperature drops to 0.3 for steadier reasoning.
2. **Reasoning is separated from the answer.** `split_reasoning()` parses the tags *while the reply streams*. The reasoning goes into a collapsible **"📓 Phoebe's panic journal (step-by-step reasoning)"** panel, which shows a spinner while she is still thinking. The final answer appears below it as a normal chat bubble, the same way reasoning models display their thoughts. If the model ever skips the tags, the whole reply is shown as the answer, so nothing is lost.
3. **Only the final answer is kept in memory**, not the reasoning (see §9).

### Demonstration: Reasoning Mode off vs on

Notebook §10 runs four classic multi-step and trick puzzles in both modes, with greedy decoding so the results are reproducible:

| # | Prompt (short) | Correct | Reasoning **OFF** | Reasoning **ON** |
|---|---|---|---|---|
| 1 | Sally has 3 brothers; each brother has 2 sisters. How many sisters does Sally have? | 1 | *to be filled from the run* | *to be filled from the run* |
| 2 | Pencils 3 for $0.75; buys 2 dozen with $10; half the change on $0.50 erasers. How many erasers? | 4 | *…* | *…* |
| 3 | 5 machines make 5 widgets in 5 min. 100 machines, 100 widgets? | 5 min | *…* | *…* |
| 4 | Lily pads double daily and cover the lake on day 48. When is it half covered? | 47 | *…* | *…* |

<!-- TODO: fill this table (and add 2 short quoted examples) from the saved outputs of notebook §10. -->

---

## 11. Usage examples

**Everyday question** (Reasoning Mode off)
> **You:** How do airplanes stay in the air?
> **Phoebe:** *(fear scan → aerophobia · 8/10)* A-airplanes?! Giant metal birds held up by *invisible air*?! 😨 …Okay. Planes fly because their wings are shaped and angled so that air flowing over them is pushed downward, which creates an upward force called **lift**… *(clear explanation of lift, thrust, drag and weight)* … Please keep your seatbelt fastened. Both of us will feel better.

**Logic puzzle** (Reasoning Mode on)
> **You:** If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?
> **📓 Panic journal:** Step 1: Machines?! … each machine makes 1 widget in 5 minutes. Step 2: … 100 machines each make 1 widget in parallel … Step 3: double-check: it is NOT 100 minutes, that's the trick…
> **Phoebe:** Mechanophobia at 7/10! 😰 It takes **5 minutes**: each machine makes one widget in 5 minutes, and all 100 work at the same time. …

**Memory across turns**
> **You:** Hi Phoebe! My name is Sam and I have a cat called Pickles.
> *… several messages later, after older turns were summarized …*
> **You:** Do you remember my name and my cat's name?
> **Phoebe:** C-cats have *claws*… 😟 Of course: you're **Sam**, and your cat is **Pickles**!

*(The replies above are illustrative; see the saved notebook outputs for real transcripts.)*

---

## 12. Project structure

```
phoebe_chatbot/
├── Deeb_LLM_Chatbot.ipynb        ← THE DELIVERABLE: the whole app in one Colab notebook
├── README.md                     ← this file
├── requirements.txt              ← dependencies for running locally
├── notebook_src/
│   └── Deeb_LLM_Chatbot.py       ← source of the notebook (plain Python, "# %%" cell markers)
└── tools/
    └── build_notebook.py         ← rebuilds the .ipynb from the source file
```

The notebook is developed as a plain Python file, which gives clean diffs and easy editing, and then
converted with `python tools/build_notebook.py`. The final committed notebook is re-run so that it
contains its outputs.

Notebook sections: §1 install · §2 config · §3 model loading and generation helpers · §4 prompts ·
§5 context manager · §6 prompt chain · §7 sanity check · §8 prompt inspector · §9 context demo ·
§10 Reasoning Mode demos · §11 UI · §12 launch.

---

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| `AssertionError: No GPU found` | **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all** again. |
| Colab says no GPU is available | Free GPU quota is temporarily used up. Try again later, or use a different Google account. |
| `CUDA out of memory` | **Runtime → Restart session**, then run all again. Locally, close other GPU-heavy programs. |
| No `gradio.live` link appears | Scroll to the bottom of the last cell's output; it can take ~10 s. If share links are blocked on your network, run locally and use `http://127.0.0.1:7860`. |
| Replies are slow | Normal on a T4: ~1–2 s for the fear scan and roughly 15–20 tokens/s for the answer. Reasoning Mode writes more text, so it takes longer. |
| Locally: `torch.cuda.is_available()` is `False` | You installed the CPU-only PyTorch. Reinstall with the CUDA command from §3, step 4. |

---

## 14. Limitations

- A 7B model can still make mistakes, including in Reasoning Mode. Chain-of-thought improves multi-step accuracy but does not guarantee it.
- Invented phobia names (e.g. "australophobia") are part of the joke and are not real medical terms.
- One GPU serves one reply at a time (a Gradio queue with concurrency 1), so simultaneous users wait their turn.
- The memory note is a lossy summary: fine details from very old turns may be lost once they are summarized.
