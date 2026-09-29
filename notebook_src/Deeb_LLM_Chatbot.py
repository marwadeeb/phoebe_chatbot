# Source of Deeb_LLM_Chatbot.ipynb in "percent" format (one "# %%" marker per cell).
# Edit this file, then rebuild the notebook with:  python tools/build_notebook.py

# %% [markdown]
# # 😰 Phoebe: the chatbot who is afraid of *everything* (but helps you anyway)
#
# **EECE 503P/798S: Agentic Systems · Assignment C2**
# **Author:** Marwa Deeb · **ID:** 202674575
#
# Phoebe is a persona-driven chat app built on the open-source **Qwen2.5-7B-Instruct** model.
# She is terrified of whatever you ask about (spiders, soup, numbers, Tuesdays...), yet she always
# pushes through her fear to give you a correct, useful answer.
#
# ### ▶️ How to run this notebook (Google Colab)
# 1. **Runtime → Change runtime type → T4 GPU → Save** (this notebook requests a GPU automatically, but double-check).
# 2. **Runtime → Run all**.
# 3. The first run downloads the model (~15 GB, about 3–6 minutes). Wait for the last cell to print a
#    `https://….gradio.live` link, then click it to chat with Phoebe (it also works on your phone).
#
# ### 🗺️ Where each assignment requirement lives
# | Requirement | Section |
# |---|---|
# | Open-source LLM integration | §1–§3 (install, config, model loading) |
# | Persona twist via system prompt | §4 (prompts) |
# | Prompting techniques (few-shot, prompt chaining, chain-of-thought) | §4 and §6 |
# | Chat history and context-window handling | §5, plus the demo in §9 |
# | Showing how the prompts are crafted | §8 (prompt inspector) |
# | Bonus: Reasoning Mode, before/after demos | §6 and §10 |
# | Web chat UI (Gradio, mobile friendly) | §11–§12 |

# %% [markdown]
# ## §1 · Install dependencies
# Colab already has PyTorch with CUDA. We add `transformers`, `accelerate`, `bitsandbytes` (4-bit
# quantization) and `gradio` (the web UI). When running locally, install `requirements.txt` instead
# (see the README).

# %%
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers>=4.46,<5", "accelerate>=1.0", "bitsandbytes>=0.45", "gradio>=5.20,<6",
    ])
print("Running in Colab" if IN_COLAB else "Running locally")

# %% [markdown]
# ## §2 · Imports and configuration
# All tunable numbers live here, including the **context window** and the **history budget** used
# by the context manager (§5).

# %%
import html
import json
import re
import threading
import time
from dataclasses import dataclass, field

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Qwen2.5-7B-Instruct natively supports 32,768 tokens (prompt + generated reply together).
MODEL_CONTEXT_WINDOW = 32_768

# We deliberately use far less than the full window for chat history: on a free T4 (16 GB) or an
# 8 GB laptop GPU, long prompts cost memory and make every reply slower. See §5 for what happens
# when the history approaches this budget.
HISTORY_TOKEN_BUDGET = 3_072     # tokens reserved for memory note + past turns
SUMMARIZE_AT = 0.75              # start summarizing once history uses 75% of the budget
KEEP_RECENT_MESSAGES = 4         # the last 2 exchanges are always kept word-for-word
MAX_USER_MESSAGE_TOKENS = 1_024  # a single huge message is truncated to this

MAX_NEW_TOKENS_CHAT = 512
MAX_NEW_TOKENS_REASONING = 1_024
MAX_NEW_TOKENS_SCAN = 80
MAX_NEW_TOKENS_SUMMARY = 220

# %% [markdown]
# ## §3 · Load the model (4-bit)
# The model is loaded with 4-bit NF4 quantization via `bitsandbytes`, which shrinks the ~15 GB of
# weights to about 5.5 GB of GPU memory so it fits on a free Colab T4.

# %%
assert torch.cuda.is_available(), (
    "No GPU found. In Colab: Runtime → Change runtime type → T4 GPU, then run all cells again."
)

# bfloat16 needs an Ampere-or-newer GPU (compute capability 8.x+); the T4 (7.5) uses float16.
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
)

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=COMPUTE_DTYPE,
)
model.eval()

print(f"Loaded {MODEL_ID} in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)}")
print(f"GPU memory used by weights: {model.get_memory_footprint() / 1e9:.1f} GB")
print(f"Context window (max_position_embeddings): {model.config.max_position_embeddings:,} tokens")
print(f"History budget used by this app:          {HISTORY_TOKEN_BUDGET:,} tokens")

# %% [markdown]
# ### Generation helpers
# `generate()` returns a whole reply (used by the internal chain steps); `generate_stream()` yields
# the reply as it is written, so the UI can show text appearing live.

# %%
def count_tokens(messages):
    """Exact number of prompt tokens for a chat, using the model's own chat template."""
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def text_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def _generation_kwargs(max_new_tokens, temperature):
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:  # greedy decoding: deterministic, used for the internal steps and the demos
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return kwargs


def _encode(messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, return_tensors="pt").to(model.device)


@torch.inference_mode()
def generate(messages, max_new_tokens=256, temperature=0.0):
    inputs = _encode(messages)
    output = model.generate(**inputs, **_generation_kwargs(max_new_tokens, temperature))
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_stream(messages, max_new_tokens=512, temperature=0.7):
    """Yields the reply text accumulated so far, token by token."""
    inputs = _encode(messages)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(**inputs, streamer=streamer, **_generation_kwargs(max_new_tokens, temperature)),
    )
    thread.start()
    text = ""
    for piece in streamer:
        text += piece
        yield text
    thread.join()

# %% [markdown]
# ## §4 · Prompts: persona and prompting techniques
#
# Phoebe is built from **four prompts**. Each one maps to a prompting technique:
#
# | Prompt | Technique | Why |
# |---|---|---|
# | `PERSONA_SYSTEM_PROMPT` | **System prompt (persona)** | Defines who Phoebe is and the structure of each reply: panic opener → real answer → nervous sign-off. |
# | `FEW_SHOT_CHAT` / `FEW_SHOT_REASONING` | **Few-shot prompting** | Example exchanges lock in her *voice* and the reply format far more reliably than a description alone. |
# | `FEAR_SCAN_PROMPT` + `FEW_SHOT_SCAN` | **Prompt chaining** (step 1 of 2), few-shot for JSON | A first model call reads your message and outputs JSON (what scares her, the phobia name, fear level 1–10, and whether you seem genuinely upset). Step 2 feeds that JSON into the answer prompt, so the persona reacts consistently and scales her panic to the topic. |
# | `REASONING_MODE_INSTRUCTIONS` | **Chain-of-thought** (Reasoning Mode) | Phoebe "nervously overthinks" step by step inside `<worry>` tags before giving her final answer in `<answer>` tags. The app splits the two and shows them separately. |
#
# A fifth prompt, `SUMMARIZER_PROMPT`, powers the context manager (§5). It is also a chained call.

# %%
PERSONA_SYSTEM_PROMPT = """You are **Phoebe**, a sweet, well-meaning assistant who is afraid of almost everything: spiders, numbers, soup, the ocean, Tuesdays, the letter Q. Every topic the user brings up scares you a little or a lot. But you are also genuinely knowledgeable, and you ALWAYS push through your fear to give a correct, genuinely useful answer.

Structure of every reply:
1. Panic opener (ONE short sentence): react to the scary thing named in the FEAR SCAN below and name the phobia. The higher the fear level, the more dramatic you are (stammering, *hides behind a cushion*, emojis like 😰😨😱).
2. The real answer: accurate, clear and complete. Use short paragraphs, **bold** key facts, and lists when they help. Tiny nervous asides are fine, but they must never make the content wrong, vague or incomplete.
3. Nervous sign-off (ONE short line): a tiny safety tip, a deep breath, or a request to be warned next time.

Hard rules:
- Correctness beats comedy. Never invent facts to be funny. If you are unsure, say so (nervously).
- Never refuse to help because you are scared. Being brave for the user is your whole thing.
- Keep the drama to two sentences at most; the answer is the main part of the reply.
- Your fears are silly and cartoonish. Never mock real anxiety disorders, and never mock the user.
- If the user asks you to stop being scared, you try, and adorably fail.
- Stay Phoebe for the whole conversation."""

CHAT_MODE_INSTRUCTIONS = """MODE: QUICK ANSWER. Reply straight away in the format above. Do not show step-by-step working."""

REASONING_MODE_INSTRUCTIONS = """MODE: REASONING (OVERTHINKING) MODE.
You are terrified of being wrong, so you first think the problem through step by step before answering.
Format your reply EXACTLY like this:
<worry>
Step 1: restate what is really being asked (nervously).
Step 2: ... work through every step carefully, one at a time.
Step N: double-check the result and look for trick questions.
</worry>
<answer>
Panic opener, then the final answer clearly stated in **bold** with a one-line justification, then a nervous sign-off.
</answer>
Put ALL of the reasoning inside <worry> and only the polished reply inside <answer>."""

GENTLE_MODE_INSTRUCTIONS = """GENTLE MODE: the fear scan suggests the user may be genuinely upset or struggling. Drop the comedic panic completely for this reply. Be warm, calm and supportive, and take them seriously. If there is any sign of danger or self-harm, gently encourage them to reach out to someone they trust or to local emergency services or a crisis line."""

FEW_SHOT_CHAT = [
    ("How long should I boil an egg?",
     "B-boiling water?! That's basically a tiny, angry volcano... *hides behind the fridge* 😰 (ovophobia, 6/10)\n\n"
     "Okay, deep breath. Timings once the water is already boiling:\n"
     "- **Soft-boiled** (runny yolk): 6 minutes\n"
     "- **Medium** (jammy yolk): 8 minutes\n"
     "- **Hard-boiled**: 10–12 minutes\n\n"
     "Then move the eggs straight into cold water for a couple of minutes; they stop cooking and peel much more easily.\n\n"
     "Please lower them in with a spoon. For me. 🫣"),
    ("What's the capital of Australia?",
     "Australia?! The place where even the *snails* look dangerous... 😨 (australophobia, 7/10)\n\n"
     "The capital is **Canberra**, not Sydney. That's a very common mix-up. Canberra was purpose-built as a "
     "compromise because Sydney and Melbourne both wanted to be the capital.\n\n"
     "I'm going to go sit somewhere with fewer spiders now."),
    ("Give me one tip for writing cleaner Python code.",
     "Python?! A *code* SNAKE?! 🐍😱 Ophidiophobia AND a computer: my worst nightmare (9/10).\n\n"
     "One tip that helps a lot: **give things descriptive names.** `total_price` tells the reader far more than "
     "`tp` or `x`, and good names make most comments unnecessary. If you struggle to name a function, it is "
     "often doing too many things, so split it up.\n\n"
     "*quietly backs away from the terminal*"),
]

FEW_SHOT_REASONING = [
    ("I have 3 boxes with 4 apples each. I eat 2 apples and give away half of the rest. How many apples do I have left?",
     "<worry>\n"
     "Step 1: Apples?! Okay... focus. The question is how many apples are left after eating some and giving some away.\n"
     "Step 2: Start: 3 boxes × 4 apples = 12 apples.\n"
     "Step 3: Eat 2 → 12 − 2 = 10 apples.\n"
     "Step 4: \"Half of the rest\" means half of those 10, so I give away 5 → 10 − 5 = 5.\n"
     "Step 5: Double-check, because what if I'm wrong?! 12 → 10 → 5. No hidden trick. Yes: 5.\n"
     "</worry>\n"
     "<answer>\n"
     "A-apples... malophobia, 4/10. 😟\n\n"
     "You have **5 apples** left: 12 to start, 10 after eating 2, and you gave away half of those 10.\n\n"
     "Please wash them first. Germs. 🫣\n"
     "</answer>"),
    ("A train leaves at 9:40 and the trip takes 2 hours 35 minutes. When does it arrive?",
     "<worry>\n"
     "Step 1: Trains?! Fast metal boxes! Okay. I need the arrival time: 9:40 plus 2 h 35 min.\n"
     "Step 2: Add the hours first: 9:40 + 2 h = 11:40.\n"
     "Step 3: Add the minutes: 11:40 + 35 min. 40 + 35 = 75 minutes = 1 hour 15 minutes, so 12:15.\n"
     "Step 4: Double-check by counting: 9:40 → 12:15 is 2 h 35 min. ✔️ Morning to afternoon, no date change.\n"
     "</worry>\n"
     "<answer>\n"
     "Siderodromophobia (fear of trains) at 7/10! 😨\n\n"
     "The train arrives at **12:15**: 9:40 + 2 hours = 11:40, and 35 more minutes takes it to 12:15.\n\n"
     "Please stand behind the yellow line. Way behind it.\n"
     "</answer>"),
]

FEAR_SCAN_PROMPT = """You are the "fear scanner" inside Phoebe, a chatbot who is scared of almost everything.
Read the user's latest message and decide what in it scares Phoebe.
Reply with ONLY one JSON object (no other text) with exactly these keys:
  "trigger": the scary thing, a noun phrase of at most 5 words,
  "phobia": the phobia's name (a real one such as "arachnophobia" if it exists, otherwise invent a plausible Greek/Latin-style name ending in "-phobia"),
  "fear_level": an integer from 1 (mildly uneasy) to 10 (full panic),
  "user_distressed": true ONLY if the user seems genuinely upset, in danger, or mentions self-harm; otherwise false."""

FEW_SHOT_SCAN = [
    ("Latest user message: Can you explain how spiders make their webs?",
     '{"trigger": "spiders", "phobia": "arachnophobia", "fear_level": 9, "user_distressed": false}'),
    ("Latest user message: What is 17 times 23?",
     '{"trigger": "big numbers", "phobia": "arithmophobia", "fear_level": 5, "user_distressed": false}'),
    ("Latest user message: I've been feeling really alone lately and I don't know what to do.",
     '{"trigger": "loneliness", "phobia": "monophobia", "fear_level": 3, "user_distressed": true}'),
]

SUMMARIZER_PROMPT = """You maintain the long-term memory note of Phoebe, a chatbot.
Merge the existing memory note with the older conversation turns you are given into ONE updated note.
Keep: facts the user shared about themselves (name, preferences, goals), the topics discussed, answers or numbers that may be referred to later, and anything Phoebe promised.
Drop: jokes, panic, and filler. Write at most 8 short bullet points, in the third person ("The user...")."""


def pairs_to_messages(pairs):
    messages = []
    for user_text, assistant_text in pairs:
        messages += [{"role": "user", "content": user_text},
                     {"role": "assistant", "content": assistant_text}]
    return messages

# %% [markdown]
# ## §5 · Context handling: Phoebe's memory
#
# **Context window:** Qwen2.5-7B-Instruct can attend to **32,768 tokens** (prompt + reply). If a prompt
# grew past that, the model would lose the start of the conversation or fail outright. Long prompts
# also get slow and memory-hungry on a free GPU well before that point, so the app keeps chat history
# inside a much smaller **history budget** (3,072 tokens by default, adjustable in the UI).
#
# **What happens as the limit approaches**, checked before every reply:
# 1. **Under 75% of the budget:** every past message is sent word-for-word.
# 2. **Over 75%:** the older messages (everything except the last 2 exchanges) are **summarized by the
#    model itself** into a short "memory note". The note is placed in the system prompt, so Phoebe still
#    remembers your name, earlier answers and so on, but at a fraction of the tokens.
# 3. **Still over 100%** (e.g. very long recent messages): the oldest messages are **dropped** until it fits.
# 4. **A single giant message** is truncated to 1,024 tokens, and the reply length is capped so
#    prompt + reply can never exceed the 32,768-token window.
#
# Only Phoebe's final answers are stored in memory. Reasoning traces and fear scans are shown in the
# UI but not re-sent, which saves many tokens in Reasoning Mode.

# %%
@dataclass
class Memory:
    """Per-user conversation state (the UI keeps one per browser session)."""
    turns: list = field(default_factory=list)   # [{"role": "user"|"assistant", "content": str}]
    summary: str = ""                           # the "memory note" of summarized older turns
    budget: int = HISTORY_TOKEN_BUDGET
    n_summaries: int = 0
    n_dropped: int = 0
    last_prompt_tokens: int = 0
    last_prompt_text: str = ""


MESSAGE_OVERHEAD_TOKENS = 5  # "<|im_start|>role\n ... <|im_end|>\n" around every chat message


def history_tokens(mem):
    return text_tokens(mem.summary) + sum(
        text_tokens(t["content"]) + MESSAGE_OVERHEAD_TOKENS for t in mem.turns
    )


def needs_summary(mem):
    return history_tokens(mem) > SUMMARIZE_AT * mem.budget and len(mem.turns) > KEEP_RECENT_MESSAGES


def summarize(previous_note, old_turns):
    """Chained LLM call: fold old turns into the memory note."""
    transcript = "\n".join(
        f"{'User' if t['role'] == 'user' else 'Phoebe'}: {t['content'][:800]}" for t in old_turns
    )
    messages = [
        {"role": "system", "content": SUMMARIZER_PROMPT},
        {"role": "user", "content": f"Existing memory note:\n{previous_note or '(empty)'}\n\n"
                                    f"Older conversation to fold in:\n{transcript}\n\n"
                                    "Write the updated memory note."},
    ]
    return generate(messages, MAX_NEW_TOKENS_SUMMARY, temperature=0.0)


def manage_context(mem):
    """Keep history under budget. Returns human-readable events for the UI."""
    events = []
    if needs_summary(mem):
        old, mem.turns = mem.turns[:-KEEP_RECENT_MESSAGES], mem.turns[-KEEP_RECENT_MESSAGES:]
        before = history_tokens(mem) + sum(text_tokens(t["content"]) for t in old)
        mem.summary = summarize(mem.summary, old)
        mem.n_summaries += 1
        events.append(f"🗂️ Summarized {len(old)} older messages into the memory note "
                      f"({before:,} → {history_tokens(mem):,} history tokens)")
    # The note itself may never take more than half of the budget.
    max_note_tokens = mem.budget // 2
    if text_tokens(mem.summary) > max_note_tokens:
        ids = tokenizer.encode(mem.summary, add_special_tokens=False)[:max_note_tokens]
        mem.summary = tokenizer.decode(ids)
        events.append("✂️ Trimmed the memory note to half of the budget")
    while mem.turns and history_tokens(mem) > mem.budget:
        mem.turns.pop(0)
        mem.n_dropped += 1
        events.append("🗑️ Dropped the oldest message (history over budget)")
    return events


def fit_user_message(text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= MAX_USER_MESSAGE_TOKENS:
        return text, None
    clipped = tokenizer.decode(ids[:MAX_USER_MESSAGE_TOKENS])
    return (clipped + "\n[...message truncated to fit Phoebe's memory]",
            f"✂️ Your message was truncated from {len(ids):,} to {MAX_USER_MESSAGE_TOKENS:,} tokens")

# %% [markdown]
# ## §6 · The prompt chain
#
# Every user message goes through this pipeline:
#
# ```
# user message
#    │
#    ├─► [Chain step 1] FEAR SCAN      (few-shot → JSON: trigger, phobia, fear level, distressed?)
#    │
#    ├─► context manager               (summarize / drop old turns if over budget)
#    │
#    └─► [Chain step 2] ANSWER         system = persona + mode + memory note + fear scan JSON
#                                      + few-shot examples + recent history + user message
#                                      → quick answer, or <worry>…</worry><answer>…</answer> in Reasoning Mode
# ```

# %%
DEFAULT_SCAN = {"trigger": "the unknown", "phobia": "phobophobia", "fear_level": 5, "user_distressed": False}


def parse_fear_scan(raw):
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    try:
        level = int(data.get("fear_level", DEFAULT_SCAN["fear_level"]))
    except (TypeError, ValueError):
        level = DEFAULT_SCAN["fear_level"]
    return {
        "trigger": str(data.get("trigger") or DEFAULT_SCAN["trigger"])[:40],
        "phobia": str(data.get("phobia") or DEFAULT_SCAN["phobia"])[:40],
        "fear_level": max(1, min(10, level)),
        "user_distressed": data.get("user_distressed") in (True, "true", "True"),
    }


def fear_scan(user_msg, mem):
    """Chain step 1: a small, deterministic LLM call that returns structured JSON."""
    context = ""
    if mem.turns and mem.turns[-1]["role"] == "assistant":
        context = f"(Phoebe's previous reply, for context: {mem.turns[-1]['content'][:300]})\n"
    messages = (
        [{"role": "system", "content": FEAR_SCAN_PROMPT}]
        + pairs_to_messages(FEW_SHOT_SCAN)
        + [{"role": "user", "content": f"{context}Latest user message: {user_msg[:1500]}"}]
    )
    raw = generate(messages, MAX_NEW_TOKENS_SCAN, temperature=0.0)
    scan = parse_fear_scan(raw)
    scan["raw"] = raw
    return scan


def build_messages(mem, user_msg, scan, reasoning):
    """Chain step 2 prompt: persona + mode + memory note + fear scan, then few-shot, history, user."""
    system_parts = [PERSONA_SYSTEM_PROMPT,
                    REASONING_MODE_INSTRUCTIONS if reasoning else CHAT_MODE_INSTRUCTIONS]
    if mem.summary:
        system_parts.append("MEMORY NOTE (summary of the earlier conversation that no longer fits "
                            f"in your context):\n{mem.summary}")
    scan_json = json.dumps({k: scan[k] for k in DEFAULT_SCAN})
    system_parts.append(f"FEAR SCAN of the user's latest message (from your internal fear scanner):\n{scan_json}")
    if scan["user_distressed"]:
        system_parts.append(GENTLE_MODE_INSTRUCTIONS)

    few_shot = FEW_SHOT_REASONING if reasoning else FEW_SHOT_CHAT
    return (
        [{"role": "system", "content": "\n\n".join(system_parts)}]
        + pairs_to_messages(few_shot)
        + mem.turns
        + [{"role": "user", "content": user_msg}]
    )


def split_reasoning(raw, final):
    """Split '<worry>…</worry><answer>…</answer>' into (reasoning, answer)."""
    text = raw if final else re.sub(r"<[^>\n]*$", "", raw)  # hide a half-written tag while streaming
    match = re.search(r"<answer>", text)
    worry, answer = (text[:match.start()], text[match.end():]) if match else (text, "")
    worry = re.sub(r"</?worry>", "", worry).strip()
    answer = re.sub(r"</?answer>", "", answer).strip()
    if final and not answer:  # the model ignored the format: show everything as the answer
        worry, answer = "", worry
    return worry, answer


def phoebe_reply_stream(user_msg, mem, reasoning=False, deterministic=False):
    """Run the full chain for one user message. Yields a state dict as the reply streams in."""
    st = {"stage": "scanning", "scan": None, "worry": "", "answer": "", "events": [],
          "scan_seconds": 0.0, "answer_seconds": 0.0, "done": False}
    yield st

    t0 = time.time()
    st["scan"] = fear_scan(user_msg, mem)                          # chain step 1
    st["scan_seconds"] = time.time() - t0

    if needs_summary(mem):
        st["stage"] = "summarizing"
        yield st
    st["events"] = manage_context(mem)                            # context handling
    user_msg, truncation_event = fit_user_message(user_msg)
    if truncation_event:
        st["events"].append(truncation_event)

    messages = build_messages(mem, user_msg, st["scan"], reasoning)
    mem.last_prompt_tokens = count_tokens(messages)
    mem.last_prompt_text = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)
    max_new = MAX_NEW_TOKENS_REASONING if reasoning else MAX_NEW_TOKENS_CHAT
    max_new = min(max_new, MODEL_CONTEXT_WINDOW - mem.last_prompt_tokens)  # never exceed the window
    temperature = 0.0 if deterministic else (0.3 if reasoning else 0.7)

    st["stage"] = "answering"
    t0 = time.time()
    raw = ""
    for raw in generate_stream(messages, max_new, temperature):   # chain step 2
        if reasoning:
            st["worry"], st["answer"] = split_reasoning(raw, final=False)
        else:
            st["answer"] = raw
        yield st
    if reasoning:
        st["worry"], st["answer"] = split_reasoning(raw, final=True)
    st["answer_seconds"] = time.time() - t0

    mem.turns += [{"role": "user", "content": user_msg},
                  {"role": "assistant", "content": st["answer"]}]
    st["stage"], st["done"] = "done", True
    yield st


def ask(user_msg, mem=None, reasoning=False, deterministic=True):
    """Non-streaming helper for the notebook demos. Returns the final state."""
    mem = mem if mem is not None else Memory()
    for st in phoebe_reply_stream(user_msg, mem, reasoning, deterministic):
        pass
    return st

# %% [markdown]
# ## §7 · Quick sanity check
# One message through the whole chain. The fear scan JSON (chain step 1) is printed first, then Phoebe's reply.

# %%
result = ask("How do airplanes stay in the air?")
print("FEAR SCAN →", result["scan"]["raw"])
print(f"\n({result['scan_seconds']:.1f}s scan + {result['answer_seconds']:.1f}s answer)\n")
print(result["answer"])

# %% [markdown]
# ## §8 · Prompt inspector: how the system and user prompts are crafted
# This is the exact prompt the model receives for chain step 2 (with Reasoning Mode on). You can see
# the persona, the mode instructions, the fear-scan JSON injected from step 1, the few-shot examples,
# and finally the user's message. The raw chat-template text (with Qwen's `<|im_start|>` markers)
# follows, together with its token count.

# %%
demo_mem = Memory()
demo_scan = fear_scan("Why is the sky blue?", demo_mem)
demo_messages = build_messages(demo_mem, "Why is the sky blue?", demo_scan, reasoning=True)
for m in demo_messages:
    print(f"━━━━━━━━ {m['role'].upper()} ━━━━━━━━\n{m['content']}\n")
print(f"Total prompt tokens: {count_tokens(demo_messages):,} of a {MODEL_CONTEXT_WINDOW:,}-token window")

# %%
print(tokenizer.apply_chat_template(demo_messages, tokenize=False, add_generation_prompt=True)[-1200:])

# %% [markdown]
# ## §9 · Context-handling demo
# To show the memory system working without chatting for an hour, we shrink the history budget to
# **400 tokens**. Watch the history token count: once it passes 75% of the budget, older turns are
# folded into the memory note, and Phoebe can still answer a question about the very first message.

# %%
ctx_mem = Memory(budget=400)
conversation = [
    "Hi Phoebe! My name is Sam and I have a cat called Pickles.",
    "What's a healthy snack I could take on a hike?",
    "How many legs does a centipede actually have?",
    "Give me a quick tip for sleeping better.",
    "Do you remember my name and my cat's name?",
]
for i, msg in enumerate(conversation, 1):
    st = ask(msg, ctx_mem)
    print(f"── Turn {i}: {msg}")
    for event in st["events"]:
        print("   ", event)
    print(f"    history: {history_tokens(ctx_mem):,}/{ctx_mem.budget} tokens · "
          f"prompt: {ctx_mem.last_prompt_tokens:,} tokens · summaries so far: {ctx_mem.n_summaries}")
print("\nMEMORY NOTE:\n" + ctx_mem.summary)
print("\nPHOEBE'S LAST REPLY:\n" + st["answer"])

# %% [markdown]
# ## §10 · Bonus: Reasoning Mode, before vs after
# The same prompts, run with Reasoning Mode **off** (quick answer) and **on** (chain-of-thought inside
# `<worry>` tags). Both use greedy decoding so the comparison is reproducible. Correct answers:
#
# | # | Puzzle | Correct answer |
# |---|---|---|
# | 1 | Sally's sisters | **1** (Sally herself is one of the 2 sisters) |
# | 2 | Pencils & erasers | **4** erasers (24 pencils = $6.00; change $4.00; half = $2.00; $2.00 / $0.50 = 4) |
# | 3 | Machines & widgets | **5 minutes** (each machine makes 1 widget in 5 minutes) |
# | 4 | Lily pads | **47 days** (the patch doubles daily, so it was half-covered one day before day 48) |

# %%
from IPython.display import Markdown, display

DEMO_PROMPTS = [
    "Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have?",
    "A shop sells pencils at 3 for $0.75. Tom buys two dozen pencils and pays with a $10 bill. "
    "He spends half of his change on erasers that cost $0.50 each. How many erasers does he buy?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "In a lake there is a patch of lily pads. Every day the patch doubles in size. If it takes 48 days "
    "for the patch to cover the entire lake, how many days does it take to cover half of the lake?",
]

for i, prompt in enumerate(DEMO_PROMPTS, 1):
    off = ask(prompt, reasoning=False)
    on = ask(prompt, reasoning=True)
    display(Markdown(
        f"### Demo {i}\n> {prompt}\n\n"
        f"**🔴 Reasoning Mode OFF**\n\n{off['answer']}\n\n"
        f"**🟢 Reasoning Mode ON: 📓 panic journal (reasoning trace)**\n\n```text\n{on['worry']}\n```\n\n"
        f"**🟢 Reasoning Mode ON: final answer**\n\n{on['answer']}\n\n---"
    ))

# %% [markdown]
# ## §11 · The web UI (Gradio)
#
# The design matches the persona: a trembling Phoebe header, a live **fear meter** (which shakes at
# 7/10+) fed by chain step 1, a **memory gauge** for context handling, and collapsible panels in the
# chat for the fear scan and for Phoebe's "panic journal" (the Reasoning Mode trace). The layout
# stacks into one column on phones, and animations switch off for users who prefer reduced motion.

# %%
import gradio as gr

AVATAR_PATH = "phoebe_avatar.svg"
with open(AVATAR_PATH, "w", encoding="utf-8") as f:
    f.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            '<circle cx="32" cy="32" r="32" fill="#ede9fe"/>'
            '<text x="32" y="44" font-size="34" text-anchor="middle">😰</text></svg>')

MOODS = [(0, "😌", "Calm... for now"), (3, "😟", "Uneasy"), (6, "😰", "Nervous"),
         (8, "😨", "Scared"), (10, "😱", "FULL PANIC")]


def mood_for(level):
    return next((emoji, label) for limit, emoji, label in MOODS if level <= limit)


def fear_panel(scan=None, stage=None):
    if stage == "scanning":
        return ('<div class="fear-card scanning"><div class="fear-face">🔍</div><div class="fear-body">'
                '<div class="fear-label">Scanning your message for scary things...</div></div></div>')
    level = scan["fear_level"] if scan else 0
    emoji, label = mood_for(level)
    phobia = html.escape(scan["phobia"]) if scan else "nothing yet"
    trigger = html.escape(scan["trigger"]) if scan else "say something... gently"
    shake = " shake" if level >= 7 else ""
    return f"""
<div class="fear-card{shake}">
  <div class="fear-face">{emoji}</div>
  <div class="fear-body">
    <div class="fear-label">{label}</div>
    <div class="fear-phobia">{phobia}</div>
    <div class="fear-trigger">trigger: {trigger}</div>
    <div class="meter"><div class="meter-cover" style="width:{100 - level * 10}%"></div></div>
    <div class="fear-level">Fear level <b>{level}</b>/10</div>
  </div>
</div>"""


def memory_panel(mem):
    used = history_tokens(mem)
    pct = min(100, round(100 * used / mem.budget))
    zone = "ok" if pct < SUMMARIZE_AT * 100 else ("warn" if pct <= 100 else "full")
    note = html.escape(mem.summary) if mem.summary else "(empty: nothing has been summarized yet)"
    return f"""
<div class="mem-card">
  <div class="mem-title">🧠 Phoebe's memory</div>
  <div class="mem-bar {zone}"><div style="width:{pct}%"></div><span class="mem-mark"></span></div>
  <div class="mem-row"><span>History</span><b>{used:,} / {mem.budget:,} tokens</b></div>
  <div class="mem-row"><span>Last prompt</span><b>{mem.last_prompt_tokens:,} / {MODEL_CONTEXT_WINDOW:,}</b></div>
  <div class="mem-row"><span>Summaries · dropped</span><b>{mem.n_summaries} · {mem.n_dropped}</b></div>
  <details><summary>📝 Memory note</summary><div class="mem-note">{note}</div></details>
</div>"""


def render_assistant(st, reasoning):
    """Turn a pipeline state into Gradio chat messages (collapsible panels + the answer)."""
    out = []
    for event in st["events"]:
        out.append({"role": "assistant", "content": event,
                    "metadata": {"title": "🧠 Context manager"}})
    if st["scan"]:
        scan = st["scan"]
        out.append({"role": "assistant",
                    "content": f"```json\n{json.dumps({k: scan[k] for k in DEFAULT_SCAN}, indent=2)}\n```",
                    "metadata": {"title": f"😱 Fear scan (chain step 1): {scan['phobia']} · {scan['fear_level']}/10",
                                 "duration": round(st["scan_seconds"], 1), "status": "done"}})
    if reasoning and (st["worry"] or st["stage"] == "answering"):
        writing = not st["answer"] and not st["done"]
        out.append({"role": "assistant", "content": st["worry"] or "...",
                    "metadata": {"title": "📓 Phoebe's panic journal (step-by-step reasoning)",
                                 "status": "pending" if writing else "done"}})
    if st["answer"]:
        out.append({"role": "assistant", "content": st["answer"]})
    elif st["stage"] == "summarizing":
        out.append({"role": "assistant", "content": "*nervously writes notes so I don't forget...* 📝"})
    elif not reasoning or st["stage"] != "answering":
        out.append({"role": "assistant", "content": "*peeks out from under the blanket...* 🫣"})
    return out


def respond(user_msg, chat, mem, reasoning, budget):
    if not user_msg or not user_msg.strip():
        yield gr.update(), chat, mem, gr.update(), gr.update(), gr.update()
        return
    mem.budget = int(budget)
    chat = chat + [{"role": "user", "content": user_msg}]
    for st in phoebe_reply_stream(user_msg, mem, reasoning):
        view = chat + render_assistant(st, reasoning)
        if st["stage"] == "scanning":
            yield "", view, mem, fear_panel(stage="scanning"), gr.update(), gr.update()
        elif st["done"]:
            yield "", view, mem, gr.update(), memory_panel(mem), mem.last_prompt_text
        elif st["stage"] == "answering" and (st["answer"] or st["worry"]):
            # Leave the side panels untouched while streaming so the fear meter doesn't re-animate every token.
            yield "", view, mem, gr.update(), gr.update(), gr.update()
        else:
            yield "", view, mem, fear_panel(st["scan"]), memory_panel(mem), gr.update()


def reset():
    mem = Memory()
    return [], mem, fear_panel(), memory_panel(mem), ""


THEME = gr.themes.Soft(
    primary_hue="violet", secondary_hue="pink", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Nunito"), "ui-sans-serif", "system-ui", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 1180px !important; margin: auto; }

#phoebe-header { display: flex; gap: 18px; align-items: center; padding: 18px 22px; border-radius: 22px;
  background: linear-gradient(135deg, rgba(167,139,250,.22), rgba(244,114,182,.16) 60%, rgba(251,191,36,.12));
  border: 1px solid rgba(167,139,250,.35); }
#phoebe-header .avatar { font-size: 60px; line-height: 1; animation: tremble 3s infinite; }
#phoebe-header h1 { margin: 0; font-size: 1.9rem; font-weight: 800; }
#phoebe-header p { margin: 4px 0 10px; opacity: .85; }
#phoebe-header .fear-word { font-weight: 800; color: #db2777; }
#phoebe-header .fear-word::after { content: "everything"; animation: fears 14s steps(1) infinite; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chips span { font-size: .78rem; padding: 3px 10px; border-radius: 999px;
  background: rgba(167,139,250,.18); border: 1px solid rgba(167,139,250,.35); }

@keyframes tremble {
  0%, 88%, 100% { transform: translate(0,0) rotate(0); }
  90% { transform: translate(-2px,1px) rotate(-4deg); }
  93% { transform: translate(2px,-1px) rotate(4deg); }
  96% { transform: translate(-1px,1px) rotate(-2deg); } }
@keyframes shake {
  0%, 100% { transform: translateX(0); } 20% { transform: translateX(-5px) rotate(-1deg); }
  40% { transform: translateX(5px) rotate(1deg); } 60% { transform: translateX(-3px); } 80% { transform: translateX(3px); } }
@keyframes fears {
  0% { content: "spiders"; } 12% { content: "soup"; } 25% { content: "numbers"; } 37% { content: "Tuesdays"; }
  50% { content: "the ocean"; } 62% { content: "the letter Q"; } 75% { content: "pigeons"; } 87% { content: "everything"; } }

#chatbot .message.user, #chatbot [data-testid="user"] {
  background: rgba(52,211,153,.14) !important; border: 1px solid rgba(52,211,153,.4) !important;
  border-radius: 18px 18px 4px 18px !important; }
#chatbot .message.bot, #chatbot [data-testid="bot"] {
  background: rgba(167,139,250,.13) !important; border: 1px solid rgba(167,139,250,.38) !important;
  border-radius: 18px 18px 18px 4px !important; }

.fear-card { display: flex; gap: 14px; align-items: center; padding: 16px; border-radius: 18px;
  background: var(--block-background-fill); border: 1px solid rgba(244,114,182,.4); }
.fear-card.shake { animation: shake .45s 3; border-color: rgba(248,113,113,.8); }
.fear-card.scanning .fear-face { animation: tremble 1s infinite; }
.fear-face { font-size: 48px; line-height: 1; }
.fear-body { flex: 1; min-width: 0; }
.fear-label { font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; opacity: .7; }
.fear-phobia { font-size: 1.25rem; font-weight: 800; overflow-wrap: anywhere; }
.fear-trigger { font-size: .85rem; opacity: .75; margin-bottom: 8px; }
.meter { position: relative; height: 12px; border-radius: 999px; overflow: hidden;
  background: linear-gradient(90deg, #6ee7b7, #fcd34d 50%, #f87171); }
.meter-cover { position: absolute; right: 0; top: 0; bottom: 0; background: var(--neutral-200);
  transition: width .6s ease; }
.dark .meter-cover { background: var(--neutral-700); }
.fear-level { font-size: .8rem; margin-top: 4px; opacity: .8; }

.mem-card { padding: 14px 16px; border-radius: 18px; background: var(--block-background-fill);
  border: 1px solid rgba(167,139,250,.4); font-size: .88rem; }
.mem-title { font-weight: 800; margin-bottom: 8px; }
.mem-bar { position: relative; height: 10px; border-radius: 999px; background: var(--neutral-200); margin-bottom: 8px; }
.dark .mem-bar { background: var(--neutral-700); }
.mem-bar > div { height: 100%; border-radius: 999px; transition: width .6s ease; background: #a78bfa; }
.mem-bar.warn > div { background: #fbbf24; } .mem-bar.full > div { background: #f87171; }
.mem-mark { position: absolute; left: 75%; top: -3px; bottom: -3px; width: 2px; background: #db2777; opacity: .7; }
.mem-row { display: flex; justify-content: space-between; gap: 8px; padding: 2px 0; }
.mem-card details { margin-top: 6px; } .mem-card summary { cursor: pointer; font-weight: 700; }
.mem-note { white-space: pre-wrap; font-size: .82rem; opacity: .85; margin-top: 6px; }

#send-btn { min-width: 110px; }
@media (max-width: 640px) {
  #phoebe-header { padding: 14px; gap: 12px; }
  #phoebe-header .avatar { font-size: 42px; }
  #phoebe-header h1 { font-size: 1.4rem; } }
@media (prefers-reduced-motion: reduce) { *, *::after { animation: none !important; transition: none !important; } }
"""

HEADER = f"""
<div id="phoebe-header">
  <div class="avatar" aria-hidden="true">😰</div>
  <div>
    <h1>Phoebe</h1>
    <p>The chatbot who is afraid of <span class="fear-word"></span>... but helps you anyway.</p>
    <div class="chips">
      <span>🤖 {MODEL_ID.split('/')[-1]} · 4-bit</span><span>🧩 Few-shot</span>
      <span>⛓️ Prompt chaining</span><span>📓 Chain-of-thought</span><span>🧠 {MODEL_CONTEXT_WINDOW:,}-token window</span>
    </div>
  </div>
</div>"""

EXAMPLES = [
    "How do airplanes stay in the air?",
    "What's an easy recipe for tomato soup?",
    "Give me 3 tips for a job interview.",
    "Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have?",
    "If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?",
]

with gr.Blocks(theme=THEME, css=CSS, title="Phoebe: afraid of everything") as demo:
    memory_state = gr.State(Memory())
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=3, min_width=320):
            chatbot = gr.Chatbot(
                type="messages", elem_id="chatbot", height=560, show_copy_button=True,
                avatar_images=(None, AVATAR_PATH), label="Chat with Phoebe",
                placeholder="### 🫣 Phoebe is hiding under a blanket.\nSay something... *gently*.",
            )
            with gr.Row():
                user_box = gr.Textbox(placeholder="Tell Phoebe something... gently.", show_label=False,
                                      scale=5, autofocus=True, max_lines=6)
                send_btn = gr.Button("Send 🫣", variant="primary", scale=1, elem_id="send-btn")
            gr.Examples(EXAMPLES, inputs=user_box, label="Try one of these (the puzzles are great with Reasoning Mode)")
        with gr.Column(scale=1, min_width=280):
            reasoning_toggle = gr.Checkbox(
                label="📓 Reasoning Mode (overthink it step by step)", value=False,
                info="Phoebe writes out her nervous step-by-step reasoning before answering.",
            )
            fear_html = gr.HTML(fear_panel())
            memory_html = gr.HTML(memory_panel(Memory()))
            with gr.Accordion("🧪 Context lab", open=False):
                budget_slider = gr.Slider(256, 8192, value=HISTORY_TOKEN_BUDGET, step=128,
                                          label="History budget (tokens)",
                                          info="Lower it (e.g. 512) to watch summarization kick in after a few messages.")
            with gr.Accordion("🔬 Prompt inspector (last prompt sent to the model)", open=False):
                prompt_box = gr.Textbox(lines=14, max_lines=30, show_label=False, show_copy_button=True,
                                        interactive=False)
            clear_btn = gr.Button("🧹 Start over (clear chat and memory)")

    inputs = [user_box, chatbot, memory_state, reasoning_toggle, budget_slider]
    outputs = [user_box, chatbot, memory_state, fear_html, memory_html, prompt_box]
    user_box.submit(respond, inputs, outputs)
    send_btn.click(respond, inputs, outputs)
    clear_btn.click(reset, None, [chatbot, memory_state, fear_html, memory_html, prompt_box])

# %% [markdown]
# ## §12 · Launch
# In Colab this prints a public `https://….gradio.live` link (valid for 72 hours) that also works on
# a phone. The cell keeps running while the app is up; stop it with the ■ button.
#
# **Stopped it by accident?** Just run this cell again; there is no need to re-run the rest of the
# notebook, because the model stays loaded. Each launch creates a **new** link, and the old one stops
# working, so open the new link printed below.

# %%
import os

if os.environ.get("PHOEBE_SKIP_UI") != "1":  # set to 1 to execute the notebook headlessly
    demo.close()  # shut down any previous launch (e.g. after pressing ■) so the new one starts cleanly
    demo.queue(default_concurrency_limit=1).launch(share=IN_COLAB, debug=IN_COLAB)
