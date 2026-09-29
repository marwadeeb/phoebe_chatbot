"""Launch the notebook's app locally for development.

Usage:
    python tools/dev_ui.py          # real model on the local GPU
    python tools/dev_ui.py --fake   # canned replies, no GPU needed (fast UI/CSS iteration)

Executes the notebook source cell by cell, skipping the install cell, the demo cells and the
Colab launch cell, then serves the Gradio app on http://127.0.0.1:7860.
"""
import os
import sys
import time
from pathlib import Path

if "--fake" in sys.argv:  # use the cached tokenizer only; no network calls to the Hugging Face Hub
    os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import SRC, parse_cells  # noqa: E402

FAKE = "--fake" in sys.argv
SKIP_MARKERS = [
    "subprocess.check_call",         # §1 install (Colab only)
    "result = ask(",                 # §7 sanity check
    "demo_mem = Memory()",           # §8 prompt inspector
    "ctx_mem = Memory(",             # §9 context demo
    "DEMOS = [",                     # §10 reasoning demos
    "demo.queue(",                   # §12 launch (we launch below)
]
if FAKE:
    SKIP_MARKERS.append("AutoModelForCausalLM.from_pretrained")  # §3 model loading

FAKE_ANSWER = ("S-soup?! A bowl of hot, bubbling *liquid*... 😰 (zomophobia, 7/10)\n\n"
               "Here is an easy **tomato soup**:\n- Soften 1 onion and 2 garlic cloves in olive oil.\n"
               "- Add 800 g canned tomatoes and 500 ml stock; simmer 20 minutes.\n"
               "- Blend, season with salt, pepper and a pinch of sugar. Costs about $3.50.\n\n"
               "Please let it cool before tasting. For me. 🫣")
FAKE_REASONING = ("<worry>\nStep 1: Soup?! Okay, the user wants a simple recipe.\n"
                  "Step 2: Base: onion, garlic, tomatoes, stock.\nStep 3: Double-check nothing is missing: "
                  "salt, pepper. Yes.\n</worry>\n<answer>\n" + FAKE_ANSWER + "\n</answer>")


def install_fakes(ns):
    from transformers import AutoTokenizer

    ns["tokenizer"] = AutoTokenizer.from_pretrained(ns["MODEL_ID"])
    ns["model"] = None

    def generate(messages, max_new_tokens=256, temperature=0.0):
        system = messages[0]["content"]
        if "fear scanner" in system:
            return '{"trigger": "hot soup", "phobia": "zomophobia", "fear_level": 7, "user_distressed": false}'
        return "- The user asked about soup and airplanes.\n- Phoebe gave a recipe."

    def generate_stream(messages, max_new_tokens=512, temperature=0.7):
        text = FAKE_REASONING if "MODE: REASONING" in messages[0]["content"] else FAKE_ANSWER
        out = ""
        for word in text.split(" "):
            out += ("" if not out else " ") + word
            time.sleep(0.02)
            yield out

    ns["generate"], ns["generate_stream"] = generate, generate_stream


ns = {"__name__": "__main__"}
for cell in parse_cells(SRC.read_text(encoding="utf-8")):
    if cell["type"] != "code":
        continue
    code = "\n".join(cell["lines"])
    if any(marker in code for marker in SKIP_MARKERS):
        continue
    exec(compile(code, str(SRC), "exec"), ns)
    if FAKE and "def generate_stream" in code:  # generation helpers just got defined: swap in fakes
        install_fakes(ns)

ns["demo"].queue(default_concurrency_limit=1).launch(server_port=7860)
