"""Why does exp13's GPT-2 arm produce NaN losses on MI250X?

All three arms (baseline included) returned nan for both epochs, so this is
neither SAL nor the masker. This replays the same data and the same optimizer
step for a few dozen steps and prints where the first non-finite value appears,
under each attention implementation. The suspect is SDPA on ROCm.
"""
import sys

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "gpt2-medium"
MAX_LEN, BS, STEPS = 64, 32, 40

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
rows = load_dataset("stanfordnlp/sst2")["train"].shuffle(seed=42).select(
    range(BS * STEPS))
enc = tok(list(rows["sentence"]), truncation=True, padding="max_length",
          max_length=MAX_LEN, return_tensors="pt")
ids, attn = enc["input_ids"], enc["attention_mask"]
y = torch.tensor(list(rows["label"]))
dev = torch.device("cuda")
print(f"device {torch.cuda.get_device_name(0)} | torch {torch.__version__}",
      flush=True)


def run(impl):
    print(f"\n===== attn_implementation={impl} =====", flush=True)
    torch.manual_seed(42)
    kw = {} if impl == "default" else {"attn_implementation": impl}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, **kw).to(dev)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    print("  resolved impl:", model.config._attn_implementation, flush=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-5,
                                                total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        s = slice(step * BS, (step + 1) * BS)
        out = model(input_ids=ids[s].to(dev), attention_mask=attn[s].to(dev),
                    labels=y[s].to(dev))
        ok_logits = bool(torch.isfinite(out.logits).all())
        out.loss.backward()
        gn = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        bad_g = [n for n, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        bad_w = [n for n, p in model.named_parameters()
                 if not torch.isfinite(p).all()]
        flag = ""
        if not ok_logits:
            flag += " LOGITS_NONFINITE"
        if bad_g:
            flag += f" NAN_GRADS({len(bad_g)}) first={bad_g[0]}"
        if bad_w:
            flag += f" NAN_WEIGHTS({len(bad_w)}) first={bad_w[0]}"
        if step < 6 or flag or step % 10 == 0:
            print(f"  step {step:3d}  loss {out.loss.item():10.4f}  "
                  f"grad_norm {gn:10.3f}{flag}", flush=True)
        if bad_w:
            print("  -> diverged; stopping this implementation", flush=True)
            return step
    del model
    torch.cuda.empty_cache()
    return None


for impl in ("default", "eager"):
    first = run(impl)
    print(f"  verdict {impl}: "
          + ("finite throughout" if first is None else f"NaN weights at step {first}"),
          flush=True)
