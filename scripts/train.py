import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import time
import urllib.request
import torch
import matplotlib.pyplot as plt
from src.model import CharGPT
from src.masks import causal_mask, sliding_window_mask, block_sparse_mask

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/input.txt"

BLOCK_SIZE = 256
BATCH_SIZE = 64
D_MODEL = 128
D_FF = 512
LR = 3e-4
NUM_STEPS = 5000
EVAL_INTERVAL = 250
EVAL_BATCHES = 50
TRAIN_FRACTION = 0.9
SEEDS = [0, 1, 2]

# chosen so sliding_window and block_sparse land on close mean cells/row at BLOCK_SIZE=256
# (measured: block_sparse=65.5, sliding_window=64.87, dense ceiling=128.5) - see NOTES.md
WINDOW_SIZE = 76
SPARSE_BLOCK_SIZE = 16
NUM_GLOBAL_BLOCKS = 1
NUM_RANDOM_BLOCKS = 2


# downloads TinyShakespeare once, data/ is gitignored so this never gets committed
def download_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH) as f:
        return f.read()


# character-level tokenizer: every unique character in the text is one token
def build_tokenizer(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    return chars, encode, decode


# one random batch, target is the input shifted one character to the right
def get_batch(data, batch_size, device):
    starts = torch.randint(0, len(data) - BLOCK_SIZE - 1, (batch_size,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in starts])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in starts])
    return x.to(device), y.to(device)


# average loss over several batches, no gradient, for a less noisy estimate
def estimate_loss(model, data, mask, device, loss_fn, num_batches):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, BATCH_SIZE, device)
            logits = model(x, mask)
            losses.append(loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def hardware_info():
    if torch.cuda.is_available():
        return {"gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
    return {"gpu": "CPU only", "torch_version": torch.__version__, "cuda_version": "n/a"}


# block_sparse gets its own generator, so its random block choices never touch the global
# RNG that controls model init and batch order - that's what keeps the comparison fair
def build_mask(pattern, seed):
    if pattern == "dense":
        return causal_mask(BLOCK_SIZE)
    if pattern == "sliding_window":
        return sliding_window_mask(BLOCK_SIZE, WINDOW_SIZE)
    if pattern == "block_sparse":
        generator = torch.Generator().manual_seed(seed)
        return block_sparse_mask(BLOCK_SIZE, SPARSE_BLOCK_SIZE, NUM_GLOBAL_BLOCKS, NUM_RANDOM_BLOCKS, generator=generator)
    raise ValueError(pattern)


def pattern_params_str(pattern):
    if pattern == "dense":
        return "causal"
    if pattern == "sliding_window":
        return f"window_size={WINDOW_SIZE}"
    return f"block_size={SPARSE_BLOCK_SIZE},global={NUM_GLOBAL_BLOCKS},random={NUM_RANDOM_BLOCKS}"


# trains one (pattern, seed) run to completion, returns the model and its loss curve
def run(pattern, seed, train_data, val_data, vocab_size, device):
    mask = build_mask(pattern, seed).to(device)
    mean_cells_per_row = mask.float().sum(dim=-1).mean().item()

    torch.manual_seed(seed)  # same seed, same init, same batch order across all three patterns
    model = CharGPT(vocab_size, BLOCK_SIZE, D_MODEL, D_FF).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    curve = []
    start = time.perf_counter()
    for step in range(NUM_STEPS):
        x, y = get_batch(train_data, BATCH_SIZE, device)
        logits = model(x, mask)
        loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 10:
            # so a bad NUM_STEPS guess is visible before all 9 runs commit to it
            projected_min = (time.perf_counter() - start) / 10 * NUM_STEPS / 60
            print(f"  [{pattern} seed={seed}] projected total time: {projected_min:.1f} min")

        if step % EVAL_INTERVAL == 0 or step == NUM_STEPS - 1:
            val_loss = estimate_loss(model, val_data, mask, device, loss_fn, EVAL_BATCHES)
            curve.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss})
            print(f"{pattern} seed={seed} step={step:5d} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

    return model, curve, mean_cells_per_row


def save_results(rows):
    os.makedirs("results", exist_ok=True)
    fieldnames = ["pattern", "seed", "window_or_block_params", "mean_cells_per_row",
                  "final_train_loss", "final_val_loss", "steps"]
    with open("results/quality.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# one line per (pattern, seed), same color per pattern, so seed spread is visible
def plot_curves(all_curves):
    colors = {"dense": "tab:blue", "sliding_window": "tab:orange", "block_sparse": "tab:green"}
    fig, ax = plt.subplots(figsize=(8, 6))
    labeled = set()
    for (pattern, seed), curve in all_curves.items():
        label = pattern if pattern not in labeled else None
        labeled.add(pattern)
        ax.plot([c["step"] for c in curve], [c["val_loss"] for c in curve],
                color=colors[pattern], alpha=0.6, label=label)

    ax.set_xlabel("step")
    ax.set_ylabel("validation loss")
    ax.set_title("val loss vs step, one line per seed")
    ax.legend()
    fig.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    fig.savefig("results/plots/quality_loss_curves.png")


# short autoregressive sample from a trained model, for the "is this Shakespeare-ish" check
def generate(model, encode, decode, device, length=200):
    model.eval()
    x = torch.tensor([encode("\n")], device=device)
    with torch.no_grad():
        for _ in range(length):
            mask = causal_mask(x.shape[1]).to(device)
            logits = model(x, mask)
            probs = torch.softmax(logits[0, -1], dim=-1)
            next_id = torch.multinomial(probs, 1)
            x = torch.cat([x, next_id.unsqueeze(0)], dim=1)
    model.train()
    return decode(x[0].tolist())


def main():
    info = hardware_info()
    print("hardware:", info)

    text = download_data()
    chars, encode, decode = build_tokenizer(text)
    vocab_size = len(chars)
    print("vocab_size:", vocab_size, "expected step-0 loss ~ ln(vocab_size) =", torch.log(torch.tensor(float(vocab_size))).item())

    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(TRAIN_FRACTION * len(data))
    train_data, val_data = data[:split], data[split:]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    all_curves = {}
    dense_model_for_sample = None
    expected_param_count = None

    for pattern in ["dense", "sliding_window", "block_sparse"]:
        for seed in SEEDS:
            model, curve, mean_cells = run(pattern, seed, train_data, val_data, vocab_size, device)

            param_count = sum(p.numel() for p in model.parameters())
            if expected_param_count is None:
                expected_param_count = param_count
            assert param_count == expected_param_count, "model size differs between runs, hyperparameters are not identical"

            rows.append({
                "pattern": pattern, "seed": seed,
                "window_or_block_params": pattern_params_str(pattern),
                "mean_cells_per_row": mean_cells,
                "final_train_loss": curve[-1]["train_loss"],
                "final_val_loss": curve[-1]["val_loss"],
                "steps": NUM_STEPS,
            })
            all_curves[(pattern, seed)] = curve

            if pattern == "dense" and seed == SEEDS[0]:
                dense_model_for_sample = model

    save_results(rows)
    plot_curves(all_curves)

    dense_losses = [r["final_val_loss"] for r in rows if r["pattern"] == "dense"]
    sparse_losses = [r["final_val_loss"] for r in rows if r["pattern"] != "dense"]
    if min(dense_losses) > min(sparse_losses):
        print("WARNING: a sparse pattern beat dense - dense has strictly more information, check for a bug")

    print("\nsample from trained dense model:")
    print(generate(dense_model_for_sample, encode, decode, device))


if __name__ == "__main__":
    main()
