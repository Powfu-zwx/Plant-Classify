import pickle
import matplotlib.pyplot as plt

# 加载日志数据
def load_log(path):
    with open(path, "rb") as f:
        log = pickle.load(f)
    return log["loss"], log["acc"]

optimizers = ["adam", "adagrad", "momentum"]
colors = ["blue", "green", "red"]

plt.figure(figsize=(12, 5))

# --- Plot Loss ---
plt.subplot(1, 2, 1)
for opt, color in zip(optimizers, colors):
    loss, _ = load_log(f"log_{opt}.pkl")
    plt.plot(loss, label=opt, color=color)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# --- Plot Accuracy ---
plt.subplot(1, 2, 2)
for opt, color in zip(optimizers, colors):
    _, acc = load_log(f"log_{opt}.pkl")
    plt.plot(acc, label=opt, color=color)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("optimizer_comparison.png")
plt.show()
