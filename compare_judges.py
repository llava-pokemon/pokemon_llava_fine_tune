import json
import argparse
from collections import Counter
import numpy as np
from math import sqrt
import os  # for creating plots directory

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


# -----------------------------
# Utility functions
# -----------------------------

def load_judgements(path):
    winners = []
    A_scores = []
    B_scores = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            winners.append(ex["winner"])
            A_scores.append(ex["A_score"])
            B_scores.append(ex["B_score"])

    return np.array(winners), np.array(A_scores), np.array(B_scores)


def score_gap(A_scores, B_scores):
    return B_scores - A_scores


def confidence_margin(A_scores, B_scores):
    return np.abs(B_scores - A_scores)


# -----------------------------
# Metrics per judge
# -----------------------------

def summarize_single_judge(name, winners, A_scores, B_scores):
    n = len(winners)
    cnt = Counter(winners)

    A_rate = cnt.get("A", 0) / n
    B_rate = cnt.get("B", 0) / n
    Tie_rate = cnt.get("Tie", 0) / n

    gaps = score_gap(A_scores, B_scores)
    margins = confidence_margin(A_scores, B_scores)

    print(f"\n==============================")
    print(f"{name} - Summary")
    print(f"==============================")
    print(f"Total examples: {n}")
    print(f"A win-rate: {A_rate:.3f}")
    print(f"B win-rate: {B_rate:.3f}")
    print(f"Tie rate:   {Tie_rate:.3f}")
    print()
    print(f"Avg A_score: {A_scores.mean():.2f}")
    print(f"Avg B_score: {B_scores.mean():.2f}")
    print()
    print(f"Gap (B - A): mean={gaps.mean():.3f}, median={np.median(gaps):.3f}, std={gaps.std():.3f}")
    print(f"Confidence margin: mean={margins.mean():.3f}, median={np.median(margins):.3f}, std={margins.std():.3f}")

    return gaps, margins


# -----------------------------
# Cohen's Kappa
# -----------------------------

def cohens_kappa(w1, w2):
    mapping = {"A": 0, "B": 1, "Tie": 2}
    x = np.array([mapping[w] for w in w1])
    y = np.array([mapping[w] for w in w2])

    n = len(x)
    assert n == len(y)

    # Observed agreement
    Po = np.sum(x == y) / n

    # Expected agreement
    px = np.array([np.sum(x == i) / n for i in range(3)])
    py = np.array([np.sum(y == i) / n for i in range(3)])
    Pe = np.sum(px * py)

    if Pe == 1:
        return 1.0

    kappa = (Po - Pe) / (1 - Pe)
    return kappa


# -----------------------------
# Cross-judge agreement metrics
# -----------------------------

def analyze_agreement(w1, w2, name1, name2):
    n = len(w1)
    assert n == len(w2)

    same = (w1 == w2)
    agreement = np.sum(same)
    overall = agreement / n

    agree_A = np.sum((w1 == "A") & (w2 == "A"))
    agree_B = np.sum((w1 == "B") & (w2 == "B"))
    agree_T = np.sum((w1 == "Tie") & (w2 == "Tie"))

    print("\n==============================")
    print("Agreement Between Judges")
    print("==============================")
    print(f"Overall agreement: {overall:.3f}")
    print(f"Agree on A: {agree_A}")
    print(f"Agree on B: {agree_B}")
    print(f"Agree on Tie: {agree_T}")

    # Disagreement breakdown
    diff = [(a, b) for a, b in zip(w1, w2) if a != b]
    cnt = Counter(diff)

    if cnt:
        print("\nDisagreement patterns (Judge1 -> Judge2):")
        for (x, y), c in cnt.items():
            print(f"  {x} -> {y} : {c}")

    # Cohen’s Kappa
    kappa = cohens_kappa(w1, w2)
    print(f"\nCohen's Kappa: {kappa:.3f}  (values near 0 = random, negative = systematic disagreement)")

    return overall, agree_A, agree_B, agree_T, cnt


# -----------------------------
# Score correlations
# -----------------------------

def analyze_correlations(A1, B1, A2, B2):
    print("\n==============================")
    print("Score Correlations")
    print("==============================")

    def safe_corr(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return np.corrcoef(x, y)[0, 1]

    print(f"A_score correlation:  {safe_corr(A1, A2):.3f}")
    print(f"B_score correlation:  {safe_corr(B1, B2):.3f}")
    print(f"Gap correlation:      {safe_corr(B1 - A1, B2 - A2):.3f}")


# -----------------------------
# Save plots to disk
# -----------------------------

def save_plots(m_w, m_A, m_B, q_w, q_A, q_B, m_gap, q_gap, m_margin, q_margin):
    if not MATPLOTLIB:
        print("\nMatplotlib not installed. Skipping plots.")
        return

    os.makedirs("plots", exist_ok=True)

    # 1. Win-rate bar chart
    plt.figure(figsize=(6, 4))
    labels = ["A", "B", "Tie"]
    m_counts = [np.sum(m_w == lab) for lab in labels]
    q_counts = [np.sum(q_w == lab) for lab in labels]
    m_total = len(m_w)
    q_total = len(q_w)
    m_rates = [c / m_total for c in m_counts]
    q_rates = [c / q_total for c in q_counts]

    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, m_rates, width, label="Mistral")
    plt.bar(x + width/2, q_rates, width, label="Qwen")
    plt.xticks(x, labels)
    plt.ylabel("Win rate")
    plt.title("Win-rate by Judge and Winner")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/win_rates.png")
    plt.close()

    # 2. Score histograms per judge (Mistral)
    plt.figure(figsize=(6, 4))
    plt.hist(m_A, bins=10, alpha=0.6, label="A_score", range=(1, 10))
    plt.hist(m_B, bins=10, alpha=0.6, label="B_score", range=(1, 10))
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Mistral Judge - Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/mistral_scores.png")
    plt.close()

    # 3. Score histograms per judge (Qwen)
    plt.figure(figsize=(6, 4))
    plt.hist(q_A, bins=10, alpha=0.6, label="A_score", range=(1, 10))
    plt.hist(q_B, bins=10, alpha=0.6, label="B_score", range=(1, 10))
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Qwen Judge - Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/qwen_scores.png")
    plt.close()

    # 4. Confidence margin histogram (both judges)
    plt.figure(figsize=(6, 4))
    plt.hist(m_margin, bins=15, alpha=0.6, label="Mistral margin")
    plt.hist(q_margin, bins=15, alpha=0.6, label="Qwen margin")
    plt.xlabel("|B_score - A_score|")
    plt.ylabel("Count")
    plt.title("Confidence Margin Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/margin_histogram.png")
    plt.close()

    # 5. Gap histogram (both judges)
    plt.figure(figsize=(6, 4))
    plt.hist(m_gap, bins=40, alpha=0.6, label="Mistral gap (B-A)")
    plt.hist(q_gap, bins=40, alpha=0.6, label="Qwen gap (B-A)")
    plt.legend()
    plt.title("Score Gap Distribution (B - A)")
    plt.xlabel("B - A score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/gap_histogram.png")
    plt.close()

    # 6. Gap correlation scatter
    plt.figure(figsize=(6, 4))
    plt.scatter(m_gap, q_gap, alpha=0.3)
    plt.xlabel("Mistral Gap (B-A)")
    plt.ylabel("Qwen Gap (B-A)")
    plt.title("Gap Correlation Scatter")
    plt.tight_layout()
    plt.savefig("plots/gap_scatter.png")
    plt.close()

    print("\nSaved plots in 'plots/' directory:")
    print("  plots/win_rates.png")
    print("  plots/mistral_scores.png")
    print("  plots/qwen_scores.png")
    print("  plots/margin_histogram.png")
    print("  plots/gap_histogram.png")
    print("  plots/gap_scatter.png")


# -----------------------------
# Main script
# -----------------------------

def main(m_path, q_path):

    # Load
    m_w, m_A, m_B = load_judgements(m_path)
    q_w, q_A, q_B = load_judgements(q_path)

    # Summaries
    m_gap, m_margin = summarize_single_judge("Mistral Judge", m_w, m_A, m_B)
    q_gap, q_margin = summarize_single_judge("Qwen Judge", q_w, q_A, q_B)

    # Agreement
    analyze_agreement(m_w, q_w, "Mistral", "Qwen")

    # Correlations
    analyze_correlations(m_A, m_B, q_A, q_B)

    # Save plots instead of showing
    save_plots(m_w, m_A, m_B, q_w, q_A, q_B, m_gap, q_gap, m_margin, q_margin)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model1", type=str, required=True)
    ap.add_argument("--model2", type=str, required=True)
    args = ap.parse_args()

    main(args.model1, args.model2)