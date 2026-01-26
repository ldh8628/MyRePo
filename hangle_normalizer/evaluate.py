# evaluate.py
import argparse
import numpy as np

def levenshtein(a, b):
    dp = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[-1][-1]


def cer(ref, hyp):
    if len(ref) == 0:
        return 0.0
    return levenshtein(ref, hyp) / len(ref)


def wer(ref, hyp):
    ref_w = ref.split()
    hyp_w = hyp.split()
    if len(ref_w) == 0:
        return 0.0
    return levenshtein(ref_w, hyp_w) / len(ref_w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.ground, encoding="utf-8") as f:
        refs = [l.strip() for l in f.readlines()]
    with open(args.output, encoding="utf-8") as f:
        hyps = [l.strip() for l in f.readlines()]

    exact = 0
    cer_list = []
    wer_list = []

    for r, h in zip(refs, hyps):
        if r == h:
            exact += 1
        cer_list.append(cer(r.replace(" ", ""), h.replace(" ", "")))
        wer_list.append(wer(r, h))

    print("===== Evaluation Summary =====")
    print(f"Num samples         : {len(refs)}")
    print(f"Exact match rate    : {exact / len(refs):.3f}")
    print(f"Avg CER             : {np.mean(cer_list):.3f}")
    print(f"Median CER          : {np.median(cer_list):.3f}")
    print(f"Avg WER             : {np.mean(wer_list):.3f}")
    print(f"Median WER          : {np.median(wer_list):.3f}")
    print(f"Max CER             : {max(cer_list):.3f}")
    print(f"Max WER             : {max(wer_list):.3f}")


if __name__ == "__main__":
    main()
