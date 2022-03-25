import sys
import json
import numpy as np
from collections import defaultdict

import ipdb


def main():
    log_path = sys.argv[1]
    outputs = json.load(open(log_path))
    
    influences = defaultdict(lambda: [])
    binary = {}
    grades = {}
    for i in outputs:
        target_word, corpus, influence, binary_gt, grade_gt, n_removed = i
        binary[target_word] = int(binary_gt)
        grades[target_word] = float(grade_gt)
        if influence < 100:
            influences[target_word].append(influence)

    
    for k in influences.keys():
        influences[k] = np.mean(influences[k])
        
    
    min_inf = min(influences.values())
    max_inf = max(influences.values())
    for thres in np.linspace(min_inf, max_inf, 10):
        print("==================================================")
        print(f"Threshold: {thres}")
        keys = np.array(list(influences.keys()))
        pred_inf = np.array([influences[k] for k in keys])
        binary_gt = np.array([binary[k] for k in keys]) == 1
        
        correct = (pred_inf > thres) == binary_gt
        acc = np.mean(correct)
        print(f"Accuracy: {acc:.2f}")
        
        print(f"Correct target words   ({correct.sum():2}): {', '.join(keys[correct])}")
        print(f"Incorrect target words ({(~correct).sum():2}): {', '.join(keys[~correct])}")
        
    
if __name__ == "__main__":
    main()
