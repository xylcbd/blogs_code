#coding: utf-8
import numpy as np
import sys

def min_list(lists):
    assert len(lists) >= 1
    mins = lists[0][:]
    for i in range(1, len(lists)):
        row = lists[i]
        for j in range(len(mins)):
            mins[j] = min(mins[j], row[j])
    return mins

def max_list(lists):
    assert len(lists) >= 1
    maxs = lists[0][:]
    for i in range(1, len(lists)):
        row = lists[i]
        for j in range(len(maxs)):
            maxs[j] = max(maxs[j], row[j])
    return maxs

def load_dataset(file_path):
    lines = open(file_path).readlines()
    inputs = []
    outputs = []
    min_params = []
    max_params = []
    for line in lines:
        line = line.strip()
        parts = line.split('\t')
        params = parts[:-1]
        label = parts[-1]
        params = [float(param) for param in params]
        label = int(label)
        inputs.append(params)
        outputs.append(label)
        if len(min_params) == 0:            
            min_params = params[:]
            max_params = params[:]
        else:
            min_params = min_list([min_params, params])
            max_params = max_list([max_params, params])
    #normalize
    norm_inputs = []
    for params in inputs:
        norm_params = []
        for i in range(len(params)):
            norm_params.append((params[i]-min_params[i]) / max((max_params[i]-min_params[i]), 1e-6))
        norm_inputs.append(norm_params)
    inputs = norm_inputs

    #split to train & test
    train_rate = 0.7
    train_cnt = int(train_rate * len(inputs))
    train_inputs = inputs[:train_cnt]
    train_outputs = outputs[:train_cnt]
    test_inputs = inputs[train_cnt:]
    test_outputs = outputs[train_cnt:]
    return train_inputs, train_outputs, test_inputs, test_outputs

def distance(lhs, rhs):
    dist = 0.0
    for i in range(len(lhs)):
        lx = lhs[i]
        rx = rhs[i]
        dist += (lx-rx)*(lx-rx)
    return dist

def classify(train_inputs, train_outputs, x, K):
    scores = []
    for train_x, train_y in zip(train_inputs, train_outputs):        
        dist = distance(x, train_x)
        score = -1.0 * dist
        scores.append((score, train_y))
    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    labels = {}
    for i in range(min(K, len(scores))):
        label = scores[i][1]
        labels.setdefault(label, 0)
        labels[label] += 1
    best_label = -1
    best_cnt = 0
    for label, count in labels.items():
        if best_cnt <= count:
            best_label = label
    return best_label

def test(train_inputs, train_outputs, test_inputs, test_outputs, K):
    total = 0
    correct = 0
    for x,y in zip(test_inputs, test_outputs):
        pd = classify(train_inputs, train_outputs, x, K)        
        if pd == y:
            correct += 1
        total += 1
    return float(correct)/float(total)

def main():
    train_inputs, train_outputs, test_inputs, test_outputs = load_dataset('dataset.txt')
    K = 5
    accuracy = test(train_inputs, train_outputs, test_inputs, test_outputs, K)
    print('KNN accuracy: %.4f' % (accuracy*100.0))

if __name__ == '__main__':
    main()
