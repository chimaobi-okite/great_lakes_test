

with open("wsj1-18.training", "r") as f:
    sentences = []
    pos_labels = []
    for line in f:
        sentence = []
        label = []
        for i, word in enumerate(line.split()):
            sentence.append(word) if i % 2 == 0 else label.append(word)
        sentences.append(sentence)
        pos_labels.append(label)
        break
    print(sentences, pos_labels)