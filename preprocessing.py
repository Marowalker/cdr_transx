import random
from data_utils import load_vocab
import pickle
from sklearn.utils import shuffle


def create_triple(infile, ent_vocab, rel_vocab, mode='cdr'):
    if mode == 'chemprot':
        file = open(infile, encoding='utf-8')
    else:
        file = open(infile)
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    ents = list(ent_vocab.keys())

    heads = []
    tails = []
    rels = []
    heads_neg = []
    tails_neg = []

    for idx, line in enumerate(lines):
        if idx != 0:
            triple = line.split()
            if len(triple) == 3:
                head, tail, rel = triple[0], triple[1], triple[2]
                head_id = ent_vocab[head]
                tail_id = ent_vocab[tail]
                rel_id = rel_vocab[rel]
                threshold = random.random()
                if threshold < 0.5:
                    head_neg = random.choice(ents)
                    head_neg_id = ent_vocab[head_neg]
                    tail_neg_id = tail_id
                else:
                    tail_neg = random.choice(ents)
                    tail_neg_id = ent_vocab[tail_neg]
                    head_neg_id = head_id
                heads.append(head_id)
                tails.append(tail_id)
                rels.append(rel_id)
                heads_neg.append(head_neg_id)
                tails_neg.append(tail_neg_id)

    return heads, tails, rels, heads_neg, tails_neg


def get_sequences(ent_file, rel_file, triple_file, outfile_train, outfile_dev):
    ent_vocab = load_vocab(ent_file)
    rel_vocab = load_vocab(rel_file)

    head, tail, rel, head_neg, tail_neg = create_triple(triple_file, ent_vocab, rel_vocab, mode='chemprot')

    head_shuffled, tail_shuffled, rel_shuffled, head_neg_shuffled, tail_neg_shuffled = shuffle(head, tail, rel,
                                                                                               head_neg, tail_neg)

    ratio = 0.8
    n_sample = int(len(head) * ratio)

    sequence_dict_train = {
        'head': head_shuffled[:n_sample],
        'tail': tail_shuffled[:n_sample],
        'rel': rel_shuffled[:n_sample],
        'head_neg': head_neg_shuffled[:n_sample],
        'tail_neg': tail_neg_shuffled[:n_sample]
    }
    sequence_dict_dev = {
        'head': head_shuffled[n_sample:],
        'tail': tail_shuffled[n_sample:],
        'rel': rel_shuffled[n_sample:],
        'head_neg': head_neg_shuffled[n_sample:],
        'tail_neg': tail_neg_shuffled[n_sample:]
    }

    return sequence_dict_train, sequence_dict_dev


def load_pickle(train_file, dev_file):
    with open(train_file, 'rb') as f:
        sequence_dict_train = pickle.load(f)

    with open(dev_file, 'rb') as f:
        sequence_dict_dev = pickle.load(f)

    return sequence_dict_train, sequence_dict_dev
