import constants
from preprocessing import get_sequences, load_pickle
import pickle
from models.transE import TransEModel
import tensorflow as tf


def main_knowledge_base():
    if constants.IS_REBUILD == 1:
        print('Build data...')
        train_dict, val_dict = get_sequences(constants.ALL_WORDS, constants.ALL_DEPENDS, constants.ALL_EDGES,
                                             constants.PICKLE + 'train.pkl', constants.PICKLE + 'dev.pkl')
        with open(constants.PICKLE + 'train.pkl', 'wb') as f:
            pickle.dump(train_dict, f)

        with open(constants.PICKLE + 'val.pkl', 'wb') as f:
            pickle.dump(val_dict, f)
    else:
        print('Load data...')
        with open(constants.PICKLE + 'train.pkl', 'rb') as f:
            train_dict = pickle.load(f)

        with open(constants.PICKLE + 'val.pkl', 'rb') as f:
            val_dict = pickle.load(f)

    print("Train shape: ", len(train_dict['head']))
    print("Validation shape: ", len(val_dict['head']))

    with tf.device('/device:GPU:0'):

        transe = TransEModel(model_path=constants.TRAINED_MODELS, batch_size=64, epochs=constants.EPOCHS,
                             score=constants.SCORE)
        transe.build(train_dict, val_dict)
        transe.train(early_stopping=True, patience=constants.PATIENCE)
        all_emb = transe.load(load_file='data/embeddings/transe_chemprot_word_' + str(constants.INPUT_W2V_DIM) + '.pkl',
                              embed_type='word')
        print(all_emb)
        print(all_emb.shape)


if __name__ == '__main__':
    main_knowledge_base()
