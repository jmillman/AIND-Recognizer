import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    hwords = test_set.get_all_Xlengths()

    # Loop over word ids
    for word_id in hwords:
        best_score = float('-Inf')
        best_word = None
        tmp_probabilities = {}
        sequences, lengths = hwords[word_id]
        # Loop over models see which model has the best score per the word, that model's word is the answer
        for model_word, model in models.items():
            try:
                tmp_score = model.score(sequences, lengths)
            except:
                tmp_score = float('-Inf')
            tmp_probabilities[model_word] = tmp_score
            if tmp_score > best_score:
                best_score = tmp_score
                best_word = model_word

        guesses.append(best_word)
        probabilities.append(tmp_probabilities)

    return probabilities, guesses