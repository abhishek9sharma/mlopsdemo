import os
import pandas as pd

from mlcore.utils import get_currtime_str
from mlcore.data_helper import save_data, load_data_from_file_path

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras import regularizers


def get_ordered_preds_seq(test_seq, model_dict, ix2word=None, return_scores=False):
    """This method reaturns a list of merchants/dictionary of merchants and scores
    that are most similar to a user vector in the learned Sequential DL model

    :param test_seq: Sequence of merchants clicked in the past
    :param model_dict: parameters releated to the sequential deep learning model
    :param ix2word: token_ix to merchant_id
    :param return_scores: a boolean flag which specifies if a list of merchants needs
        to be returned or a dictionary of merchants with the similarty scores
    :return:  a list or dictionary of next merchants which should be recommended
    """

    train_tokenizer = model_dict["train_tokenizer"]
    max_sequence_len = model_dict["max_sequence_len"]
    model = model_dict["obj"]

    tokenized_test_seq = train_tokenizer.texts_to_sequences([test_seq])[0]
    tokenized_test_seq = pad_tokenized_seq(
        [tokenized_test_seq], padlen=max_sequence_len - 1
    )

    probs = model.predict(tokenized_test_seq, verbose=0)
    best_n = (-probs).argsort()

    # predicted_top_class = np.argmax(probs,axis=1)
    # for word, index in train_tokenizer.word_index.items():
    #     if index == predicted:
    #         output_word = word
    #         break
    if ix2word is None:
        word2ix = model_dict["train_tokenizer"].word_index
        ix2word = {w: i for i, w in word2ix.items()}

    pred_merchants = {}
    for ix in best_n[0]:
        if ix in ix2word:
            merchant_id = ix2word[ix]
            pred_merchants[merchant_id] = probs[0][ix]

    if return_scores:
        return pred_merchants

    return list(pred_merchants.keys())


def extract_merchants_clicked_seq(uid, click_data):
    """Extracts sequence of merchants clicked by a user

    :param uid: User id of the user
    :param click_data: Data Frame containing click data of a user
    :return: A sequence of merchants clicked by the user
    """

    user_log = click_data[click_data.user_id == uid]
    user_log_sorted = user_log.sort_values(by="created_at")
    clicked_seq = user_log_sorted["merchant_id"].tolist()
    full_hist = ["merch" + str(i) for i in clicked_seq]
    uniq_hist = []
    for ix, m in enumerate(full_hist):
        if ix == 0:
            uniq_hist.append(m)
        else:
            if uniq_hist[-1] != m:
                uniq_hist.append(m)

    if len(full_hist) > 30:
        trimmed_hist = uniq_hist[-30:]
    else:
        trimmed_hist = uniq_hist
    return trimmed_hist


def extract_user_merchant_hist(click_data):
    """Extracts sequence of merchants clicked by a users in a dataframe

    :param click_data: Merchants clicked by a user
    return: A data frame containing users and the sequence of merchants clicked by them
    """

    uids = set(click_data.user_id.tolist())
    user_merchant_hist = pd.DataFrame(list(uids), columns=["uids"])
    user_merchant_hist["merchant_click_seq"] = user_merchant_hist.uids.apply(
        extract_merchants_clicked_seq, click_data=click_data
    )

    return user_merchant_hist


def text_to_tokenized_seq(texts):
    """
        Uses the built in keras tokenizer to transform a given list of string tokens
        to sequence of numbers and also returns the mapping from numbers to original tokens

    :param texts: a list of character strings that need to be transformed
    :return: a list of number tokens and the tokenizer used
    """

    seq_tokenizer = Tokenizer()

    seq_tokenizer.fit_on_texts(texts)
    tokenized_seq = seq_tokenizer.texts_to_sequences(texts)

    return tokenized_seq, seq_tokenizer


def pad_tokenized_seq(tokenized_seq, padlen=None):
    """
        Adjusts the given input of tokens to a given length so that all the token sequences are transformed to vectors of
        same length
    :param tokenized_seq: a sequence of number tokens
    :param padlen: the length to which thise sequence of tokens needs to be adjusted to
    :return:
    """

    if padlen is None:
        padlen = max([len(tokenized_seq) for tokenized_seq in tokenized_seq])
    return pad_sequences(tokenized_seq, maxlen=padlen, padding="pre")


def get_input_seqs(train_X_tokenized):

    input_sequences = []
    for token_list in train_X_tokenized:
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[: i + 1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = pad_tokenized_seq(input_sequences, max_sequence_len)
    return input_sequences, max_sequence_len


def create_training_input(seq_data_list):

    train_X_tokenized, train_tokenizer = text_to_tokenized_seq(seq_data_list)
    total_words = len(train_tokenizer.word_index) + 1
    input_sequences, max_sequence_len = get_input_seqs(train_X_tokenized)

    predictors, labels = input_sequences[:, :-1], input_sequences[:, -1]
    labels = ku.to_categorical(labels, num_classes=total_words)

    return train_tokenizer, total_words, predictors, labels, max_sequence_len


def configure_model_lstm(MAX_NB_WORDS, MAX_SEQ_LEN, EMBEDDING_DIM=200):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQ_LEN - 1))
    model.add(LSTM(100, activation="tanh", return_sequences=True, dropout=0.2))
    model.add(LSTM(200, activation="tanh", return_sequences=False, dropout=0.2))
    # model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dense(total_words, activation='softmax'))
    return model


def configure_model(
    total_words, max_sequence_len, EMBEDDING_DIM=200, model_type="LSTM"
):

    if model_type == "LSTM":
        model = configure_model_lstm(total_words, max_sequence_len, EMBEDDING_DIM=200)

    # model.add(
    #     Dense(
    #         total_words / 2, activation="relu", kernel_regularizer=regularizers.l2(0.01)
    #     )
    # )
    model.add(Dense(total_words, activation="Softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.summary()
    return model


def fit_configured_model(
    model,
    predictors,
    labels,
    epochs=1,
    verbose=1,
    batch_size=128,
    early_stop=False,
    enable_callbacks=False,
):

    curr_ts = get_currtime_str()
    if early_stop:
        early_stopping = EarlyStopping(
            monitor="val_loss", min_delta=0.00001, patience=20
        )
        model_name = os.path.join("../models", "early_" + curr_ts + "_best__model.h5")
    else:
        model_name = os.path.join(
            "../models", "notearly_" + curr_ts + "_best__model.h5"
        )

    model_checkpoint = ModelCheckpoint(
        model_name,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )

    train_X, val_X, train_Y, val_Y = train_test_split(
        predictors,
        labels,
        test_size=0.15,
        random_state=27,
    )

    # train the model
    if enable_callbacks:
        if early_stop:
            history = model.fit(
                x=train_X,
                y=train_Y,
                epochs=epochs,
                shuffle=True,
                batch_size=batch_size,
                validation_data=(val_X, val_Y),
                callbacks=[early_stopping, model_checkpoint],
            )
        else:
            history = model.fit(
                x=train_X,
                y=train_Y,
                epochs=epochs,
                shuffle=True,
                batch_size=batch_size,
                validation_data=(val_X, val_Y),
                callbacks=[model_checkpoint],
            )
    else:
        history = model.fit(
            x=train_X,
            y=train_Y,
            epochs=epochs,
            shuffle=True,
            batch_size=batch_size,
            validation_data=(val_X, val_Y),
        )
    return model, history


def fit_seq_model_seq_data(seq_data_list, param_dict=None):

    (
        train_tokenizer,
        total_words,
        predictors,
        labels,
        max_sequence_len,
    ) = create_training_input(seq_data_list)
    model = configure_model(
        total_words, max_sequence_len, EMBEDDING_DIM=200, model_type="LSTM"
    )

    epochs = 5
    early_stop = False
    if param_dict:
        epochs = param_dict.get("epochs", 10)
        early_stop = param_dict.get("early_stop", False)
        batch_size = param_dict.get("batch_size", 128)

    model, history = fit_configured_model(
        model,
        predictors,
        labels,
        epochs=epochs,
        verbose=1,
        early_stop=early_stop,
        batch_size=batch_size,
    )

    return model, train_tokenizer, total_words, max_sequence_len, history


def fit_sequence_model(
    train_data,
    features,
    param_dict=None,
    logger=None,
    user_merchant_hist_data_path=None,
):
    """Trains a Sequence (e.g RNN ) model given information about users and items clicks

    :param train_data: a pandas dataframe contain click interactions between users and items
    :param features: features which constitute the interaction between user and merchant nodes
    :param param_dict: parameter settings for the sequence model training
    :return model_fitted_params: a dict containing metadata about the trained model
    """
    if user_merchant_hist_data_path:
        try:
            logstr = " loading user hist from {}".format(user_merchant_hist_data_path)
            logger.info(logstr)
            train_data_valid = load_data_from_file_path(
                user_merchant_hist_data_path, logger=logger
            )
        except:
            user_merchant_hist_data_path_trimmed = user_merchant_hist_data_path.replace(
                "../", ""
            )
            logstr = " loading user hist from {}".format(
                user_merchant_hist_data_path_trimmed
            )
            logger.info(logstr)
            train_data_valid = load_data_from_file_path(
                user_merchant_hist_data_path_trimmed, logger=logger
            )
        train_data_valid[
            "merchant_click_seq"
        ] = train_data_valid.merchant_click_seq.apply(lambda x: " ".join(list(x)))
    else:
        train_data_valid = extract_user_merchant_hist(train_data[features])
        curr_ts = get_currtime_str()
        user_merchant_hist_data_path = save_data(
            data=train_data_valid,
            schema_name="user_merchant_hist_" + curr_ts,
            logger=logger,
        )

    logger.info(
        "loaded user_merchant hist  of size:: {}".format(train_data_valid.shape)
    )
    seq_data_list = train_data_valid["merchant_click_seq"].tolist()
    (
        model,
        train_tokenizer,
        total_words,
        max_sequence_len,
        history,
    ) = fit_seq_model_seq_data(seq_data_list, param_dict=param_dict)

    model_fitted_params = {
        "obj": model,
        "train_tokenizer": train_tokenizer,
        "max_sequence_len": max_sequence_len,
        "user_merchant_hist_data_path": user_merchant_hist_data_path,
    }

    return model_fitted_params  # , max_sequence_len#, user_merchant_hist_data_path,train_data_valid
