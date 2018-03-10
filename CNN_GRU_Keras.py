import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, Dropout, MaxPooling1D, Conv1D
from keras.models import Model
from keras.preprocessing import text, sequence

MAX_FEATURES = 60000
MAX_TEXT_LENGTH = 1000
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.1
CLASSES_LIST = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# MODEL_FILE_PATH = 'log/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
MODEL_FILE_PATH = 'log/CNN_GRU_Keras_weights_best.hdf5'
DATA_SUBMISSION_FILE_PATH = 'data/sample_submission.csv'


def get_train_test(train, test):
    train_raw_text = train["comment_text"].fillna("MISSINGVALUE").values
    test_raw_text = test["comment_text"].fillna("MISSINGVALUE").values
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)

    tokenizer.fit_on_texts(list(train_raw_text))
    train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
    test_tokenized = tokenizer.texts_to_sequences(test_raw_text)
    return sequence.pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH), \
           sequence.pad_sequences(test_tokenized, maxlen=MAX_TEXT_LENGTH)


def get_y(train):
    return train[CLASSES_LIST].values


def get_model():
    embed_size = 300
    inp = Input(shape=(MAX_TEXT_LENGTH,))
    model = Embedding(MAX_FEATURES, embed_size)(inp)
    model = Dropout(0.2)(model)
    model = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(model)
    model = MaxPooling1D(pool_size=2)(model)
    model = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(model)
    model = MaxPooling1D(pool_size=2)(model)
    model = GRU(128)(model)
    model = Dense(64, activation="relu")(model)
    model = Dense(32, activation="relu")(model)
    model = Dense(16, activation="relu")(model)
    model = Dense(6, activation="sigmoid")(model)
    model = Model(inputs=inp, outputs=model)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def train_fit_predict(model, x_train, x_test, y):
    checkpoint = ModelCheckpoint(MODEL_FILE_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                                 period=1)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    callbacks_list = [checkpoint, early]
    model.fit(x_train, y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS, verbose=1,
              validation_split=VALIDATION_SPLIT,
              callbacks=callbacks_list)

    model.load_weights(MODEL_FILE_PATH)
    return model.predict(x_test)


def gen_submit(y_test):
    sample_submission = pd.read_csv(DATA_SUBMISSION_FILE_PATH)[:len(y_test)]
    sample_submission[CLASSES_LIST] = y_test
    sample_submission.to_csv("log/CNN_GRU_Keras_Baseline.csv", index=False)


def main():
    # train = pd.read_csv("data/small_train.csv")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    # test = pd.read_csv("data/small_test.csv")

    x_train, x_test = get_train_test(train, test)
    print len(x_train), len(x_test)

    y = get_y(train)
    y_test = train_fit_predict(get_model(), x_train, x_test, y)

    print "Generate Submission"
    gen_submit(y_test)


if __name__ == "__main__":
    main()
