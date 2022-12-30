import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def buildModel(layers_neurons_num, act_functions, loss_fun, optimizer, dropout_val, min_weight_val, max_weight_val):
    if not len(layers_neurons_num) == len(act_functions) == (len(dropout_val) + 1):
        raise ValueError("Error: One of the List sizes is incompatible with the number of layers.")

    initializer = tf.keras.initializers.RandomUniform(minval=min_weight_val, maxval=max_weight_val, seed=5)
    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.layers.Dense(layers_neurons_num[0], input_dim=layers_neurons_num[0],
                                    activation=act_functions[0], kernel_initializer=initializer))
    model.add(tf.keras.layers.Dropout(dropout_val[0]))

    for index, layerNeuronsNum in enumerate(layers_neurons_num[1:len(layers_neurons_num) - 1]):
        model.add(
            tf.keras.layers.Dense(layerNeuronsNum, activation=act_functions[index], kernel_initializer=initializer))
        model.add(tf.keras.layers.Dropout(dropout_val[index]))

    # Output layer, does not need dropout
    model.add(tf.keras.layers.Dense(layers_neurons_num[len(layers_neurons_num)-1],
                                    activation=act_functions[len(layers_neurons_num)-1], kernel_initializer=initializer))

    model.compile(loss=loss_fun, optimizer=optimizer, metrics=['accuracy'])
    return model


def looTrainNeuralNetwork(dataset, epochs):
    X = dataset[:, 0:4]
    y = dataset[:, 4]

    # Divido il dataset in training e test set, dopodiché estrapolo il validation set dal test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=22)

    # Creo il modello di rete neurale
    mod = buildModel(layers_neurons_num=[4, 4, 1], act_functions=['linear', 'sigmoid', 'sigmoid'],
                     loss_fun='binary_crossentropy', optimizer='adam', dropout_val=[0.2, 0.2],
                     min_weight_val=-0.5, max_weight_val=0.5)

    # Definisco un criterio di early stopping per la fase di addestramento
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    # Addestro il modello di rete neurale
    history = mod.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=epochs,
                      batch_size=128, verbose=1, callbacks=[es])

    # Leave One Out
    _, accuracy = mod.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))

    # Plot accuratezza
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Plot funzione di errore
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    # mod.save("modelloReteNeurale-LeaveOneOut.h5")


def kfoldTrainNeuralNetwork(dataset, epochs):
    X = dataset[:, 0:4]
    y = dataset[:, 4]

    # Definisco le liste che conterranno i valori di accuratezza e di loss per ciascuna fold
    acc_per_fold = []
    loss_per_fold = []

    kfold = KFold(n_splits=10, shuffle=True)
    fold_no = 1

    for train, test in kfold.split(X, y):
        # Creazione modello
        model = buildModel(layers_neurons_num=[4, 4, 1], act_functions=['linear', 'sigmoid', 'sigmoid'],
                           loss_fun='binary_crossentropy', optimizer='adam', dropout_val=[0.2, 0.2],
                           min_weight_val=-0.5, max_weight_val=0.5)

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Addestramento del modello
        history = model.fit(X[train], y[train], batch_size=64, epochs=epochs, verbose=1)

        # Generazione delle metriche
        scores = model.evaluate(X[test], y[test], verbose=0)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Aggiornamento indice fold corrente
        fold_no = fold_no + 1

    # == Stampa valori medi per ciascuna fold ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')
