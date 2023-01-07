import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def buildModel(layers_neurons_num, act_functions, loss_fun, optimizer, dropout_val, min_weight_val, max_weight_val):
    if not len(layers_neurons_num) == len(act_functions) == (len(dropout_val) + 1):
        raise ValueError("Error: One of the List sizes is incompatible with the number of layers.")

    initializer = tf.keras.initializers.RandomUniform(minval=min_weight_val, maxval=max_weight_val, seed=15)
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
    model.add(tf.keras.layers.Dense(layers_neurons_num[len(layers_neurons_num) - 1],
                                    activation=act_functions[len(layers_neurons_num) - 1],
                                    kernel_initializer=initializer))

    model.compile(loss=loss_fun, optimizer=optimizer, metrics=['accuracy'])
    return model


def looTrainNeuralNetwork(dataset, epochs):
    X = dataset[:, 0:4]
    y = dataset[:, 4]

    # Divido il dataset in training e test set, dopodiché estrapolo il validation set dal test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=43)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=47)

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

    fig, axes = plt.subplots(1, 2)

    # Plot accuratezza
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set_title('Model accuracy')
    axes[0].set(xlabel='epoch', ylabel='accuracy')
    axes[0].legend(['train', 'valid'], loc='upper left')
    x0, x1 = axes[0].get_xlim()
    y0, y1 = axes[0].get_ylim()
    axes[0].set_aspect(abs(x1 - x0) / abs(y1 - y0))

    # Plot funzione di errore
    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set_title('Model loss')
    axes[1].set(xlabel='epoch', ylabel='loss')
    axes[1].legend(['train', 'valid'], loc='upper left')
    x0, x1 = axes[1].get_xlim()
    y0, y1 = axes[1].get_ylim()
    axes[1].set_aspect(abs(x1 - x0) / abs(y1 - y0))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

    # plt.show()
    plt.savefig('plots/plot.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.savefig('plots/plot_noTransparency.png', bbox_inches='tight', dpi=200)

    # mod.save("modelloReteNeurale-2-LeaveOneOut.h5")


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
        model.fit(X[train], y[train], batch_size=64, epochs=epochs, verbose=1)

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
