from tensorflow import keras
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from numpy import array, argmax
from keras.utils import to_categorical

plt.style.use('default');
plt.rc('figure', autolayout=True);
plt.rc('axes', labelweight='bold',
       labelsize='large', titleweight='bold', titlesize=18, titlepad=10);
plt.rc('animation', html='html5')

data = pd.read_csv("./titanic_set/train.csv");
test = pd.read_csv("./titanic_set/test.csv");

data.dropna(axis=0, subset=['Survived'], inplace=True);

x = data.copy();
x = pd.get_dummies(data=x, columns=['Sex'], prefix=['s']);
test_x = pd.get_dummies(data=test, columns=['Sex'], prefix=['s']);
rem_feature = ['PassengerId', 'Name', 'Fare', 'Age', 'Ticket', 'Cabin', 'Embarked']

x.drop(axis=1, labels=rem_feature, inplace=True);
test_x.drop(axis=1, labels=rem_feature, inplace=True);
y = data['Survived'];

x_train = x.sample(frac=0.7, random_state=0);
x_valid = x.drop(x_train.index);

y_train = x_train['Survived'];
y_valid = x_valid['Survived'];

x_train.drop(axis=1, labels=['Survived'], inplace=True);
x_valid.drop(axis=1, labels=['Survived'], inplace=True);

print(x_train.head());
print(x_valid.head());
print(test);

model = keras.Sequential([
    layers.BatchNormalization(input_shape=[5]),
    layers.Dense(512, activation='relu', input_shape=[5]),
    layers.BatchNormalization(input_shape=[5]),
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(input_shape=[5]),
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(input_shape=[5]),
    layers.Dense(1, activation='sigmoid')
]);

opt = keras.optimizers.SGD(learning_rate=0.005);

model.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
);

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
);

history = model.fit(
    x_train, y_train,
    validation_data=(x_valid, y_valid),
    batch_size=10,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0
);

test_y = model.predict(test_x);
test_y = pd.DataFrame(test_y);
test_y = test_y.round();
test_y.index = test['PassengerId'];
test_y[0] = test_y[0].astype(int);
test_y.rename(columns={0: 'Survived'}, inplace=True);
test_y.to_csv('test_y_norm.csv');

history_df = pd.DataFrame(history.history);
history_df.loc[:, ['loss', 'val_loss']].plot();
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
plt.show()
print(("Minimum Validation Loss: {:0.4f}").format(history_df['val_loss'].min()))
print(("Maximum Validation Accuracy: {:0.4f}").format(history_df['val_binary_accuracy'].max()))