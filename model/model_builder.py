import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class ModelBuilder():
    '''
    Creates a search model, given positive and negative examples

    Args:
        dimension (int): The target output dimension (x and y) of the input images
        save_path (str): The path to save the completed model (including file name)
    
    '''
    def __init__(self, dimension=180, save_path=False):
        
        self.dimension = dimension
        self.save_path = save_path

        # Enable to generate an accuracy report after training
        self.report_accuracy = False

        # Assumes a balanced training set
        self.assume_balanced = True

        self.epochs = 10

    def build(self, positives, negatives):
        ''' Constructs a new (simple) CNN model and trains it using the positive and negative examples.

            Saves the model to the target location, if provided.

            Args:
                positives (numpy array): The positive training examples
                negatives (numpy array): The negative training examples

            Returns:
                model: The newly constructed model
        '''

        if self.assume_balanced and positives.shape[0] != negatives.shape[0]:
            raise Exception("Unexpectedly unbalanced examples")
        
        len = positives.shape[0]

        examples = np.concatenate((positives, negatives), axis=0)
        labels = np.concatenate((np.ones(len),np.zeros(len)), axis=0)

        # Shuffle
        idx = np.random.permutation(len*2)
        x, y = examples[idx], labels[idx]

        model = tf.keras.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.dimension, self.dimension, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=['accuracy',
                        tf.keras.metrics.AUC(),
                        tf.keras.metrics.TruePositives(),
                        tf.keras.metrics.TrueNegatives(),
                        tf.keras.metrics.FalsePositives(),
                        tf.keras.metrics.FalseNegatives()
                    ])

        history = model.fit(x, y, epochs=self.epochs, validation_split=0.20)

        if self.save_path:
            model.save(self.save_path)

        if self.report_accuracy:

            train_accuracy = history.history['accuracy'][-1]
            val_accuracy = history.history['val_accuracy'][-1]

            val_tp = history.history['val_true_positives'][-1]
            val_fp = history.history['val_false_positives'][-1]
            val_tn = history.history['val_true_negatives'][-1]
            val_fn = history.history['val_false_negatives'][-1]
            val_pos = val_tp + val_fn
            val_neg = val_fp + val_tn
            
            val_fn_rate = val_fn / val_pos
            val_fp_rate = val_fp / val_neg

            print(f'Validation Set Accuracy: {round(val_accuracy*100,3)}%'
                + f' / Training Set Accuracy: {round(train_accuracy*100,3)}%')
            print(f'Validation Set False Positive Rate: {round(val_fp_rate*100,3)}%'
                + f', Validation Set False Negative Rate: {round(val_fn_rate*100,3)}%')

            print('Important Assumptions:')
            print(' * Your Supplied Character Does Not Exist in Sample (Negative) Area')
            print(' * The Search Area Contains Similar Road Geometry to Sample (Negative) Area')
            print(' * Your Supplied Character Exists in Target Search Area')

            if self.assume_balanced:
                print('Note that training was performed on a simple balanced dataset'
                    + 'and the search problem is likely inbalanced, depending on your supplied character.')
            else:
                print('(!) Warning: Accuracy metric used without balanced assumption. '
                    + 'Consider removing accuracy measure.')

            if val_fp_rate > 0.005:
                print(f'(!) Warning: A high false positive rate was observed. Subject to the assumptions above, '
                    + f'this model may produce a significant amount of false positives. At a 0.5 threshold, ' 
                    + f'for every 1,000 possible road shapes evaluated {round(val_fp_rate*1000)} will ' 
                    + f'be incorrectly identified as matching. Note that millions of road shapes may be considered.')

            if val_fn_rate > 0.005:
                print(f'(!) Warning: A high false negative rate was observed. Subject to the assumptions above, '
                    + f'this model may produce a significant amount of false negatives. At a 0.5 threshold, ' 
                    + f'for every 1,000 correctly matching road shapes evaluated {round(val_fn_rate*1000)} will ' 
                    + f'be incorrectly discarded as not relevant.')

            if (positives.shape[0] < 1000):
                print('(!) Warning: Low training data volume supplied. Consider increasing the volume of '
                    + 'training data to at least 1000 examples of each class.')

        return model