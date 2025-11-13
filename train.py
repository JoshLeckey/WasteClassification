import tensorflow as tf
import random
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8 # Fast, but enough for convergence
DATA_DIR = './garbage-dataset' #Change if needed
LABELS_JSON = 'labels.json'

# --- Utility ---

def load_class_names(labels_json):
    import json
    with open(labels_json, 'r') as f:
        class_indices = json.load(f)
    sorted_labels = sorted(class_indices.items(), key=lambda x: x[1])
    return [label for label, idx in sorted_labels]

#--- Model

def build_hybrid_model(num_classes, hybrid=True):
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    mnet = tf.keras.applications.MobileNetV3Large(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg')
    mnet.trainable = False
    x1 = mnet(inputs, training=False)

    if hybrid:
        enet = tf.keras.applications.EfficientNetB0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg')
        enet.trainable = False
        x2 = enet(inputs, training=False)
        x = tf.keras.layers.Concatenate()([x1, x2])
    else:
        x = x1
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)

#--- Data Augmentation

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
    rotation_range=10, #less aggressive
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
    validation_split=0.2)

if __name__ == '__main__':
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    #Print class distribution
    from collections import Counter
    class_names = load_class_names(LABELS_JSON)
    num_classes = len(class_names)

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True, seed=42)

    class_counts = Counter(train_gen.classes)
    print('Class distribution:')
    for idx, name in enumerate(class_names):
        print(f'{name:12} : {class_counts.get(idx,0)}')

    val_gen = val_datagen.flow_from_directory(
        DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False, seed=42)

    print('\ntrain_gen.class_indices:')
    print(train_gen.class_indices)
    print('\nlabels.txt contents:')
    with open('labels.txt', 'r') as f:
        for i, line in enumerate(f):
            print(f'{i}: {line.strip()}')

    # Print class distribution in training set (fixed)
    print("\nClass distribution in training set:")
    from collections import Counter
    class_counts = Counter(train_gen.classes)
    idx_to_label = {v: k for k, v in train_gen.class_indices.items()}
    for idx in range(len(train_gen.class_indices)):
        label = idx_to_label[idx]
        count = class_counts.get(idx, 0)
        print(f"{label}: {count}")

    # Visualize a batch of training images
    def plot_batch(generator, labels, n=8):
        x_batch, y_batch = next(generator)
        plt.figure(figsize=(16, 2))
        for i in range(n):
            plt.subplot(1, n, i+1)
            img = x_batch[i]
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            plt.imshow(img.astype('uint8'))
            label_idx = np.argmax(y_batch[i])
            plt.title(labels[label_idx])
            plt.axis('off')
        plt.suptitle('Sample training batch')
        plt.tight_layout()
        plt.show()

    print("\nVisualizing a batch of training images...")
    plot_batch(train_gen, class_names)

    #--- Compute class weights for imbalance
    y_train = train_gen.classes
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    print('Class weights:', class_weights_dict)

    print('Building hybrid model...')
    model = build_hybrid_model(num_classes, hybrid=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0007), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    # Quick fine-tuning: unfreeze more layers
    print('Fine-tuning top 40 layers of both backbones...')
    mnet = model.layers[1]
    enet = model.layers[2]
    
    for layer in mnet.layers[-40:]:
        layer.trainable = True
    for layer in enet.layers[-40:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history_ft = model.fit(
        train_gen,
        epochs=3,
        validation_data=val_gen,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    model.save_weights('best_weights.h5')

    #--- TFLite Export ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model) # Use the fine-tuned model instance
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print('Saved model.tflite')

    #--- Save Training Curves
    plt.figure()
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig('hybrid_quick_accuracy.png')
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('hybrid_quick_loss.png')
    plt.close()

    #--- Save Confusion Matrix and CSV ---
    val_gen.reset()
    preds = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title('Hybrid Quick Confusion Matrix')
    plt.savefig('hybrid_quick_confusion_matrix.png')
    plt.close()

    #Save confusion matrix as CSV
    import pandas as pd
    pd.DataFrame(cm, index=class_names,
                 columns=class_names).to_csv('hybrid_quick_confusion_matrix.csv')

    #--- Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(class_names):
        idxs = np.where(y_true == i)[0]
        if len(idxs) > 0:
            acc = np.mean(y_pred[idxs] == i)
        else:
            acc = np.nan
        per_class_acc[name] = acc
    
    print("\nPer-class accuracy:")
    for name, acc in per_class_acc.items():
        print(f'{name:12}: {acc:.3f}')
    
    pd.Series(per_class_acc).to_csv('hybrid_quick_per_class_accuracy.csv')

    #--- Sample misclassifications
    misclassified = np.where(y_pred != y_true)[0]
    
    if len(misclassified) > 0:
        print(f'\nSample misclassifications (up to 10):')
        for idx in misclassified[:10]:
            print(f' File: {val_gen.filepaths[idx]} | True: {class_names[y_true[idx]]} | Pred: {class_names[y_pred[idx]]}')
        
        #Save to CSV
        miscls = [(val_gen.filepaths[idx], class_names[y_true[idx]], class_names[y_pred[idx]]) for
                  idx in misclassified]
        pd.DataFrame(miscls,
                     columns=['filepath', 'true_label', 'pred_label']).to_csv('hybrid_quick_misclassifications.csv',
                                                                             index=False)
    else:
        print('No misclassifications found!')
    
    #--- Save model summary to file
    with open('hybrid_quick_model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    print('Model, weights, TFLite, figures, and reports saved.')
