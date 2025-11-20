import argparse
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from preprocess import create_generators
from utils import ensure_dir


def build_simple_cnn(input_shape=(128, 128, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def main(args):
    data_dir = "Mask-Detection-System/data"
    model_dir = "Mask-Detection-System/models"

    ensure_dir(model_dir)

    train_gen, val_gen, test_gen = create_generators(base_dir=data_dir)

    model = build_simple_cnn()

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = os.path.join(model_dir, "mask_detector.h5")

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        patience=3,
        factor=0.5,
        verbose=1
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[checkpoint, reduce_lr]
    )

    print("\nEvaluating on test data:")
    loss, acc = model.evaluate(test_gen)
    print(f"Test Loss: {loss:.4f}  |  Test Accuracy: {acc:.4f}")

    print(f"\nBest model saved at: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)
