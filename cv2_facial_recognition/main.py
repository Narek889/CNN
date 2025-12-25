import cv2
import numpy as np
import os
import pickle
import time
import gc
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

print("🚀 Advanced Face Recognition System")
print("=" * 60)

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detected: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"⚠️ GPU config error: {e}")


class FaceRecognizer:
    def __init__(self, input_shape=(112, 112, 3)):
        self.input_shape = input_shape
        self.num_classes = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        self.is_trained = False

        # Initialize face detector
        try:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception as e:
            print(f"⚠️ Error loading face detector: {e}")

    def load_training_data_with_min_samples(self, training_folder, max_images_per_person=30, min_samples_per_class=10):
        """Load data with minimum samples per class requirement"""
        print("📂 Loading training data...")

        if not os.path.exists(training_folder):
            print(f"❌ Training folder not found!")
            return None, None, None

        person_folders = [f for f in os.listdir(training_folder)
                          if os.path.isdir(os.path.join(training_folder, f))]

        if not person_folders:
            print("❌ No person folders found!")
            return None, None, None

        all_images = []
        all_labels = []
        person_images = {}

        print(f"🔍 Processing {len(person_folders)} people...")

        for person_name in person_folders:
            person_path = os.path.join(training_folder, person_name)
            images = []

            # Get image files
            image_files = [f for f in os.listdir(person_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Limit to max_images_per_person
            image_files = image_files[:max_images_per_person]

            for image_name in image_files:
                image_path = os.path.join(person_path, image_name)

                try:
                    # Read and preprocess image
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Convert to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Resize to input shape
                        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))

                        # Normalize
                        image = image.astype('float32') / 255.0

                        images.append(image)
                except Exception as e:
                    continue

            # Only include if we have enough samples
            if len(images) >= min_samples_per_class:
                all_images.extend(images)
                all_labels.extend([person_name] * len(images))
                person_images[person_name] = images
            else:
                print(f"⚠️  Skipped {person_name}: {len(images)} images (need {min_samples_per_class})")

        if not all_images:
            print("❌ No valid images found!")
            return None, None, None

        print(f"\n📊 Dataset loaded:")
        print(f"   Total people: {len(person_images)}")
        print(f"   Total images: {len(all_images)}")
        print(f"   Average images per person: {len(all_images) / len(person_images):.1f}")

        return np.array(all_images), np.array(all_labels), person_images

    def manual_train_test_split(self, X, y, test_size=0.3, min_samples_in_test=2):
        """Manual train-test split that ensures each class has enough samples in test set"""
        print("🔄 Performing manual train-test split...")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        unique_classes = np.unique(y_encoded)

        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []

        for class_idx in unique_classes:
            # Get indices for this class
            class_indices = np.where(y_encoded == class_idx)[0]

            if len(class_indices) >= min_samples_in_test + 1:  # Need at least 1 for train
                # Random shuffle
                np.random.shuffle(class_indices)

                # Split
                test_count = max(1, int(len(class_indices) * test_size))
                test_count = min(test_count, len(class_indices) - 1)  # Leave at least 1 for train

                test_indices = class_indices[:test_count]
                train_indices = class_indices[test_count:]

                X_train_list.append(X[train_indices])
                X_test_list.append(X[test_indices])
                y_train_list.append(y_encoded[train_indices])
                y_test_list.append(y_encoded[test_indices])
            else:
                # If not enough samples, put all in training
                print(
                    f"⚠️  Class {self.label_encoder.inverse_transform([class_idx])[0]} has only {len(class_indices)} samples - using all for training")
                X_train_list.append(X[class_indices])
                y_train_list.append(y_encoded[class_indices])

        # Combine all splits
        X_train = np.concatenate(X_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.array([])
        y_train = np.concatenate(y_train_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.array([])

        # Shuffle
        train_indices = np.random.permutation(len(X_train))
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]

        if len(X_test) > 0:
            test_indices = np.random.permutation(len(X_test))
            X_test = X_test[test_indices]
            y_test = y_test[test_indices]

        print(f"✅ Split completed:")
        print(f"   Training: {X_train.shape[0]} images")
        print(f"   Testing: {X_test.shape[0]} images")

        return X_train, X_test, y_train, y_test

    def build_improved_model(self, num_classes, learning_rate=0.001):
        """Build an improved CNN model"""
        print(f"🧠 Building improved model for {num_classes} classes...")

        model = keras.Sequential([
            layers.Input(shape=self.input_shape),

            # Data augmentation
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),

            # First block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            # Third block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),

            # Fourth block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),

            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])

        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        # Use sparse categorical crossentropy with sparse top-k metrics
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print(f"✅ Model built for {num_classes} classes")
        print(f"   Total parameters: {model.count_params():,}")
        return model

    def train_model(self, training_folder, epochs=50, batch_size=32, learning_rate=0.001, min_samples_per_class=10):
        """Main training function with proper data splitting"""
        print("\n" + "=" * 60)
        print("🚀 STARTING TRAINING")
        print("=" * 60)

        # Load data with minimum samples requirement
        data = self.load_training_data_with_min_samples(
            training_folder,
            max_images_per_person=50,
            min_samples_per_class=min_samples_per_class
        )

        if data[0] is None:
            return None

        X, y, person_images = data

        print(f"\n📊 Dataset info:")
        print(f"   Classes: {len(person_images)}")
        print(f"   Images: {len(X)}")

        # Manual split to ensure proper distribution
        X_train, X_test, y_train, y_test = self.manual_train_test_split(
            X, y, test_size=0.2, min_samples_in_test=2
        )

        if len(X_test) == 0:
            print("❌ Not enough data for testing!")
            return None

        # Further split test into validation and test
        test_indices = np.random.permutation(len(X_test))
        split_point = len(test_indices) // 2

        X_val = X_test[test_indices[:split_point]]
        y_val = y_test[test_indices[:split_point]]
        X_test = X_test[test_indices[split_point:]]
        y_test = y_test[test_indices[split_point:]]

        print(f"\n✂️ Final split:")
        print(f"   Training: {X_train.shape[0]} images")
        print(f"   Validation: {X_val.shape[0]} images")
        print(f"   Testing: {X_test.shape[0]} images")

        # Build model
        self.num_classes = len(self.label_encoder.classes_)
        self.build_improved_model(self.num_classes, learning_rate)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                mode='max',
                verbose=1,
                min_delta=0.001
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                mode='max',
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.CSVLogger('training_log.csv', append=True)
        ]

        # Adjust batch size for large datasets
        if len(X_train) > 10000:
            batch_size = min(batch_size * 2, 128)
            print(f"📊 Large dataset detected. Increasing batch size to {batch_size}")

        print(f"\n🎯 Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Input shape: {self.input_shape}")

        # Train the model
        start_time = time.time()

        try:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )

            training_time = time.time() - start_time
            print(f"\n✅ Training completed in {training_time:.2f} seconds ({training_time / 60:.1f} minutes)")

            # Evaluate
            self.evaluate_model(X_test, y_test)

            self.is_trained = True
            return self.history

        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_model(self, X_test, y_test):
        """Enhanced evaluation with detailed metrics"""
        print("\n" + "=" * 60)
        print("📊 DETAILED MODEL EVALUATION")
        print("=" * 60)

        try:
            # Get predictions
            print("🔍 Making predictions...")
            y_pred_proba = self.model.predict(X_test, verbose=0, batch_size=32)
            y_pred = np.argmax(y_pred_proba, axis=1)

            from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                         f1_score, confusion_matrix, classification_report)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Top-k accuracy
            def top_k_accuracy(y_true, y_pred_proba, k=3):
                top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
                correct = 0
                for i in range(len(y_true)):
                    if y_true[i] in top_k_preds[i]:
                        correct += 1
                return correct / len(y_true)

            top_3_acc = top_k_accuracy(y_test, y_pred_proba, k=3)
            top_5_acc = top_k_accuracy(y_test, y_pred_proba, k=5)
            top_10_acc = top_k_accuracy(y_test, y_pred_proba, k=10)

            print("\n🎯 Performance Metrics:")
            print("=" * 40)
            print(f"   ✅ Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"   ✅ Test Precision: {precision:.4f}")
            print(f"   ✅ Test Recall: {recall:.4f}")
            print(f"   ✅ Test F1-Score: {f1:.4f}")
            print(f"   🏆 Top-3 Accuracy: {top_3_acc:.4f} ({top_3_acc * 100:.2f}%)")
            print(f"   🏆 Top-5 Accuracy: {top_5_acc:.4f} ({top_5_acc * 100:.2f}%)")
            print(f"   🏆 Top-10 Accuracy: {top_10_acc:.4f} ({top_10_acc * 100:.2f}%)")

            # Confidence analysis
            max_confidences = np.max(y_pred_proba, axis=1)
            correct_mask = (y_pred == y_test)
            wrong_mask = (y_pred != y_test)

            if np.any(correct_mask):
                correct_conf = max_confidences[correct_mask]
                print(f"\n📊 Confidence Analysis:")
                print(f"   ✓ Correct predictions ({np.sum(correct_mask)}):")
                print(f"      Average confidence: {np.mean(correct_conf):.2%}")
                print(f"      Min confidence: {np.min(correct_conf):.2%}")
                print(f"      Max confidence: {np.max(correct_conf):.2%}")

            if np.any(wrong_mask):
                wrong_conf = max_confidences[wrong_mask]
                print(f"   ✗ Wrong predictions ({np.sum(wrong_mask)}):")
                print(f"      Average confidence: {np.mean(wrong_conf):.2%}")
                print(f"      Min confidence: {np.min(wrong_conf):.2%}")
                print(f"      Max confidence: {np.max(wrong_conf):.2%}")

            # Show sample predictions
            print(f"\n🔍 Sample Predictions (first 5 test samples):")
            print("=" * 60)

            for i in range(min(5, len(X_test))):
                true_class = self.label_encoder.inverse_transform([y_test[i]])[0]
                pred_class = self.label_encoder.inverse_transform([y_pred[i]])[0]
                confidence = max_confidences[i]

                # Get top 3 predictions
                top_3_idx = np.argsort(y_pred_proba[i])[-3:][::-1]
                top_3_classes = self.label_encoder.inverse_transform(top_3_idx)
                top_3_conf = y_pred_proba[i][top_3_idx]

                is_correct = (y_pred[i] == y_test[i])
                symbol = "✓" if is_correct else "✗"

                print(f"\n{symbol} Sample {i + 1}:")
                print(f"   True: {true_class}")
                print(f"   Predicted: {pred_class} ({confidence:.2%})")

                if not is_correct:
                    print(f"   Top 3 alternatives:")
                    for j in range(3):
                        print(f"     {j + 1}. {top_3_classes[j]} ({top_3_conf[j]:.2%})")

            # Per-class accuracy for first 10 classes
            print(f"\n📈 Per-class accuracy (first 10 classes):")
            unique_classes = np.unique(y_test)

            for cls_idx in unique_classes[:10]:
                cls_mask = y_test == cls_idx
                if np.sum(cls_mask) > 0:
                    cls_acc = np.mean(y_pred[cls_mask] == y_test[cls_mask])
                    class_name = self.label_encoder.inverse_transform([cls_idx])[0]
                    print(f"   {class_name}: {cls_acc:.2%} ({np.sum(cls_mask)} samples)")

            # Save evaluation report
            self.save_evaluation_report(y_test, y_pred, y_pred_proba)

        except Exception as e:
            print(f"⚠️ Evaluation error: {e}")
            import traceback
            traceback.print_exc()

    def evaluate_face_detection(self, specific_person_id="85553"):
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import cv2

        def compute_iou(boxA, boxB):
            x1 = max(boxA[0], boxB[0])
            y1 = max(boxA[1], boxB[1])
            x2 = min(boxA[2], boxB[2])
            y2 = min(boxA[3], boxB[3])
            if x2 <= x1 or y2 <= y1:
                return 0.0
            inter = (x2 - x1) * (y2 - y1)
            areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            return inter / float(areaA + areaB - inter)

        def compute_ap(tp, fp, num_gt):
            if num_gt == 0:
                return 0.0
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / num_gt
            precision = tp_cum / (tp_cum + fp_cum + 1e-8)
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([0.0], precision, [0.0]))
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            return ap

        print("\n" + "=" * 60)
        print("🧪 FACE DETECTION EVALUATION (Haar Cascade)")
        print("=" * 60)

        folder_path = os.path.join("training_data", specific_person_id)
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return

        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        target_filenames = ['5810062.jpg', '5810063.jpg', '5810066.jpg', '5810067.jpg', '5810068.jpg', '5810061.jpg']
        image_paths = [p for p in image_paths if os.path.basename(p) in target_filenames]

        if not image_paths:
            print("❌ No target images found!")
            return

        print(f"🔍 Processing {len(image_paths)} selected images...")

        manual_gt = {
            '5810061.jpg': [10, 15, 100, 110],
            '5810062.jpg': [10, 15, 100, 110],
            '5810063.jpg': [10, 15, 100, 110],
            '5810066.jpg': [10, 15, 100, 110],
            '5810067.jpg': [10, 15, 100, 110],
            '5810068.jpg': [10, 15, 100, 110],
        }

        tp = []
        fp = []
        iou_list = []
        num_gt = len(image_paths)  # 6 նկար, յուրաքանչյուրում 1 դեմք

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                tp.append(0)
                fp.append(0)  # skip
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = self.face_detector.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

            filename = os.path.basename(img_path)
            gt_box = manual_gt.get(filename)

            has_detection = len(detected) > 0
            has_gt = gt_box is not None

            if has_detection and has_gt:
                pred_box = [detected[0][0], detected[0][1], detected[0][0] + detected[0][2],
                            detected[0][1] + detected[0][3]]
                iou = compute_iou(pred_box, gt_box)
                iou_list.append(iou)
                if iou >= 0.5:
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            elif has_detection and not has_gt:
                tp.append(0)
                fp.append(1)
            elif not has_detection and has_gt:
                tp.append(0)
                fp.append(0)  # miss, ոչ FP
            else:
                tp.append(0)
                fp.append(0)

        ap_50 = compute_ap(tp, fp, num_gt)

        print(f"\n🎯 FINAL METRICS (6 images):")
        if iou_list:
            print(f"   Average IoU: {np.mean(iou_list):.4f}")
        print(f"   AP@0.5: {ap_50:.4f} ({ap_50 * 100:.2f}%)")

        # PR Curve
        if sum(tp) + sum(fp) > 0:
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / num_gt
            precision = tp_cum / (tp_cum + fp_cum + 1e-8)
            plt.figure(figsize=(10, 7))
            plt.plot(recall, precision, 'b-', linewidth=3, label=f'AP@0.5 = {ap_50:.4f}')
            plt.fill_between(recall, precision, alpha=0.2, color='blue')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(0, 1)
            plt.ylim(0, 1.05)
            plt.tight_layout()
            plt.show()

        # Visualization
        rows = 2
        cols = 3
        plt.figure(figsize=(15, 10))
        for i, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = self.face_detector.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(rows, cols, i + 1)

            filename = os.path.basename(img_path)
            gt_box = manual_gt.get(filename)

            if len(detected) > 0:
                pred_box = [detected[0][0], detected[0][1], detected[0][0] + detected[0][2],
                            detected[0][1] + detected[0][3]]
                plt.gca().add_patch(
                    plt.Rectangle((pred_box[0], pred_box[1]), pred_box[2] - pred_box[0], pred_box[3] - pred_box[1],
                                  fill=False, edgecolor='lime', linewidth=3))

            if gt_box is not None:
                plt.gca().add_patch(plt.Rectangle((gt_box[0], gt_box[1]), gt_box[2] - gt_box[0], gt_box[3] - gt_box[1],
                                                  fill=False, edgecolor='red', linewidth=3))
                if len(detected) > 0:
                    iou = compute_iou(pred_box, gt_box)
                    plt.text(10, 30, f'IoU: {iou:.4f}', color='yellow', fontsize=14,
                             bbox=dict(facecolor='black', alpha=0.8))

            plt.imshow(img_rgb)
            plt.title(f"{filename}\nDetected: {len(detected)} face(s)")
            plt.axis('off')

        plt.suptitle("IoU Visualization (Red: GT, Green: Predicted)")
        plt.tight_layout()
        plt.show()

    def save_evaluation_report(self, y_test, y_pred, y_pred_proba):
        """Save detailed evaluation report"""
        report = {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'label_encoder_classes': self.label_encoder.classes_,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_classes': self.num_classes,
            'test_samples': len(y_test)
        }

        with open('evaluation_report.pkl', 'wb') as f:
            pickle.dump(report, f)

        print(f"\n💾 Evaluation report saved to evaluation_report.pkl")

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return

        try:
            plt.figure(figsize=(15, 5))

            # Accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in self.history.history:
                plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

            # Loss
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'], label='Training Loss')
            if 'val_loss' in self.history.history:
                plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
            plt.show()

        except Exception as e:
            print(f"⚠️ Plotting error: {e}")

    def analyze_dataset(self, training_folder):
        """Analyze dataset distribution"""
        print("\n📊 DATASET ANALYSIS")
        print("=" * 60)

        if not os.path.exists(training_folder):
            print(f"❌ Training folder not found!")
            return

        person_folders = [f for f in os.listdir(training_folder)
                          if os.path.isdir(os.path.join(training_folder, f))]

        if not person_folders:
            print("❌ No person folders found!")
            return

        person_counts = {}
        for person in person_folders:
            person_path = os.path.join(training_folder, person)
            images = [f for f in os.listdir(person_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            person_counts[person] = len(images)

        print(f"📈 Dataset Statistics:")
        print(f"   Total people: {len(person_counts)}")
        print(f"   Total images: {sum(person_counts.values())}")
        print(f"   Average images per person: {np.mean(list(person_counts.values())):.1f}")
        print(f"   Minimum images: {np.min(list(person_counts.values()))}")
        print(f"   Maximum images: {np.max(list(person_counts.values()))}")

        # Distribution
        counts = list(person_counts.values())
        print(f"\n📊 Distribution:")
        bins = [0, 5, 10, 20, 30, 50, 100, 200]
        for i in range(len(bins) - 1):
            count = sum(1 for c in counts if bins[i] <= c < bins[i + 1])
            if count > 0:
                print(f"   {bins[i]}-{bins[i + 1] - 1} images: {count} people")

        # Recommendations
        print(f"\n💡 Recommendations:")
        min_count = np.min(counts)
        if min_count < 5:
            print(f"   ⚠️  {sum(1 for c in counts if c < 5)} people have less than 5 images")
            print(f"   Consider: Remove or collect more images for these people")

        if len(person_counts) > 300:
            print(f"   ⚠️  Large number of classes ({len(person_counts)})")
            print(f"   Consider: Start with a reduced dataset of 50-100 people")

        print("=" * 60)

    def create_reduced_dataset(self, training_folder, output_folder="training_data_reduced", max_people=100,
                               min_images=20):
        """Create a reduced dataset for testing"""
        print(f"\n📉 Creating reduced dataset...")

        if not os.path.exists(training_folder):
            print(f"❌ Training folder not found!")
            return None

        person_folders = [f for f in os.listdir(training_folder)
                          if os.path.isdir(os.path.join(training_folder, f))]

        # Count images per person
        person_counts = {}
        for person in person_folders:
            person_path = os.path.join(training_folder, person)
            images = [f for f in os.listdir(person_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) >= min_images:
                person_counts[person] = len(images)

        # Sort by number of images (descending)
        sorted_people = sorted(person_counts.items(), key=lambda x: x[1], reverse=True)
        selected_people = sorted_people[:max_people]

        # Create output folder
        if os.path.exists(output_folder):
            import shutil
            shutil.rmtree(output_folder)

        os.makedirs(output_folder, exist_ok=True)

        print(f"\n✅ Selected {len(selected_people)} people for reduced dataset:")
        for i, (person, count) in enumerate(selected_people[:20], 1):
            print(f"   {i:3d}. {person}: {count} images")

        if len(selected_people) > 20:
            print(f"   ... and {len(selected_people) - 20} more")

        # Copy selected people
        print(f"\n📁 Copying data to {output_folder}...")
        for person, _ in selected_people:
            src_path = os.path.join(training_folder, person)
            dst_path = os.path.join(output_folder, person)

            import shutil
            shutil.copytree(src_path, dst_path)

        print(f"\n📊 Reduced Dataset Summary:")
        print(f"   Total people: {len(selected_people)}")
        total_images = sum(count for _, count in selected_people)
        print(f"   Total images: {total_images}")
        print(f"   Average images per person: {total_images / len(selected_people):.1f}")
        print(f"   Min images: {min(selected_people, key=lambda x: x[1])[1]}")
        print(f"   Max images: {max(selected_people, key=lambda x: x[1])[1]}")

        return output_folder

    def recognize_face(self, face_image, confidence_threshold=0.7):
        """Recognize a face"""
        if not self.is_trained or self.model is None:
            return "Unknown", 0.0, [], []

        try:
            # Convert to RGB if needed
            if len(face_image.shape) == 2:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 1:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 4:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2RGB)

            # Preprocess
            face_resized = cv2.resize(face_image, (self.input_shape[1], self.input_shape[0]))
            face_normalized = face_resized.astype('float32') / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)

            # Predict
            predictions = self.model.predict(face_batch, verbose=0)
            confidence = np.max(predictions[0])
            predicted_class = np.argmax(predictions[0])

            # Get top 5 predictions
            top_5_idx = np.argsort(predictions[0])[-5:][::-1]
            top_5_classes = self.label_encoder.inverse_transform(top_5_idx)
            top_5_conf = predictions[0][top_5_idx]

            if confidence > confidence_threshold:
                name = self.label_encoder.inverse_transform([predicted_class])[0]
                return name, confidence, top_5_classes, top_5_conf
            else:
                return "Unknown", confidence, top_5_classes, top_5_conf

        except Exception as e:
            print(f"⚠️ Recognition error: {e}")
            return "Unknown", 0.0, [], []

    def save_model(self, model_path='face_model.h5'):
        """Save the trained model"""
        if self.is_trained and self.model is not None:
            try:
                # Save model
                self.model.save(model_path, save_format='tf')

                # Save metadata
                metadata = {
                    'label_encoder_classes': self.label_encoder.classes_,
                    'input_shape': self.input_shape,
                    'num_classes': self.num_classes,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                with open('model_metadata.pkl', 'wb') as f:
                    pickle.dump(metadata, f)

                print(f"💾 Model saved as {model_path}")
                print(f"💾 Metadata saved as model_metadata.pkl")
                return True

            except Exception as e:
                print(f"❌ Error saving model: {e}")
                return False
        else:
            print("❌ No trained model to save!")
            return False

    def load_model(self, model_path='face_model.h5'):
        """Load a trained model"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                return False

            if not os.path.exists('model_metadata.pkl'):
                print("❌ Metadata file not found!")
                return False

            print("🔄 Loading model...")

            # Load model
            self.model = keras.models.load_model(model_path, compile=False)

            # Recompile
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Load metadata
            with open('model_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)

            # Restore label encoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = metadata['label_encoder_classes']

            self.input_shape = metadata['input_shape']
            self.num_classes = metadata['num_classes']
            self.is_trained = True

            print("✅ Model loaded successfully!")
            print(f"   Model: {model_path}")
            print(f"   Input shape: {self.input_shape}")
            print(f"   Number of classes: {self.num_classes}")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


def capture_new_face(recognizer, person_name, save_folder="training_data"):
    """Capture face images for training with optimized capture rate, FPS tracking and UI improvements"""

    if recognizer.face_detector is None:
        print("❌ Face detector not available!")
        return

    person_folder = os.path.join(save_folder, person_name)
    os.makedirs(person_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return

    # Camera configuration
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    max_images = 50
    min_face_size = 110    # better quality than 100px
    capture_interval = 1.0 / 3.0  # 3 captures/sec
    last_capture_time = 0
    count = 0

    window_name = "Face Capture - Press Q to stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 650)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(window_name, 80, 50)

    fps_start = time.time()
    fps_frames = 0
    fps_value = 0

    print(f"\n📸 CAPTURING FACES FOR: {person_name}")
    print("=" * 60)
    print(f"Target: {max_images} images  |  Rate: 3 img/sec")
    print("Move head, change angles, light, expressions")
    print("Press Q anytime to stop")
    print("=" * 60)

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = recognizer.face_detector.detectMultiScale(gray, 1.1, 5, minSize=(min_face_size, min_face_size))

        now = time.time()
        passed = now - last_capture_time

        if len(faces) > 0 and passed >= capture_interval:
            (x, y, w, h) = faces[0]  # capture only first face
            face = frame[y:y + h, x:x + w]
            resized = cv2.resize(face, (112, 112))

            filename = os.path.join(person_folder, f"{person_name}_{count:04d}.jpg")
            cv2.imwrite(filename, resized)

            last_capture_time = now
            count += 1

            cv2.putText(display, f"CAPTURED {count}/{max_images}", (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw bounding boxes for display
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # FPS calculation
        fps_frames += 1
        if fps_frames >= 30:
            fps_value = fps_frames / (time.time() - fps_start)
            fps_start = time.time()
            fps_frames = 0

        # HUD
        cv2.putText(display, f"Person: {person_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.putText(display, f"Images: {count}/{max_images}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.putText(display, f"FPS: {fps_value:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.putText(display, f"Next capture: {max(0, capture_interval - passed):.2f}s",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
        cv2.putText(display, "Press 'Q' to stop", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255),2)

        cv2.imshow(window_name, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        print(f"\n🎉 Successfully captured {count} images for {person_name}")
        print(f"📁 Saved in: {person_folder}")
    else:
        print("❌ No faces captured!")



def real_time_recognition(recognizer):
    """Real-time face recognition"""
    if not recognizer.is_trained:
        print("❌ Please train or load a model first!")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n🔍 REAL-TIME FACE RECOGNITION")
    print("=" * 50)
    print("Press 'Q' to quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = recognizer.face_detector.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Recognize
            name, confidence, top_5_classes, top_5_conf = recognizer.recognize_face(face_roi)

            # Draw rectangle
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Display name and confidence
            if name != "Unknown":
                label = f"{name} ({confidence:.1%})"
            else:
                label = f"Unknown ({confidence:.1%})"

            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display instructions
        cv2.putText(frame, "Face Recognition - Press Q to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("🤖 ADVANCED FACE RECOGNITION SYSTEM")
    print("=" * 60)

    # Check for training data folder
    if not os.path.exists("training_data"):
        os.makedirs("training_data")
        print("📁 Created training_data folder")

    recognizer = FaceRecognizer(input_shape=(112, 112, 3))

    while True:
        print(f"\n🏠 MAIN MENU")
        print("1. 📸 Add new person")
        print("2. 📊 Analyze dataset")
        print("3. 📉 Create reduced dataset")
        print("4. 🚀 Train model")
        print("5. 🔍 Real-time recognition")
        print("6. 📈 Show training history")
        print("7. 💾 Save model")
        print("8. 📂 Load model")
        print("9. 🚪 Exit")
        print("10. 🧪 Evaluate Face Detection (AP@0.5 + IoU)")

        choice = input("\nChoose option (1-10): ").strip()

        if choice == '1':
            print("\n📸 ADD NEW PERSON")
            print("-" * 30)
            name = input("Enter person's name: ").strip()
            if name:
                if os.path.exists(os.path.join("training_data", name)):
                    print(f"⚠️  Person '{name}' already exists!")
                    overwrite = input("Add more images? (y/n): ").strip().lower()
                    if overwrite != 'y':
                        continue
                capture_new_face(recognizer, name)
            else:
                print("❌ Please enter a valid name")

        elif choice == '2':
            print("\n📊 DATASET ANALYSIS")
            print("-" * 30)
            recognizer.analyze_dataset("training_data")

        elif choice == '3':
            print("\n📉 CREATE REDUCED DATASET")
            print("-" * 30)
            print("This creates a smaller dataset for faster testing.")

            try:
                max_people = int(input("Max number of people (default 100): ") or "100")
                min_images = int(input("Min images per person (default 20): ") or "20")
            except ValueError:
                max_people, min_images = 100, 20

            reduced_folder = recognizer.create_reduced_dataset("training_data", max_people=max_people,
                                                               min_images=min_images)

            if reduced_folder:
                print(f"\n✅ Reduced dataset created: {reduced_folder}")
                train_now = input("Train on reduced dataset now? (y/n): ").strip().lower()
                if train_now == 'y':
                    print("\n🚀 Training on reduced dataset...")

                    try:
                        epochs = int(input("Epochs (default 30): ") or "30")
                        batch_size = int(input("Batch size (default 32): ") or "32")
                        learning_rate = float(input("Learning rate (default 0.001): ") or "0.001")
                        min_samples = int(input("Min samples per class (default 10): ") or "10")
                    except ValueError:
                        epochs, batch_size, learning_rate, min_samples = 30, 32, 0.001, 10

                    recognizer.train_model(
                        reduced_folder,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        min_samples_per_class=min_samples
                    )

        elif choice == '4':
            if not os.path.exists("training_data"):
                print("❌ No training data! Please add people first.")
                continue

            print("\n🚀 TRAIN MODEL")
            print("-" * 30)

            try:
                epochs = int(input("Epochs (default 50): ") or "50")
                batch_size = int(input("Batch size (default 32): ") or "32")
                learning_rate = float(input("Learning rate (default 0.001): ") or "0.001")
                min_samples = int(input("Min samples per class (default 10): ") or "10")
            except ValueError:
                epochs, batch_size, learning_rate, min_samples = 50, 32, 0.001, 10

            print(f"\n🎯 Training Configuration:")
            print(f"  Epochs: {epochs}")
            print(f"  Batch size: {batch_size}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Min samples per class: {min_samples}")

            confirm = input("\nStart training? (y/n): ").strip().lower()
            if confirm == 'y':
                recognizer.train_model(
                    "training_data",
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    min_samples_per_class=min_samples
                )
            else:
                print("❌ Training cancelled")

        elif choice == '5':
            if recognizer.is_trained:
                real_time_recognition(recognizer)
            else:
                print("❌ Please train or load a model first!")

        elif choice == '6':
            if recognizer.history:
                recognizer.plot_training_history()
            else:
                print("❌ No training history available!")

        elif choice == '7':
            if recognizer.is_trained:
                model_name = input("Model filename (default 'face_model.h5'): ").strip()
                if not model_name:
                    model_name = 'face_model.h5'

                if recognizer.save_model(model_name):
                    print("✅ Model saved successfully!")
                else:
                    print("❌ Failed to save model!")
            else:
                print("❌ No trained model to save!")

        elif choice == '8':
            print("\n📂 LOAD EXISTING MODEL")
            print("-" * 30)

            model_files = [f for f in os.listdir('.')
                           if f.endswith('.h5') or f.endswith('.keras')]

            if model_files:
                print("Available model files:")
                for i, f in enumerate(model_files, 1):
                    print(f"  {i}. {f}")
                print("  *. Enter custom filename")
            else:
                print("No model files found")

            model_choice = input("\nEnter filename or number: ").strip()

            if model_choice.isdigit():
                idx = int(model_choice) - 1
                if 0 <= idx < len(model_files):
                    model_name = model_files[idx]
                else:
                    print("❌ Invalid number!")
                    continue
            else:
                model_name = model_choice if model_choice else 'face_model.h5'

            if not model_name.endswith(('.h5', '.keras')):
                model_name += '.h5'

            if recognizer.load_model(model_name):
                print("✅ Model loaded successfully!")
            else:
                print("❌ Failed to load model!")

        elif choice == '9':
            print("\n👋 Goodbye!")
            print("Thank you for using the Face Recognition System!")
            break

        elif choice == '10':
            print("\n🧪 FACE DETECTION EVALUATION")
            print("-" * 50)

            recognizer.evaluate_face_detection(specific_person_id="85553")
        else:
            print("❌ Invalid choice! Please enter 1-9")


if __name__ == "__main__":
    main()

