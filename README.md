# GAE_Edge_IOS[README.md](https://github.com/user-attachments/files/23400277/README.md)
# Edge-Based Age, Gender and Facial Expression Recognition on iOS

This repository contains the code for an edge AI project that recognises **age group**, **gender**, and **facial expression** from face images, using **TensorFlow** for training and **TensorFlow Lite** for deployment in a native **iOS** app.

The system consists of:

- A Python / Colab training pipeline for three CNN models:
  - **Gender** (2 classes: male / female) – trained on **UTKFace**
  - **Age group** (3 classes: child / adult / elderly) – trained on **UTKFace**
  - **Expression** (7 classes: angry / disgust / fear / happy / neutral / sad / surprise) – trained on **FER2013**
- Conversion of the trained Keras models to **TensorFlow Lite** (`.tflite`)
- A **SwiftUI** iOS app that runs all three models on-device using `TensorFlowLiteSwift`

> This repository is intended as a companion to a university project report on edge-based age, gender and expression recognition.

---

## Repository structure

```text
.
├─ README.md
├─ notebooks/
│   └─ age_gender_expression_edge.ipynb
├─ models/
│   ├─ gender_model.tflite
│   ├─ age3_model.tflite
│   └─ expression7_model.tflite
└─ ios/
    └─ AgeGenderExpressionApp/
        ├─ AgeGenderExpressionApp.xcodeproj
        ├─ AgeGenderExpressionApp/
        │   ├─ ContentView.swift
        │   ├─ (other Swift source files)
        │   └─ Assets.xcassets
        └─ (test targets, project files, etc.)
```

- `notebooks/` – Jupyter/Colab notebooks used to train the models and generate the results reported in the project.
- `models/` – Exported TensorFlow Lite models used by the iOS app.
- `ios/` – Xcode project implementing the on-device inference app.

---

## Datasets

This project uses two public face datasets:

- **UTKFace** – large-scale face dataset with labels for **age**, **gender** and **ethnicity**. Used here for age-group and gender classification.
- **FER2013** – facial expression dataset with 7 emotion labels (angry, disgust, fear, happy, neutral, sad, surprise). Used for expression classification.

> **Note:** The datasets themselves are **not** included in this repository due to size and licensing. Please download them separately from their original sources (e.g. Kaggle or the official project pages) if you want to reproduce training.

---

## Models

Three separate models are trained:

- `gender_model.tflite`  
  - Input: 128×128 RGB face image  
  - Output: 2 softmax scores – `[Male, Female]`

- `age3_model.tflite`  
  - Input: 128×128 RGB face image  
  - Output: 3 softmax scores – `[Child, Adult, Elderly]`  
    (age bins derived from UTKFace ages: `<18`, `18–59`, `60+`)

- `expression7_model.tflite`  
  - Input: 128×128 RGB face image  
  - Output: 7 softmax scores –  
    `[Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise]`  
    trained on resized FER2013 images.

All three models use the same architecture pattern:

- Backbone: `MobileNetV2` (`include_top=False`, `pooling="avg"`, ImageNet weights)
- Head: Dropout + Dense(`num_classes`, `softmax`)  
- Loss: `sparse_categorical_crossentropy`  
- Optimiser: Adam

---

## Training (Python / Colab)

The main training pipeline is in:

- `notebooks/age_gender_expression_edge.ipynb`

The notebook covers:

1. **Dataset loading & pre-processing**
   - UTKFace: parse labels from filenames; map age to 3 groups; gender to 2 classes.
   - FER2013: load images from class folders; resize 48×48 → 128×128; convert grayscale → RGB.

2. **Model definition**
   - MobileNetV2 backbone + task-specific heads for:
     - gender (2 classes)
     - age group (3 classes)
     - expression (7 classes)

3. **Training on CPU vs GPU**
   - 5 epochs per model
   - Training on both CPU and GPU for UTKFace tasks to compare runtime
   - Measure: time per epoch, total time, final validation accuracy

4. **Export to TensorFlow Lite**
   - Use `tf.lite.TFLiteConverter.from_keras_model(model)`  
   - Save `gender_model.tflite`, `age3_model.tflite`, `expression7_model.tflite` to `models/`

To use the notebook:

1. Open it in **Google Colab** (you can upload from this repo).  
2. Mount Google Drive (if you store datasets there).  
3. Download UTKFace and FER2013 into your Drive and update paths in the notebook.  
4. Run all cells to train and export the models.

---

## iOS app (Swift / SwiftUI / TensorFlow Lite)

The iOS app is in:

- `ios/AgeGenderExpressionApp/`

It is a **SwiftUI** project that:

1. Loads a face image from the app’s asset catalog.
2. Resizes it to 128×128 and normalises to `[0, 1]` float32 RGB.
3. Runs three TensorFlow Lite models on-device:
   - `gender_model.tflite`
   - `age3_model.tflite`
   - `expression7_model.tflite`
4. Displays the predicted **Gender**, **Age group** and **Expression** along with confidence scores.

### Dependencies

- Xcode 15 or later
- iOS 17 simulator or an iOS device
- Swift Package: `TensorFlowLiteSwift`

To add `TensorFlowLiteSwift` via Swift Package Manager:

1. In Xcode, go to **File → Add Packages…**
2. Enter the TensorFlow Lite Swift package URL (from the official TensorFlow Lite Swift repo).
3. Add the `TensorFlowLiteSwift` product to your app target.

### Important Swift file: `ContentView.swift`

A simplified outline of `ContentView.swift`:

- `ContentView` shows:
  - The current evaluation image
  - A button *“Run Age, Gender & Expression Model”*
  - A text block with predictions and confidence scores
- The `runGAEModel(imageName:)` function:
  - Loads the `UIImage`
  - Preprocesses it to `Data` (float32 RGB)
  - Creates three `Interpreter` instances (gender, age, expression)
  - Copies the input into each model, invokes them, and decodes outputs

You can see a fully commented version of `ContentView.swift` in:

- `ios/AgeGenderExpressionApp/AgeGenderExpressionApp/ContentView.swift`

---

## How to run the iOS app

1. **Clone this repository**

   ```bash
   git clone https://github.com/<your-username>/age-gender-expression-edge-ios.git
   cd age-gender-expression-edge-ios/ios/AgeGenderExpressionApp
   ```

2. **Open the Xcode project**

   - Double-click `AgeGenderExpressionApp.xcodeproj`.

3. **Check TensorFlowLiteSwift dependency**

   - Make sure the Swift Package appears in **Package Dependencies**.
   - Ensure the app target links against `TensorFlowLiteSwift`.

4. **Run the app**

   - Select an iOS Simulator (e.g. iPhone 13).
   - Press **Run** in Xcode.
   - Tap **“Run Age, Gender & Expression Model”** to execute the models on the current image.

---

## Reproducing the evaluation on 20 images

In the project, 20 smartphone images (10 male, 10 female) were used to evaluate the system:

- Images were imported into `Assets.xcassets` as `eval01`, `eval02`, …, `eval20`.
- `ContentView` cycles through these images and calls `runGAEModel(imageName:)`.
- Predictions were logged in the Xcode console and compared against manually annotated ground truth.

The final reported performance on these 20 images:

- **Gender accuracy:** 70%  
- **Age group accuracy:** 70%  
- **Expression accuracy:** 70%  
- **All three correct simultaneously:** 30%

The actual evaluation images are **not** included in this repository for privacy reasons. You can reproduce the protocol by adding your own 20 face images and updating the asset names and ground truth accordingly.

---

## Use of AI tools

During this project, OpenAI’s ChatGPT (GPT-5 Thinking model) was used as a coding and writing assistant. ChatGPT was used to:

- help design and debug TensorFlow/Keras training and data loading code,
- suggest Swift/TensorFlow Lite integration patterns for the iOS app,
- and assist in structuring this README and the corresponding project report.

All code was executed, tested and adapted by the author, and all final design decisions, parameter choices and interpretations of the results are the author’s own.

---

## License

Add an appropriate license here (e.g. MIT, or “All rights reserved” if you prefer). For example:

```text
MIT License – see LICENSE file for details.
```
