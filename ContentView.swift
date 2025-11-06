import SwiftUI
import TensorFlowLiteSwift
import UIKit

struct ContentView: View {
    @State private var predictionText: String = "Tap the button to run the model."
    @State private var currentIndex: Int = 0

    // Names of evaluation images in Assets.xcassets
    let evalImages = [
        "eval01", "eval02", "eval03", "eval04", "eval05",
        "eval06", "eval07", "eval08", "eval09", "eval10",
        "eval11", "eval12", "eval13", "eval14", "eval15",
        "eval16", "eval17", "eval18", "eval19", "eval20"
    ]

    var currentImageName: String {
        evalImages[currentIndex]
    }

    var body: some View {
        VStack(spacing: 16) {
            Text("Image: \(currentImageName)")
                .font(.headline)

            Image(currentImageName)
                .resizable()
                .scaledToFit()
                .frame(width: 200, height: 200)

            Text(predictionText)
                .font(.footnote)
                .multilineTextAlignment(.center)
                .padding()

            HStack {
                Button("Run Age, Gender & Expression Model") {
                    runGAEModel(imageName: currentImageName)
                }

                Button("Next image") {
                    currentIndex = (currentIndex + 1) % evalImages.count
                    predictionText = "Tap the button to run the model."
                }
            }
            .padding(.top, 8)
        }
        .padding()
    }

    /// Runs gender, age (3 classes) and expression (7 classes) models
    /// on the given image from the asset catalog.
    private func runGAEModel(imageName: String) {
        // 1. Load the UIImage from assets
        guard let uiImage = UIImage(named: imageName) else {
            predictionText = "Could not load image \(imageName)."
            return
        }

        // 2. Resize to 128x128 and convert to normalized RGB float32
        let targetSize = CGSize(width: 128, height: 128)
        guard let resizedImage = uiImage.resized(to: targetSize),
              let rgbData = resizedImage.rgbData() else {
            predictionText = "Failed to preprocess image."
            return
        }

        do {
            var options = Interpreter.Options()
            options.threadCount = 2

            // ----------------- GENDER MODEL -----------------
            guard let genderURL = Bundle.main.url(forResource: "gender_model", withExtension: "tflite") else {
                predictionText = "Could not find gender_model.tflite."
                return
            }
            let genderData = try Data(contentsOf: genderURL)
            let genderInterpreter = try Interpreter(modelData: genderData, options: options)
            try genderInterpreter.allocateTensors()
            try genderInterpreter.copy(rgbData, toInputAt: 0)
            try genderInterpreter.invoke()
            let genderOutput = try genderInterpreter.output(at: 0)
            let genderScores = [Float](unsafeData: genderOutput.data) ?? []

            guard genderScores.count == 2 else {
                predictionText = "Unexpected gender output size: \(genderScores.count)"
                return
            }

            let maleScore = genderScores[0]
            let femaleScore = genderScores[1]
            let isFemale = femaleScore > maleScore
            let genderLabel = isFemale ? "Female" : "Male"
            let genderConf = max(maleScore, femaleScore)

            // ----------------- AGE MODEL (3 CLASSES) -----------------
            guard let ageURL = Bundle.main.url(forResource: "age3_model", withExtension: "tflite") else {
                predictionText = "Could not find age3_model.tflite."
                return
            }
            let ageData = try Data(contentsOf: ageURL)
            let ageInterpreter = try Interpreter(modelData: ageData, options: options)
            try ageInterpreter.allocateTensors()
            try ageInterpreter.copy(rgbData, toInputAt: 0)
            try ageInterpreter.invoke()
            let ageOutput = try ageInterpreter.output(at: 0)
            let ageScores = [Float](unsafeData: ageOutput.data) ?? []

            guard ageScores.count == 3 else {
                predictionText = "Unexpected age output size: \(ageScores.count)"
                return
            }

            let ageIndex = ageScores.enumerated().max(by: { $0.element < $1.element })?.offset ?? 1
            let ageConf = ageScores[ageIndex]
            let ageLabel: String
            switch ageIndex {
            case 0: ageLabel = "Child"
            case 1: ageLabel = "Adult"
            case 2: ageLabel = "Elderly"
            default: ageLabel = "Unknown"
            }

            // ----------------- EXPRESSION MODEL (7 CLASSES) -----------------
            guard let exprURL = Bundle.main.url(forResource: "expression7_model", withExtension: "tflite") else {
                predictionText = "Could not find expression7_model.tflite."
                return
            }
            let exprData = try Data(contentsOf: exprURL)
            let exprInterpreter = try Interpreter(modelData: exprData, options: options)
            try exprInterpreter.allocateTensors()
            try exprInterpreter.copy(rgbData, toInputAt: 0)
            try exprInterpreter.invoke()
            let exprOutput = try exprInterpreter.output(at: 0)
            let exprScores = [Float](unsafeData: exprOutput.data) ?? []

            guard exprScores.count == 7 else {
                predictionText = "Unexpected expression output size: \(exprScores.count)"
                return
            }

            // FER2013 class order: 0=angry,1=disgust,2=fear,3=happy,4=neutral,5=sad,6=surprise
            let exprLabels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
            let exprIndex = exprScores.enumerated().max(by: { $0.element < $1.element })?.offset ?? 3
            let exprConf = exprScores[exprIndex]
            let exprLabel = exprLabels[exprIndex]

            // ----------------- UPDATE UI TEXT -----------------
            predictionText = String(
                format: """
                        Image: %@

                        Gender: %@ (%.1f%%)
                        Age group: %@ (%.1f%%)
                        Expression: %@ (%.1f%%)
                        """,
                imageName,
                genderLabel, genderConf * 100,
                ageLabel, ageConf * 100,
                exprLabel, exprConf * 100
            )

            // Also log a CSV-style line for later analysis
            print(
                "EVAL,\(imageName)," +
                "PredGender=\(genderLabel)," +
                "PredAge=\(ageLabel)," +
                "PredExpr=\(exprLabel)," +
                String(format: "GenderConf=%.3f,AgeConf=%.3f,ExprConf=%.3f",
                       genderConf, ageConf, exprConf)
            )

        } catch {
            print("Interpreter error:", error)
            predictionText = "Interpreter error: \(error)"
        }
    }
}

// MARK: - UIImage helpers

extension UIImage {
    /// Resize a UIImage to the given size.
    func resized(to size: CGSize) -> UIImage? {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0   // Ensure output is exactly 128x128 pixels
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: size))
        }
    }

    /// Convert UIImage to float32 RGB Data, normalized to [0, 1].
    func rgbData() -> Data? {
        guard let cgImage = self.cgImage else { return nil }

        let width = cgImage.width
        let height = cgImage.height

        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8

        var rawData = [UInt8](repeating: 0, count: Int(height * width * bytesPerPixel))
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        guard let context = CGContext(
            data: &rawData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        var floatArray = [Float]()
        floatArray.reserveCapacity(width * height * 3)

        for y in 0..<height {
            for x in 0..<width {
                let index = (y * width + x) * bytesPerPixel
                let r = Float(rawData[index]) / 255.0
                let g = Float(rawData[index + 1]) / 255.0
                let b = Float(rawData[index + 2]) / 255.0
                floatArray.append(r)
                floatArray.append(g)
                floatArray.append(b)
            }
        }

        return Data(bytes: floatArray, count: floatArray.count * MemoryLayout<Float>.size)
    }
}

// MARK: - Data -> [Float] helper

extension Array where Element == Float {
    init?(unsafeData: Data) {
        let count = unsafeData.count / MemoryLayout<Float>.size
        self = unsafeData.withUnsafeBytes { ptr in
            let buffer = ptr.bindMemory(to: Float.self)
            return Array(buffer[0..<count])
        }
    }
}
