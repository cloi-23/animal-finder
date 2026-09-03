import Capacitor
import Foundation
import TensorFlowLite
import UIKit

@objc(AnimalAIPlugin)
public class AnimalAIPlugin: CAPPlugin {
    private let modelPath = "Models/oxford_iiit_pet"
    private var interpreter: Interpreter?
    private var labels: [String] = []

    private let dogBreeds: Set<String> = [
        "American Bulldog", "American Pit Bull Terrier", "Basset Hound", "Beagle", "Boxer",
        "Chihuahua", "English Cocker Spaniel", "English Setter", "German Shorthaired",
        "Great Pyrenees", "Havanese", "Japanese Chin", "Keeshond", "Leonberger",
        "Miniature Pinscher", "Newfoundland", "Pomeranian", "Pug", "Saint Bernard",
        "Samoyed", "Scottish Terrier", "Shiba Inu", "Staffordshire Bull Terrier",
        "Wheaten Terrier", "Yorkshire Terrier"
    ]

    private let catBreeds: Set<String> = [
        "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
        "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx"
    ]

    public override func load() {
        super.load()
        guard let path = Bundle.main.path(forResource: "oxford_iiit_pet", ofType: "tflite", inDirectory: "Models") else { return }
        do {
            interpreter = try Interpreter(modelPath: path)
            try interpreter?.allocateTensors()
            labels = try loadLabels()
        } catch {
            interpreter = nil
            print("Animal AI model unavailable: \(error)")
        }
    }

    @objc public func modelInfo(_ call: CAPPluginCall) {
        guard let interpreter else {
            call.reject("Animal AI model is not loaded")
            return
        }
        do {
            let input = try interpreter.input(at: 0)
            let output = try interpreter.output(at: 0)
            call.resolve([
                "inputCount": interpreter.inputs.count,
                "outputCount": interpreter.outputs.count,
                "inputs": [["index": 0, "name": input.name, "type": String(describing: input.dataType), "shape": input.shape.dimensions]],
                "outputs": [["index": 0, "name": output.name, "type": String(describing: output.dataType), "shape": output.shape.dimensions]]
            ])
        } catch {
            call.reject("Unable to inspect Animal AI model", error)
        }
    }

    @objc public func classify(_ call: CAPPluginCall) {
        guard let interpreter else {
            call.reject("Animal AI model is not loaded")
            return
        }
        guard let imageData = call.getString("image"), let image = decodeImage(imageData) else {
            call.reject("Unable to decode image")
            return
        }

        do {
            let input = try interpreter.input(at: 0)
            let width = input.shape.dimensions[2]
            let height = input.shape.dimensions[1]
            let resized = resize(image, width: width, height: height)
            var inputData = Data()
            for pixel in rgbPixels(resized) {
                appendInputValue(&inputData, value: Float(pixel), tensor: input)
            }
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()
            let output = try interpreter.output(at: 0)
            let logits = readOutput(output)
            var scores = softmax(logits)

            if let requestedCategory = call.getString("category"), ["Dog", "Cat"].contains(requestedCategory) {
                let total = scores.indices.reduce(Float(0)) { total, index in
                    if category(for: label(at: index)) == requestedCategory { return total + scores[index] }
                    scores[index] = 0
                    return total
                }
                if total > 0 { scores = scores.map { $0 / total } }
            }

            let indices = scores.indices.sorted { scores[$0] > scores[$1] }
            let bestIndex = indices.first ?? 0
            let name = label(at: bestIndex)
            let breedCategory = category(for: name)
            let predictions = indices.prefix(10).map { index in
                ["classId": index, "taxonId": NSNull(), "label": label(at: index), "category": category(for: label(at: index)), "confidence": scores[index]] as [String: Any]
            }
            if breedCategory != "Dog" && breedCategory != "Cat" {
                call.resolve(["classId": bestIndex, "taxonId": NSNull(), "name": "Unknown", "category": "Unknown", "confidence": scores[bestIndex], "predictions": predictions])
            } else {
                call.resolve(["classId": bestIndex, "taxonId": NSNull(), "name": name, "category": breedCategory, "confidence": scores[bestIndex], "predictions": predictions])
            }
        } catch {
            call.reject("Animal AI classification failed", error)
        }
    }

    private func loadLabels() throws -> [String] {
        guard let url = Bundle.main.url(forResource: "breed_labels", withExtension: "txt", subdirectory: "Models") else {
            throw NSError(domain: "AnimalAI", code: 1, userInfo: [NSLocalizedDescriptionKey: "breed_labels.txt is missing"])
        }
        return try String(contentsOf: url).split(whereSeparator: \.isNewline).map(String.init)
    }

    private func decodeImage(_ value: String) -> UIImage? {
        let encoded = value.components(separatedBy: ",").last ?? value
        guard let data = Data(base64Encoded: encoded, options: .ignoreUnknownCharacters) else { return nil }
        return UIImage(data: data)
    }

    private func resize(_ image: UIImage, width: Int, height: Int) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        return UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format).image { _ in
            image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        }
    }

    private func rgbPixels(_ image: UIImage) -> [UInt8] {
        guard let cgImage = image.cgImage else { return [] }
        var pixels = [UInt8](repeating: 0, count: cgImage.width * cgImage.height * 4)
        let context = CGContext(data: &pixels, width: cgImage.width, height: cgImage.height, bitsPerComponent: 8, bytesPerRow: cgImage.width * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
        var rgb: [UInt8] = []
        for index in stride(from: 0, to: pixels.count, by: 4) { rgb += [pixels[index], pixels[index + 1], pixels[index + 2]] }
        return rgb
    }

    private func appendInputValue(_ data: inout Data, value: Float, tensor: Tensor) {
        switch tensor.dataType {
        case .float32:
            var value = value
            data.append(Data(bytes: &value, count: 4))
        case .uint8:
            data.append(UInt8(max(0, min(255, Int(value)))) )
        default:
            data.append(UInt8(max(0, min(255, Int(value)))) )
        }
    }

    private func readOutput(_ tensor: Tensor) -> [Float] {
        tensor.data.withUnsafeBytes { bytes in
            switch tensor.dataType {
            case .float32: return stride(from: 0, to: bytes.count, by: 4).map { bytes.load(fromByteOffset: $0, as: Float.self) }
            default: return Array(repeating: 0, count: tensor.shape.dimensions.reduce(1, *) )
            }
        }
    }

    private func softmax(_ values: [Float]) -> [Float] {
        guard let maxValue = values.max() else { return [] }
        let exponentials = values.map { exp($0 - maxValue) }
        let total = exponentials.reduce(0, +)
        return total == 0 ? exponentials : exponentials.map { $0 / total }
    }

    private func label(at index: Int) -> String { index < labels.count ? labels[index] : "Unknown animal" }

    private func category(for label: String) -> String {
        if dogBreeds.contains(label) { return "Dog" }
        if catBreeds.contains(label) { return "Cat" }
        let normalized = label.lowercased()
        if normalized.contains("dog") { return "Dog" }
        if normalized.contains("cat") { return "Cat" }
        return "Unknown"
    }
}
