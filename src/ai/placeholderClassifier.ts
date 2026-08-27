import {
  AnimalClassifier,
  AnimalPrediction,
} from "./animalClassifier";

export class PlaceholderClassifier implements AnimalClassifier {
  async initialize(): Promise<void> {
    console.log("Animal classifier initialized");
  }

  async classify(
    _image: HTMLImageElement
  ): Promise<AnimalPrediction[]> {
    // Temporary result until we install the real animal model.
    return [
      {
        label: "Unknown animal",
        confidence: 0,
      },
    ];
  }

  dispose(): void {
    console.log("Animal classifier disposed");
  }
}
