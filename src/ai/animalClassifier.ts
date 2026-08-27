export interface AnimalPrediction {
  label: string;
  confidence: number;
}

export interface AnimalClassifier {
  initialize(): Promise<void>;

  classify(
    image: HTMLImageElement
  ): Promise<AnimalPrediction[]>;

  dispose(): void;
}
