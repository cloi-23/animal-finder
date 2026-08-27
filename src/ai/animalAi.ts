import { registerPlugin } from "@capacitor/core";

export interface AnimalAiPlugin {
  modelInfo(): Promise<{
    inputCount: number;
    outputCount: number;
    inputs: Array<{
      index: number;
      name: string;
      type: string;
      shape: number[];
    }>;
    outputs: Array<{
      index: number;
      name: string;
      type: string;
      shape: number[];
    }>;
  }>;

  classify(options: { image: string }): Promise<{
    predictions: Array<{
      label: string;
      confidence: number;
    }>;
  }>;
}

export const AnimalAI = registerPlugin<AnimalAiPlugin>("AnimalAI");
