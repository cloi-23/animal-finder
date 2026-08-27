import { registerPlugin } from "@capacitor/core";

export interface AnimalPrediction {
  label: string;
  confidence: number;
}

export interface AnimalModelInfo {
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
}

export interface AnimalClassificationResult {
  classId: number;
  name: string;
  confidence: number;
}

export interface AnimalAiPlugin {
  modelInfo(): Promise<AnimalModelInfo>;

  classify(options: { image: string }): Promise<AnimalClassificationResult>;
}

export const AnimalAI = registerPlugin<AnimalAiPlugin>("AnimalAI");
