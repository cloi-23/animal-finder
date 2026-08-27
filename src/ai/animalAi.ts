import { registerPlugin } from "@capacitor/core";

export interface AnimalPrediction {
  classId: number;
  name: string;
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

export interface AnimalClassifyResult {
  classId: number;
  name: string;
  category: string;
  confidence: number;
  predictions: AnimalPrediction[];
}

export interface AnimalAiPlugin {
  modelInfo(): Promise<AnimalModelInfo>;

  classify(options: { image: string }): Promise<{
    category: string;
    name: string;
    confidence: number;
    classId: number;
  }>;
}

export const AnimalAI = registerPlugin<AnimalAiPlugin>("AnimalAI");
