import { registerPlugin } from "@capacitor/core";

export interface AnimalPrediction {
  classId: number;
  taxonId: number | null;
  name: string;
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

export interface AnimalClassifyResult {
  classId: number;
  taxonId: number | null;
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
    taxonId: number | null;
    predictions: AnimalPrediction[];
  }>;
}

export const AnimalAI = registerPlugin<AnimalAiPlugin>("AnimalAI");
