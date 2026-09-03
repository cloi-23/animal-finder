import { readFile, writeFile } from "node:fs/promises";

const packagePath = "ios/App/CapApp-SPM/Package.swift";
let packageText = await readFile(packagePath, "utf8");

if (!packageText.includes("tflite-swift")) {
  packageText = packageText.replace(
    '        .package(name: "CapacitorStatusBar", path: "../../../node_modules/@capacitor/status-bar")',
    '        .package(name: "CapacitorStatusBar", path: "../../../node_modules/@capacitor/status-bar"),\n        .package(name: "tflite-swift", url: "https://github.com/kewlbear/TensorFlowLiteSwift", exact: "2.17.0")',
  );
  packageText = packageText.replace(
    '                .product(name: "CapacitorStatusBar", package: "CapacitorStatusBar")',
    '                .product(name: "CapacitorStatusBar", package: "CapacitorStatusBar"),\n                .product(name: "TensorFlowLiteSwift", package: "tflite-swift")',
  );
  await writeFile(packagePath, packageText);
}
