import { readFile, writeFile } from "node:fs/promises";

const packagePath = "ios/App/CapApp-SPM/Package.swift";
let packageText = await readFile(packagePath, "utf8");

if (!packageText.includes("tflite-swift")) {
  packageText = packageText.replace(
    '        .package(name: "CapacitorStatusBar", path: "../../../node_modules/@capacitor/status-bar")',
    '        .package(name: "CapacitorStatusBar", path: "../../../node_modules/@capacitor/status-bar"),\n        .package(url: "https://github.com/kewlbear/TensorFlowLiteSwift", branch: "master")',
  );
  packageText = packageText.replace(
    '                .product(name: "CapacitorStatusBar", package: "CapacitorStatusBar")',
    '                .product(name: "CapacitorStatusBar", package: "CapacitorStatusBar"),\n                .product(name: "TensorFlowLiteSwift", package: "TensorFlowLiteSwift")',
  );
  await writeFile(packagePath, packageText);
}
