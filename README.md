# Animal Finder

An Ionic + Capacitor app that identifies dog and cat breeds from a photo using an on-device TensorFlow Lite model.

## What it does

- Lets users take a photo or upload an image
- Runs breed recognition locally on the device
- Focuses on dog and cat breeds only
- Returns a low-confidence warning when the prediction is not reliable enough
- Opens the matched database record when a confident breed match is found

## Tech stack

- React + Ionic
- Capacitor for Android packaging
- TensorFlow Lite model for on-device pet breed recognition
- SQLite animal database for breed details

## Local development

1. Install dependencies:
   npm install
2. Start the app in dev mode:
   npm run dev
3. Build the web app:
   npm run build

## Android build

1. Sync Capacitor:
   npx cap sync android
2. Open Android Studio or build from the command line:
   cd android && ./gradlew assembleDebug

## Model focus

This app is intentionally limited to cat and dog breeds. If a photo is not recognized as a cat or dog breed with enough confidence, the app returns an unknown breed result instead of a general animal classification.

## Notes

- The app is designed for offline use after install.
- The breed model is loaded from the native app bundle.
- The database and model assets are bundled for Android builds.

## iOS build

The iOS target requires macOS and Xcode. Install dependencies, download the latest model artifact, and sync with:

npm run ios:sync

Open `ios/App/App.xcodeproj` in Xcode to configure the Apple development team and signing. The GitHub Actions workflow currently builds an unsigned `.xcarchive`, which does not require Apple Developer secrets. A signed `.ipa` for TestFlight or the App Store requires an Apple Developer account, certificate, and provisioning profile.
