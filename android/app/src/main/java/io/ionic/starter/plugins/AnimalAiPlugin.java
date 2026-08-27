package io.ionic.starter.plugins;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;

import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@CapacitorPlugin(name = "AnimalAI")
public class AnimalAiPlugin extends Plugin {

    private static final int IMAGE_SIZE = 224;
    private static final int CLASS_COUNT = 1000;

    private Interpreter interpreter;
    private final List<String> labels = new ArrayList<>();

    @PluginMethod
    public void modelInfo(PluginCall call) {
        if (interpreter == null) {
            call.reject("Animal AI model is not loaded");
            return;
        }

        try {
            JSObject result = new JSObject();

            result.put("inputCount", interpreter.getInputTensorCount());
            result.put("outputCount", interpreter.getOutputTensorCount());

            JSArray inputs = new JSArray();

            for (int i = 0; i < interpreter.getInputTensorCount(); i++) {
                org.tensorflow.lite.Tensor tensor =
                    interpreter.getInputTensor(i);

                JSObject info = new JSObject();

                info.put("index", i);
                info.put("name", tensor.name());
                info.put("type", tensor.dataType().toString());
                info.put("shape", new JSArray(tensor.shape()));

                inputs.put(info);
            }

            JSArray outputs = new JSArray();

            for (int i = 0; i < interpreter.getOutputTensorCount(); i++) {
                org.tensorflow.lite.Tensor tensor =
                    interpreter.getOutputTensor(i);

                JSObject info = new JSObject();

                info.put("index", i);
                info.put("name", tensor.name());
                info.put("type", tensor.dataType().toString());
                info.put("shape", new JSArray(tensor.shape()));

                outputs.put(info);
            }

            result.put("inputs", inputs);
            result.put("outputs", outputs);

            call.resolve(result);

        } catch (Exception e) {
            call.reject("Unable to inspect MobileNet model", e);
        }
    }

    @Override
    public void load() {
        super.load();

        try {
            InputStream input =
                getContext().getAssets().open(
                    "camera/mobilenet_v3_small.tflite"
                );

            byte[] modelBytes = readAllBytes(input);

            ByteBuffer modelBuffer =
                ByteBuffer.allocateDirect(modelBytes.length)
                    .order(ByteOrder.nativeOrder());

            modelBuffer.put(modelBytes);
            modelBuffer.rewind();

            interpreter = new Interpreter(modelBuffer);

            loadLabels();

            System.out.println(
                "Animal AI MobileNet model loaded. Labels: "
                    + labels.size()
            );

        } catch (Exception e) {
            e.printStackTrace();
            interpreter = null;
        }
    }

    @PluginMethod
    public void classify(PluginCall call) {
        if (interpreter == null) {
            call.reject("Animal AI model is not loaded");
            return;
        }

        String imageData = call.getString("image");

        if (imageData == null || imageData.isEmpty()) {
            call.reject("No image supplied");
            return;
        }

        Bitmap bitmap = null;
        Bitmap resized = null;

        try {
            bitmap = decodeImage(imageData);

            if (bitmap == null) {
                call.reject("Unable to decode image");
                return;
            }

            resized = Bitmap.createScaledBitmap(
                bitmap,
                IMAGE_SIZE,
                IMAGE_SIZE,
                true
            );

            ByteBuffer inputBuffer =
                ByteBuffer.allocateDirect(
                    4 * IMAGE_SIZE * IMAGE_SIZE * 3
                ).order(ByteOrder.nativeOrder());

            int[] pixels = new int[IMAGE_SIZE * IMAGE_SIZE];

            resized.getPixels(
                pixels,
                0,
                IMAGE_SIZE,
                0,
                0,
                IMAGE_SIZE,
                IMAGE_SIZE
            );

            /*
             * MobileNetV3 input:
             *
             * RGB values are normalized from 0..255
             * to approximately -1..1.
             */
            for (int pixel : pixels) {
                float red = ((pixel >> 16) & 0xFF) / 127.5f - 1.0f;
                float green = ((pixel >> 8) & 0xFF) / 127.5f - 1.0f;
                float blue = (pixel & 0xFF) / 127.5f - 1.0f;

                inputBuffer.putFloat(red);
                inputBuffer.putFloat(green);
                inputBuffer.putFloat(blue);
            }

            inputBuffer.rewind();

            float[][] output = new float[1][CLASS_COUNT];

            interpreter.run(inputBuffer, output);

            float[] scores = output[0];

            int bestIndex = 0;
            float bestScore = scores[0];

            for (int i = 1; i < scores.length; i++) {
                if (scores[i] > bestScore) {
                    bestScore = scores[i];
                    bestIndex = i;
                }
            }

            String animalName = getLabel(bestIndex);

            if (animalName == null || animalName.isEmpty()) {
                animalName = "Unknown animal";
            }

            String category = getBroadCategory(animalName);

            JSObject result = new JSObject();

            result.put("classId", bestIndex);
            result.put("name", animalName);
            result.put("category", category);
            result.put("confidence", bestScore);

            JSArray predictions = new JSArray();

            Integer[] indices = new Integer[scores.length];

            for (int i = 0; i < scores.length; i++) {
                indices[i] = i;
            }

            Arrays.sort(
                indices,
                (a, b) -> Float.compare(scores[b], scores[a])
            );

            for (int i = 0; i < Math.min(10, indices.length); i++) {
                int classId = indices[i];
                float score = scores[classId];

                String label = getLabel(classId);

                if (label == null || label.isEmpty()) {
                    label = "Unknown";
                }

                JSObject prediction = new JSObject();

                prediction.put("classId", classId);
                prediction.put("label", label);
                prediction.put("category", getBroadCategory(label));
                prediction.put("confidence", score);

                predictions.put(prediction);

                System.out.println(
                    "Animal AI TOP "
                        + (i + 1)
                        + ": "
                        + label
                        + " classId="
                        + classId
                        + " score="
                        + score
                );
            }

            result.put("predictions", predictions);

            System.out.println(
                "Animal AI prediction: "
                    + animalName
                    + " category="
                    + category
                    + " classId="
                    + bestIndex
                    + " score="
                    + bestScore
            );

            call.resolve(result);

        } catch (Exception e) {
            e.printStackTrace();
            call.reject("Animal AI classification failed", e);

        } finally {
            if (bitmap != null && bitmap != resized) {
                bitmap.recycle();
            }

            if (resized != null) {
                resized.recycle();
            }
        }
    }

    private Bitmap decodeImage(String imageData) {
        try {
            String base64Data = imageData;

            if (base64Data.contains(",")) {
                base64Data =
                    base64Data.substring(
                        base64Data.indexOf(",") + 1
                    );
            }

            byte[] imageBytes =
                Base64.decode(
                    base64Data,
                    Base64.DEFAULT
                );

            return BitmapFactory.decodeByteArray(
                imageBytes,
                0,
                imageBytes.length
            );

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private void loadLabels() throws Exception {
        labels.clear();

        InputStream input =
            getContext().getAssets().open(
                "camera/imagenet_labels.txt"
            );

        BufferedReader reader =
            new BufferedReader(
                new InputStreamReader(input)
            );

        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();

            if (!line.isEmpty()) {
                labels.add(line);
            }
        }

        reader.close();

        /*
         * ImageNetLabels.txt has a background entry at index 0.
         * The MobileNet output is zero-based over the 1000 classes.
         *
         * If the file contains 1001 entries, remove the background.
         */
        if (labels.size() == 1001) {
            labels.remove(0);
        }

        if (labels.size() != CLASS_COUNT) {
            throw new Exception(
                "Expected "
                    + CLASS_COUNT
                    + " ImageNet labels but loaded "
                    + labels.size()
            );
        }
    }

    private String getLabel(int classId) {
        if (classId < 0 || classId >= labels.size()) {
            return "Unknown animal";
        }

        return labels.get(classId);
    }

    private String getBroadCategory(String label) {
        String value = label.toLowerCase();

        /*
         * Dogs
         */
        if (
            value.contains("dog") ||
            value.contains("husky") ||
            value.contains("retriever") ||
            value.contains("shepherd") ||
            value.contains("terrier") ||
            value.contains("spaniel") ||
            value.contains("poodle") ||
            value.contains("collie") ||
            value.contains("beagle") ||
            value.contains("boxer") ||
            value.contains("rottweiler") ||
            value.contains("chihuahua") ||
            value.contains("dalmatian") ||
            value.contains("doberman") ||
            value.contains("mastiff") ||
            value.contains("malamute") ||
            value.contains("samoyed") ||
            value.contains("pomeranian") ||
            value.contains("corgi") ||
            value.contains("great dane") ||
            value.contains("saint bernard") ||
            value.contains("newfoundland") ||
            value.contains("schipperke")
        ) {
            return "Dog";
        }

        /*
         * Cats
         */
        if (
            value.contains("cat") ||
            value.contains("tabby") ||
            value.contains("siamese") ||
            value.contains("persian")
        ) {
            return "Cat";
        }

        /*
         * Birds
         */
        if (
            value.contains("bird") ||
            value.contains("eagle") ||
            value.contains("owl") ||
            value.contains("hawk") ||
            value.contains("falcon") ||
            value.contains("parrot") ||
            value.contains("macaw") ||
            value.contains("cockatoo") ||
            value.contains("penguin") ||
            value.contains("robin") ||
            value.contains("sparrow") ||
            value.contains("finch") ||
            value.contains("jay") ||
            value.contains("crow") ||
            value.contains("raven") ||
            value.contains("hen") ||
            value.contains("chicken") ||
            value.contains("duck") ||
            value.contains("goose") ||
            value.contains("swan") ||
            value.contains("flamingo") ||
            value.contains("ostrich")
        ) {
            return "Bird";
        }

        /*
         * Horses
         */
        if (
            value.contains("horse") ||
            value.contains("pony") ||
            value.contains("zebra")
        ) {
            return "Horse";
        }

        /*
         * Cows
         */
        if (
            value.contains("cow") ||
            value.contains("ox") ||
            value.contains("bull")
        ) {
            return "Cattle";
        }

        /*
         * Pigs
         */
        if (
            value.contains("pig") ||
            value.contains("boar")
        ) {
            return "Pig";
        }

        /*
         * Sheep and goats
         */
        if (
            value.contains("sheep") ||
            value.contains("ram") ||
            value.contains("goat")
        ) {
            return "Sheep/Goat";
        }

        /*
         * Big cats
         */
        if (
            value.contains("lion") ||
            value.contains("tiger") ||
            value.contains("leopard") ||
            value.contains("jaguar") ||
            value.contains("cheetah")
        ) {
            return "Wild cat";
        }

        /*
         * Bears
         */
        if (value.contains("bear")) {
            return "Bear";
        }

        /*
         * Reptiles
         */
        if (
            value.contains("snake") ||
            value.contains("lizard") ||
            value.contains("gecko") ||
            value.contains("iguana") ||
            value.contains("turtle") ||
            value.contains("tortoise") ||
            value.contains("crocodile") ||
            value.contains("alligator") ||
            value.contains("chameleon")
        ) {
            return "Reptile";
        }

        /*
         * Amphibians
         */
        if (
            value.contains("frog") ||
            value.contains("toad") ||
            value.contains("salamander")
        ) {
            return "Amphibian";
        }

        /*
         * Fish
         */
        if (
            value.contains("fish") ||
            value.contains("shark") ||
            value.contains("ray") ||
            value.contains("eel") ||
            value.contains("trout") ||
            value.contains("salmon")
        ) {
            return "Fish";
        }

        /*
         * Insects
         */
        if (
            value.contains("butterfly") ||
            value.contains("moth") ||
            value.contains("bee") ||
            value.contains("wasp") ||
            value.contains("ant") ||
            value.contains("beetle") ||
            value.contains("dragonfly") ||
            value.contains("grasshopper") ||
            value.contains("cricket") ||
            value.contains("mosquito") ||
            value.contains("fly")
        ) {
            return "Insect";
        }

        /*
         * Spiders and other arachnids
         */
        if (
            value.contains("spider") ||
            value.contains("scorpion") ||
            value.contains("tick")
        ) {
            return "Arachnid";
        }

        return "Animal";
    }

    private byte[] readAllBytes(InputStream input) throws Exception {
        ByteArrayOutputStream output =
            new ByteArrayOutputStream();

        byte[] buffer = new byte[8192];
        int length;

        while ((length = input.read(buffer)) != -1) {
            output.write(buffer, 0, length);
        }

        input.close();

        return output.toByteArray();
    }
}
