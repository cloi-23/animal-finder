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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@CapacitorPlugin(name = "AnimalAI")
public class AnimalAiPlugin extends Plugin {

    private static final String MODEL_PATH = "camera/inat_vision.tflite";

    private Interpreter interpreter;
    private final List<String> labels = new ArrayList<>();
    private final Map<Integer, Integer> taxonIds = new HashMap<>();

    @PluginMethod
    public void modelInfo(PluginCall call) {
        if (interpreter == null) {
            System.out.println("ANIMAL AI MODEL INFO: MODEL NOT LOADED");
            call.reject("Animal AI model is not loaded");
            return;
        }

        try {
            System.out.println("========== ANIMAL AI MODEL INFO ==========");

            System.out.println(
                "Input tensor count: " +
                interpreter.getInputTensorCount()
            );

            for (int i = 0; i < interpreter.getInputTensorCount(); i++) {
                org.tensorflow.lite.Tensor tensor =
                    interpreter.getInputTensor(i);

                System.out.println(
                    "INPUT " + i +
                    " name=" + tensor.name() +
                    " type=" + tensor.dataType() +
                    " shape=" + Arrays.toString(tensor.shape())
                );
            }

            System.out.println(
                "Output tensor count: " +
                interpreter.getOutputTensorCount()
            );

            for (int i = 0; i < interpreter.getOutputTensorCount(); i++) {
                org.tensorflow.lite.Tensor tensor =
                    interpreter.getOutputTensor(i);

                System.out.println(
                    "OUTPUT " + i +
                    " name=" + tensor.name() +
                    " type=" + tensor.dataType() +
                    " shape=" + Arrays.toString(tensor.shape())
                );
            }

            System.out.println("==========================================");

            JSObject info = new JSObject();
            info.put("inputCount", interpreter.getInputTensorCount());
            info.put("outputCount", interpreter.getOutputTensorCount());
            call.resolve(info);

        } catch (Exception e) {
            e.printStackTrace();
            call.reject("Unable to inspect MobileNet model", e);
        }
    }

    @Override
    public void load() {
        super.load();

        try {
            InputStream input =
                getContext().getAssets().open(
                    MODEL_PATH
                );

            byte[] modelBytes = readAllBytes(input);

            ByteBuffer modelBuffer =
                ByteBuffer.allocateDirect(modelBytes.length)
                    .order(ByteOrder.nativeOrder());

            modelBuffer.put(modelBytes);
            modelBuffer.rewind();

            interpreter = new Interpreter(modelBuffer);

            loadTaxonomy();

            System.out.println(
                "Animal AI iNaturalist model loaded. Labels: "
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

            org.tensorflow.lite.Tensor inputTensor = interpreter.getInputTensor(0);
            int[] inputShape = inputTensor.shape();
            int imageHeight = inputShape[1];
            int imageWidth = inputShape[2];

            resized = Bitmap.createScaledBitmap(
                bitmap,
                imageWidth,
                imageHeight,
                true
            );

            int inputElements = imageHeight * imageWidth * 3;
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(
                inputElements * bytesPerElement(inputTensor.dataType())
            ).order(ByteOrder.nativeOrder());

            int[] pixels = new int[imageElements(imageWidth, imageHeight)];

            resized.getPixels(
                pixels,
                0,
                imageWidth,
                0,
                0,
                imageWidth,
                imageHeight
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

                putInputValue(inputBuffer, inputTensor, red);
                putInputValue(inputBuffer, inputTensor, green);
                putInputValue(inputBuffer, inputTensor, blue);
            }

            inputBuffer.rewind();

            org.tensorflow.lite.Tensor outputTensor = interpreter.getOutputTensor(0);
            int outputElements = tensorElementCount(outputTensor.shape());
            ByteBuffer outputBuffer = ByteBuffer.allocateDirect(
                outputElements * bytesPerElement(outputTensor.dataType())
            ).order(ByteOrder.nativeOrder());

            interpreter.run(inputBuffer, outputBuffer);
            outputBuffer.rewind();

            /*
             * The model output is treated as logits.
             *
             * Convert logits into proper probabilities
             * using softmax.
             */
            float[] logits = new float[outputElements];
            for (int i = 0; i < outputElements; i++) {
                logits[i] = readOutputValue(outputBuffer, outputTensor);
            }
            float[] scores = softmax(logits);

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
            result.put("taxonId", taxonIds.get(bestIndex));
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
                prediction.put("taxonId", taxonIds.get(classId));
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
                        + " probability="
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
                    + " probability="
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

    /*
     * Converts model logits into probabilities.
     *
     * The largest logit is subtracted first for numerical stability.
     *
     * Output values are between 0 and 1 and sum to approximately 1.
     */
    private float[] softmax(float[] logits) {
        float maxLogit = Float.NEGATIVE_INFINITY;

        for (float logit : logits) {
            if (logit > maxLogit) {
                maxLogit = logit;
            }
        }

        float[] probabilities = new float[logits.length];

        double sum = 0.0;

        for (int i = 0; i < logits.length; i++) {
            double exp = Math.exp(logits[i] - maxLogit);

            probabilities[i] = (float) exp;

            sum += exp;
        }

        if (sum == 0.0 || Double.isNaN(sum)) {
            return probabilities;
        }

        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= (float) sum;
        }

        return probabilities;
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

    private void loadTaxonomy() throws Exception {
        labels.clear();
        taxonIds.clear();

        InputStream input =
            getContext().getAssets().open(
                "camera/taxonomy.csv"
            );

        BufferedReader reader =
            new BufferedReader(
                new InputStreamReader(input)
            );

        String line;
        boolean header = true;

        while ((line = reader.readLine()) != null) {
            if (header) {
                header = false;
                continue;
            }

            String[] columns = line.split(",", -1);
            if (columns.length < 8 || columns[3].isEmpty()) continue;

            int taxonId = Integer.parseInt(columns[1]);
            int classId = Integer.parseInt(columns[3]);
            taxonIds.put(classId, taxonId);
            while (labels.size() <= classId) labels.add("");
            labels.set(classId, columns[7]);
        }

        reader.close();
    }

    private int imageElements(int width, int height) {
        return width * height;
    }

    private int tensorElementCount(int[] shape) {
        int count = 1;
        for (int dimension : shape) count *= dimension;
        return count;
    }

    private int bytesPerElement(org.tensorflow.lite.DataType dataType) {
        if (dataType == org.tensorflow.lite.DataType.FLOAT32) return 4;
        if (dataType == org.tensorflow.lite.DataType.INT32) return 4;
        return 1;
    }

    private void putInputValue(
        ByteBuffer buffer,
        org.tensorflow.lite.Tensor tensor,
        float value
    ) {
        if (tensor.dataType() == org.tensorflow.lite.DataType.FLOAT32) {
            buffer.putFloat(value);
            return;
        }

        org.tensorflow.lite.Tensor.QuantizationParams quantization = tensor.quantizationParams();
        int quantized = Math.round(value / quantization.getScale()) + quantization.getZeroPoint();
        if (tensor.dataType() == org.tensorflow.lite.DataType.UINT8) {
            buffer.put((byte) Math.max(0, Math.min(255, quantized)));
        } else {
            buffer.put((byte) Math.max(-128, Math.min(127, quantized)));
        }
    }

    private float readOutputValue(
        ByteBuffer buffer,
        org.tensorflow.lite.Tensor tensor
    ) {
        if (tensor.dataType() == org.tensorflow.lite.DataType.FLOAT32) {
            return buffer.getFloat();
        }

        int raw = buffer.get();
        if (tensor.dataType() == org.tensorflow.lite.DataType.UINT8) raw &= 0xFF;
        org.tensorflow.lite.Tensor.QuantizationParams quantization = tensor.quantizationParams();
        return (raw - quantization.getZeroPoint()) * quantization.getScale();
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
