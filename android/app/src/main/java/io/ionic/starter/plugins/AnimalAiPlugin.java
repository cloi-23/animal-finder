```java
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
import java.util.HashMap;
import java.util.Map;

@CapacitorPlugin(name = "AnimalAI")
public class AnimalAiPlugin extends Plugin {

    private static final int IMAGE_SIZE = 299;
    private static final int CLASS_COUNT = 507;

    private Interpreter interpreter;
    private final Map<Integer, String> taxonomyNames = new HashMap<>();

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
            call.reject("Unable to inspect TFLite model", e);
        }
    }

    @Override
    public void load() {
        super.load();

        try {
            InputStream input =
                getContext().getAssets().open(
                    "camera/INatVision_Small_2_fact256_8bit.tflite"
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
                "Animal AI TFLite model loaded. Taxonomy classes: "
                + taxonomyNames.size()
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

        try {
            Bitmap bitmap = decodeImage(imageData);

            if (bitmap == null) {
                call.reject("Unable to decode image");
                return;
            }

            Bitmap resized = Bitmap.createScaledBitmap(
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
             * Model input:
             *   [1, 299, 299, 3]
             *   FLOAT32
             *
             * Pixel values are converted from 0..255 to 0..1.
             */
            for (int pixel : pixels) {
                float red = ((pixel >> 16) & 0xFF) / 255.0f;
                float green = ((pixel >> 8) & 0xFF) / 255.0f;
                float blue = (pixel & 0xFF) / 255.0f;

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

            String animalName = taxonomyNames.get(bestIndex);

            if (animalName == null || animalName.isEmpty()) {
                animalName = "Unknown animal";
            }

            JSObject result = new JSObject();

            result.put("classId", bestIndex);
            result.put("name", animalName);
            result.put("confidence", bestScore);

            System.out.println(
                "Animal AI prediction: "
                + animalName
                + " classId="
                + bestIndex
                + " score="
                + bestScore
            );

            call.resolve(result);

            if (bitmap != resized) {
                bitmap.recycle();
            }

            resized.recycle();

        } catch (Exception e) {
            e.printStackTrace();
            call.reject("Animal AI classification failed", e);
        }
    }

    private Bitmap decodeImage(String imageData) {
        try {
            String base64Data = imageData;

            /*
             * Supports:
             *
             * data:image/jpeg;base64,...
             * data:image/png;base64,...
             * data:image/webp;base64,...
             *
             * as well as plain base64.
             */
            if (base64Data.contains(",")) {
                base64Data =
                    base64Data.substring(
                        base64Data.indexOf(",") + 1
                    );
            }

            byte[] imageBytes =
                Base64.decode(base64Data, Base64.DEFAULT);

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
        taxonomyNames.clear();

        InputStream input =
            getContext().getAssets().open(
                "camera/taxonomy.csv"
            );

        BufferedReader reader =
            new BufferedReader(
                new InputStreamReader(input)
            );

        String header = reader.readLine();

        if (header == null) {
            reader.close();
            throw new Exception("taxonomy.csv is empty");
        }

        String line;

        while ((line = reader.readLine()) != null) {
            String[] columns = line.split(",", -1);

            if (columns.length < 8) {
                continue;
            }

            String leafClassId = columns[3].trim();
            String name = columns[7].trim();

            if (leafClassId.isEmpty() || name.isEmpty()) {
                continue;
            }

            try {
                int classId = Integer.parseInt(leafClassId);

                if (classId >= 0 && classId < CLASS_COUNT) {
                    taxonomyNames.put(classId, name);
                }

            } catch (NumberFormatException ignored) {
                // Ignore malformed taxonomy rows.
            }
        }

        reader.close();

        if (taxonomyNames.size() != CLASS_COUNT) {
            throw new Exception(
                "Expected "
                    + CLASS_COUNT
                    + " taxonomy classes but loaded "
                    + taxonomyNames.size()
            );
        }
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
```
