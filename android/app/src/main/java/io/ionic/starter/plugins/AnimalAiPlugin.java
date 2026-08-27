package io.ionic.starter.plugins;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

@CapacitorPlugin(name = "AnimalAI")
public class AnimalAiPlugin extends Plugin {

    private Interpreter interpreter;

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

            System.out.println("Animal AI TFLite model loaded");

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

        call.reject(
            "TFLite model loaded successfully, but image preprocessing/output mapping is not implemented yet"
        );
    }

    private byte[] readAllBytes(InputStream input) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();

        byte[] buffer = new byte[8192];
        int length;

        while ((length = input.read(buffer)) != -1) {
            output.write(buffer, 0, length);
        }

        input.close();

        return output.toByteArray();
    }
}
