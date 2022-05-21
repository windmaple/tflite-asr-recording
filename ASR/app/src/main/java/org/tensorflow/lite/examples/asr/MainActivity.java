package org.tensorflow.lite.examples.asr;

import static android.content.res.AssetManager.ACCESS_BUFFER;
import static android.os.ParcelFileDescriptor.MODE_READ_ONLY;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.google.common.io.ByteStreams;

import com.jlibrosa.audio.JLibrosa;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.audio.TensorAudio;
import org.tensorflow.lite.task.audio.classifier.AudioClassifier;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private MappedByteBuffer tfLiteModel;
    private Interpreter tfLite;
    private final static int REQUEST_RECORD_AUDIO = 17;
    private Button recordButton;
    private TextView resultTextview;
    private final static int AUDIO_LEN_IN_SECOND = 10;
    private final static int SAMPLE_RATE = 16000;
    private final static int RECORDING_LENGTH = SAMPLE_RATE * AUDIO_LEN_IN_SECOND;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        String tfliteFile = "conformer.tflite";
        int defaultSampleRate = -1;		//-1 value implies the method to use default sample rate
        int defaultAudioDuration = -1;	//-1 value implies the method to process complete audio duration

        recordButton = findViewById(R.id.record);
        resultTextview = findViewById(R.id.result);
        recordButton.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {

                try {

                    int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
                    AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.DEFAULT, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                            bufferSize);

                    if (record.getState() != AudioRecord.STATE_INITIALIZED) {
                        Log.e("ASR", "Audio Record can't initialize!");
                        return;
                    }

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN) {
                        if (android.media.audiofx.NoiseSuppressor.isAvailable()) {
                            android.media.audiofx.NoiseSuppressor noiseSuppressor = android.media.audiofx.NoiseSuppressor
                                    .create(record.getAudioSessionId());
                            if (noiseSuppressor != null) {
                                noiseSuppressor.setEnabled(true);
                            }
                        }
                    }
                    record.startRecording();

                    long shortsRead = 0;
                    int recordingOffset = 0;
                    short[] audioBuffer = new short[bufferSize / 2];
                    short[] recordingBuffer = new short[RECORDING_LENGTH];

                    while (shortsRead < RECORDING_LENGTH) {
                        int numberOfShort = record.read(audioBuffer, 0, audioBuffer.length);
                        shortsRead += numberOfShort;
                        System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, numberOfShort);
                        recordingOffset += numberOfShort;
                    }

                    record.stop();
                    record.release();

                    float audioFeatureValues[] = new float[RECORDING_LENGTH];
                    for (int i = 0; i < RECORDING_LENGTH; ++i) {
                        audioFeatureValues[i] = recordingBuffer[i] / (float)Short.MAX_VALUE;
                    }

                    Object[] inputArray = {audioFeatureValues};
                    IntBuffer outputBuffer = IntBuffer.allocate(20000);

                    Map<Integer, Object> outputMap = new HashMap<>();
                    outputMap.put(0, outputBuffer);

                    tfLiteModel = loadModelFile(getAssets(), tfliteFile);
                    Interpreter.Options tfLiteOptions = new Interpreter.Options();
                    tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
                    tfLite.resizeInput(0, new int[] {audioFeatureValues.length});

                    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

                    int size = tfLite.getOutputTensor(0).shape()[0];
                    int[] outputArray = new int[size];
                    outputBuffer.rewind();
                    outputBuffer.get(outputArray);
                    StringBuilder finalResult = new StringBuilder();
                    for (int i=0; i < size; i++) {
                        char c = (char) outputArray[i];
                        if (outputArray[i] != 0) {
                            finalResult.append((char) outputArray[i]);
                        }
                    }
                    resultTextview.setText(finalResult.toString());

                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });

        requestMicrophonePermission();
    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void requestMicrophonePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(
                    new String[]{android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
        }
    }
}
