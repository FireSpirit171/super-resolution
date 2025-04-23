package com.example.superresolutionapp

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfDouble
import org.opencv.core.Core
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.Executors

import org.opencv.core.Size as OpenCVSize
import org.opencv.core.Scalar

class MainActivity : ComponentActivity() {
    private lateinit var superResolutionNet: Net
    private val TAG = "SuperResolutionApp"

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            Log.d(TAG, "Camera permission granted")
            startCamera()
        } else {
            Log.e(TAG, "Camera permission denied")
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "onCreate called")

        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV loaded successfully")
            Toast.makeText(this, "OpenCV loaded successfully", Toast.LENGTH_SHORT).show()
        } else {
            Log.e(TAG, "OpenCV failed to load")
            Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_LONG).show()
            return
        }

        try {
            val modelStream = assets.open("FSRCNN-small_x2_no_unsqueeze.onnx")
            val modelBytes = modelStream.readBytes()
            Log.d(TAG, "Model size: ${modelBytes.size} bytes")
            modelStream.close()
            val modelMat = MatOfByte(*modelBytes)
            superResolutionNet = Dnn.readNetFromONNX(modelMat)
            modelMat.release()
            Log.d(TAG, "FSRCNN loaded successfully")
            Toast.makeText(this, "FSRCNN loaded successfully", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load FSRCNN: ${e.message}", e)
            Toast.makeText(this, "Failed to load FSRCNN: ${e.message}", Toast.LENGTH_LONG).show()
            return
        }

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            Log.d(TAG, "Camera permission already granted")
            startCamera()
        } else {
            Log.d(TAG, "Requesting camera permission")
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        try {
            setContent {
                val blurScoreState = remember { mutableStateOf("Blur Score: N/A") }
                val isSuperResolutionActive = remember { mutableStateOf(false) }
                val superResolvedBitmap = remember { mutableStateOf<Bitmap?>(null) }
                CameraPreview(
                    blurScoreState = blurScoreState,
                    isSuperResolutionActive = isSuperResolutionActive,
                    superResolvedBitmap = superResolvedBitmap
                )
            }
            Log.d(TAG, "Camera started successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start camera: ${e.message}", e)
            Toast.makeText(this, "Failed to start camera: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    fun calculateBlurScore(frame: Mat): Double {
        try {
            if (frame.empty()) {
                Log.e(TAG, "Input frame is empty in calculateBlurScore")
                return 0.0
            }
            val gray = Mat()
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGBA2GRAY)
            val laplacian = Mat()
            Imgproc.Laplacian(gray, laplacian, CvType.CV_64F)
            val mean = MatOfDouble()
            val stddev = MatOfDouble()
            Core.meanStdDev(laplacian, mean, stddev)
            val variance = stddev.toArray()[0] * stddev.toArray()[0]
            gray.release()
            laplacian.release()
            mean.release()
            stddev.release()
            Log.d(TAG, "Calculated Blur Score: $variance")
            return variance
        } catch (e: Exception) {
            Log.e(TAG, "Error in calculateBlurScore: ${e.message}", e)
            return 0.0
        }
    }

    fun applySuperResolution(frame: Mat): Mat {
        try {
            if (frame.empty()) {
                Log.e(TAG, "Input frame is empty in applySuperResolution")
                return Mat()
            }

            Log.d(TAG, "Input frame size: ${frame.cols()}x${frame.rows()}")

            // Поворачиваем изображение, если нужно
            val rotatedMat = Mat()
            if (frame.width() > frame.height()) {
                Core.rotate(frame, rotatedMat, Core.ROTATE_90_CLOCKWISE)
            } else {
                frame.copyTo(rotatedMat)
            }

            // Преобразуем в YUV и берём Y-канал
            val yuvMat = Mat()
            Imgproc.cvtColor(rotatedMat, yuvMat, Imgproc.COLOR_RGBA2RGB)
            Imgproc.cvtColor(yuvMat, yuvMat, Imgproc.COLOR_RGB2YUV)
            val yuvChannels = mutableListOf<Mat>()
            Core.split(yuvMat, yuvChannels)
            val yMat = yuvChannels[0] // Y-канал (монохромный)
            val uMat = yuvChannels[1] // U-канал
            val vMat = yuvChannels[2] // V-канал

            // Ресайзим Y-канал до 320x240
            val resizedYMat = Mat()
            Imgproc.resize(yMat, resizedYMat, OpenCVSize(320.0, 240.0))
            Log.d(TAG, "Resized Y-channel to ${resizedYMat.cols()}x${resizedYMat.rows()}")

            // Нормализуем и конвертируем в CV_32F
            val inputMat = Mat()
            resizedYMat.convertTo(inputMat, CvType.CV_32F, 1.0 / 255.0)

            // Создаём blob для сети: [1, 1, 240, 320]
            val blob = Dnn.blobFromImage(inputMat, 1.0, OpenCVSize(320.0, 240.0), Scalar(0.0), false, false, CvType.CV_32F)
            val blobShape = IntArray(blob.dims()) { blob.size(it).toInt() }
            Log.d(TAG, "Blob shape: ${blobShape.joinToString(", ", "[", "]")}")

            // Передаем в модель
            superResolutionNet.setInput(blob)
            val output = superResolutionNet.forward()
            val outputShape = IntArray(output.dims()) { output.size(it).toInt() }
            Log.d(TAG, "FSRCNN output shape: ${outputShape.joinToString(", ", "[", "]")}")

            // Проверка диапазона значений
            // output имеет размер [1, 1, 480, 640], преобразуем в 2D матрицу
            val output2D = output.reshape(1, output.size(2).toInt()) // [480, 640]
            val minMax = Core.minMaxLoc(output2D)
            Log.d(TAG, "Output range: min=${minMax.minVal}, max=${minMax.maxVal}")
            if (minMax.minVal.isNaN() || minMax.maxVal.isNaN()) {
                Log.e(TAG, "Output contains NaN values")
                output2D.release()
                return Mat()
            }
            output2D.release()

            // Нормализация выхода в [0, 1]
            // Преобразуем в 2D перед нормализацией
            val normalized2D = Mat()
            val temp2D = output.reshape(1, output.size(2).toInt()) // [480, 640]
            Core.normalize(temp2D, normalized2D, 0.0, 1.0, Core.NORM_MINMAX, CvType.CV_32F)
            Log.d(TAG, "Normalized 2D size: ${normalized2D.cols()}x${normalized2D.rows()}")

            // Преобразуем в uint8
            val outputMat2D = Mat()
            normalized2D.convertTo(outputMat2D, CvType.CV_8U, 255.0)
            Log.d(TAG, "Output Mat 2D size: ${outputMat2D.cols()}x${outputMat2D.rows()}")

            // Преобразуем обратно в 4D [1, 1, 480, 640]
            val outputMat = outputMat2D.reshape(1, intArrayOf(1, 1, output.size(2).toInt(), output.size(3).toInt()))
            val outputMatShape = IntArray(outputMat.dims()) { outputMat.size(it).toInt() }
            Log.d(TAG, "Output Mat shape: ${outputMatShape.joinToString(", ", "[", "]")}")

            // Ожидаемый размер выхода: 640x480 (240*2, 320*2)
            if (outputMatShape[3] != 640 || outputMatShape[2] != 480) {
                Log.e(TAG, "Unexpected output shape: ${outputMatShape.joinToString(", ", "[", "]")}")
                return Mat()
            }

            // Ресайзим U и V каналы до 640x480
            val resizedUMat = Mat()
            val resizedVMat = Mat()
            Imgproc.resize(uMat, resizedUMat, OpenCVSize(640.0, 480.0))
            Imgproc.resize(vMat, resizedVMat, OpenCVSize(640.0, 480.0))

            // Объединяем YUV-каналы
            val yuvOutput = Mat()
            val yuvChannelsOutput = listOf(outputMat2D, resizedUMat, resizedVMat)
            Core.merge(yuvChannelsOutput, yuvOutput)

            // Преобразуем YUV обратно в RGB
            val rgbOutput = Mat()
            Imgproc.cvtColor(yuvOutput, rgbOutput, Imgproc.COLOR_YUV2RGB)
            Imgproc.cvtColor(rgbOutput, rgbOutput, Imgproc.COLOR_RGB2RGBA)

            // Освобождаем ресурсы
            rotatedMat.release()
            yuvMat.release()
            yMat.release()
            uMat.release()
            vMat.release()
            resizedYMat.release()
            inputMat.release()
            blob.release()
            output.release()
            temp2D.release()
            normalized2D.release()
            outputMat.release()
            outputMat2D.release()
            resizedUMat.release()
            resizedVMat.release()
            yuvOutput.release()

            if (rgbOutput.empty()) {
                Log.e(TAG, "Output Mat is empty after applySuperResolution")
                return Mat()
            }

            Log.d(TAG, "Super-Resolution output ready: ${rgbOutput.cols()}x${rgbOutput.rows()}")
            return rgbOutput
        } catch (e: Exception) {
            Log.e(TAG, "Error in applySuperResolution: ${e.message}", e)
            return Mat()
        }
    }
}

@Composable
fun CameraPreview(
    blurScoreState: MutableState<String>,
    isSuperResolutionActive: MutableState<Boolean>,
    superResolvedBitmap: MutableState<Bitmap?>
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val configuration = LocalConfiguration.current
    val isPortrait = configuration.orientation == Configuration.ORIENTATION_PORTRAIT
    val TAG = "CameraPreview"
    val originalBitmap = remember { mutableStateOf<Bitmap?>(null) }
    val rotatedSuperResolvedBitmap = remember { mutableStateOf<Bitmap?>(null) }

    Box(modifier = Modifier.fillMaxSize()) {
        if (isPortrait) {
            // Вертикальное деление (сверху Original, снизу Super-Resolution)
            Column(modifier = Modifier.fillMaxSize()) {
                Box(modifier = Modifier
                    .weight(1f)
                    .fillMaxSize()) {
                    AndroidView(
                        modifier = Modifier.fillMaxSize(),
                        factory = { ctx ->
                            val previewView = PreviewView(ctx)
                            val executor = Executors.newSingleThreadExecutor()
                            val mainActivity = ctx as? MainActivity

                            cameraProviderFuture.addListener({
                                try {
                                    val cameraProvider = cameraProviderFuture.get()
                                    val preview = Preview.Builder()
                                        .setTargetRotation(previewView.display.rotation)
                                        .build().also {
                                            it.setSurfaceProvider(previewView.surfaceProvider)
                                        }
                                    val imageAnalysis = ImageAnalysis.Builder()
                                        .setTargetResolution(Size(640, 480))
                                        .setTargetRotation(previewView.display.rotation)
                                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                        .build()

                                    var frameCount = 0
                                    imageAnalysis.setAnalyzer(executor) { imageProxy ->
                                        try {
                                            frameCount++
                                            if (frameCount % 5 == 0) {
                                                val bitmap = imageProxy.toBitmap()
                                                originalBitmap.value = bitmap
                                                val mat = Mat()
                                                Utils.bitmapToMat(bitmap, mat)

                                                val blurScore = mainActivity?.calculateBlurScore(mat) ?: 0.0
                                                blurScoreState.value = "Blur Score: %.2f".format(blurScore)
                                                Log.d(TAG, "Blur Score: $blurScore")

                                                if (blurScore < 1000) {
                                                    val superResolvedMat = mainActivity?.applySuperResolution(mat)
                                                    superResolvedMat?.let { mat ->
                                                        if (mat.empty()) {
                                                            Log.e(TAG, "SuperResolvedMat is empty before bitmap creation")
                                                            isSuperResolutionActive.value = false
                                                            superResolvedBitmap.value = null
                                                            rotatedSuperResolvedBitmap.value = null
                                                        } else {
                                                            val superResolvedBitmapResult = Bitmap.createBitmap(
                                                                mat.cols(),
                                                                mat.rows(),
                                                                Bitmap.Config.ARGB_8888
                                                            )
                                                            Utils.matToBitmap(mat, superResolvedBitmapResult)
                                                            superResolvedBitmap.value = superResolvedBitmapResult

                                                            // Поворачиваем superResolvedBitmap в зависимости от ориентации
                                                            val matrix = Matrix()
                                                            if (isPortrait) {
                                                                matrix.postRotate(90f)
                                                            } else {
                                                                matrix.postRotate(-90f) // Поворот на 90 градусов влево
                                                            }
                                                            val rotatedBitmap = Bitmap.createBitmap(
                                                                superResolvedBitmapResult,
                                                                0,
                                                                0,
                                                                superResolvedBitmapResult.width,
                                                                superResolvedBitmapResult.height,
                                                                matrix,
                                                                true
                                                            )
                                                            rotatedSuperResolvedBitmap.value = rotatedBitmap

                                                            isSuperResolutionActive.value = true
                                                            Log.d(TAG, "Super-Resolution applied, bitmap size: ${superResolvedBitmapResult.width}x${superResolvedBitmapResult.height}")
                                                            mat.release()
                                                        }
                                                    } ?: run {
                                                        isSuperResolutionActive.value = false
                                                        superResolvedBitmap.value = null
                                                        rotatedSuperResolvedBitmap.value = null
                                                        Log.w(TAG, "Super-Resolution returned null")
                                                    }
                                                } else {
                                                    isSuperResolutionActive.value = false
                                                    superResolvedBitmap.value = null
                                                    rotatedSuperResolvedBitmap.value = null
                                                    Log.d(TAG, "Blur Score too high ($blurScore), skipping super-resolution")
                                                }

                                                mat.release()
                                            }
                                        } catch (e: Exception) {
                                            Log.e(TAG, "Error in image analysis: ${e.message}", e)
                                        } finally {
                                            imageProxy.close()
                                        }
                                    }

                                    val cameraSelector = CameraSelector.Builder()
                                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                                        .build()

                                    cameraProvider.unbindAll()
                                    cameraProvider.bindToLifecycle(
                                        lifecycleOwner,
                                        cameraSelector,
                                        preview,
                                        imageAnalysis
                                    )
                                    Log.d(TAG, "Camera bound to lifecycle")
                                } catch (e: Exception) {
                                    Log.e(TAG, "Error setting up camera: ${e.message}", e)
                                    (ctx as? MainActivity)?.runOnUiThread {
                                        Toast.makeText(ctx, "Camera setup failed: ${e.message}", Toast.LENGTH_LONG).show()
                                    }
                                }
                            }, ContextCompat.getMainExecutor(context))

                            previewView
                        }
                    )
                    Text(
                        text = "Original",
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(8.dp)
                    )
                    Text(
                        text = blurScoreState.value,
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(8.dp)
                    )
                }

                Box(modifier = Modifier
                    .weight(1f)
                    .fillMaxSize()) {
                    rotatedSuperResolvedBitmap.value?.let { bitmap ->
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = "Super-Resolved Image",
                            modifier = Modifier.fillMaxSize()
                        )
                    }
                    Text(
                        text = if (isSuperResolutionActive.value) "Super-Resolution" else "No SR Output",
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(8.dp)
                    )
                }
            }
        } else {
            // Горизонтальное деление (слева Original, справа Super-Resolution)
            Row(modifier = Modifier.fillMaxSize()) {
                Box(modifier = Modifier
                    .weight(1f)
                    .fillMaxSize()) {
                    AndroidView(
                        modifier = Modifier.fillMaxSize(),
                        factory = { ctx ->
                            val previewView = PreviewView(ctx)
                            val executor = Executors.newSingleThreadExecutor()
                            val mainActivity = ctx as? MainActivity

                            cameraProviderFuture.addListener({
                                try {
                                    val cameraProvider = cameraProviderFuture.get()
                                    val preview = Preview.Builder()
                                        .setTargetRotation(previewView.display.rotation)
                                        .build().also {
                                            it.setSurfaceProvider(previewView.surfaceProvider)
                                        }
                                    val imageAnalysis = ImageAnalysis.Builder()
                                        .setTargetResolution(Size(640, 480))
                                        .setTargetRotation(previewView.display.rotation)
                                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                        .build()

                                    var frameCount = 0
                                    imageAnalysis.setAnalyzer(executor) { imageProxy ->
                                        try {
                                            frameCount++
                                            if (frameCount % 5 == 0) {
                                                val bitmap = imageProxy.toBitmap()
                                                originalBitmap.value = bitmap
                                                val mat = Mat()
                                                Utils.bitmapToMat(bitmap, mat)

                                                val blurScore = mainActivity?.calculateBlurScore(mat) ?: 0.0
                                                blurScoreState.value = "Blur Score: %.2f".format(blurScore)
                                                Log.d(TAG, "Blur Score: $blurScore")

                                                if (blurScore < 1000) {
                                                    val superResolvedMat = mainActivity?.applySuperResolution(mat)
                                                    superResolvedMat?.let { mat ->
                                                        if (mat.empty()) {
                                                            Log.e(TAG, "SuperResolvedMat is empty before bitmap creation")
                                                            isSuperResolutionActive.value = false
                                                            superResolvedBitmap.value = null
                                                            rotatedSuperResolvedBitmap.value = null
                                                        } else {
                                                            val superResolvedBitmapResult = Bitmap.createBitmap(
                                                                mat.cols(),
                                                                mat.rows(),
                                                                Bitmap.Config.ARGB_8888
                                                            )
                                                            Utils.matToBitmap(mat, superResolvedBitmapResult)
                                                            superResolvedBitmap.value = superResolvedBitmapResult

                                                            // Поворачиваем superResolvedBitmap в зависимости от ориентации
                                                            val matrix = Matrix()
                                                            if (isPortrait) {
                                                                matrix.postRotate(90f)
                                                            } else {
                                                                matrix.postRotate(-90f) // Поворот на 90 градусов влево
                                                            }
                                                            val rotatedBitmap = Bitmap.createBitmap(
                                                                superResolvedBitmapResult,
                                                                0,
                                                                0,
                                                                superResolvedBitmapResult.width,
                                                                superResolvedBitmapResult.height,
                                                                matrix,
                                                                true
                                                            )
                                                            rotatedSuperResolvedBitmap.value = rotatedBitmap

                                                            isSuperResolutionActive.value = true
                                                            Log.d(TAG, "Super-Resolution applied, bitmap size: ${superResolvedBitmapResult.width}x${superResolvedBitmapResult.height}")
                                                            mat.release()
                                                        }
                                                    } ?: run {
                                                        isSuperResolutionActive.value = false
                                                        superResolvedBitmap.value = null
                                                        rotatedSuperResolvedBitmap.value = null
                                                        Log.w(TAG, "Super-Resolution returned null")
                                                    }
                                                } else {
                                                    isSuperResolutionActive.value = false
                                                    superResolvedBitmap.value = null
                                                    rotatedSuperResolvedBitmap.value = null
                                                    Log.d(TAG, "Blur Score too high ($blurScore), skipping super-resolution")
                                                }

                                                mat.release()
                                            }
                                        } catch (e: Exception) {
                                            Log.e(TAG, "Error in image analysis: ${e.message}", e)
                                        } finally {
                                            imageProxy.close()
                                        }
                                    }

                                    val cameraSelector = CameraSelector.Builder()
                                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                                        .build()

                                    cameraProvider.unbindAll()
                                    cameraProvider.bindToLifecycle(
                                        lifecycleOwner,
                                        cameraSelector,
                                        preview,
                                        imageAnalysis
                                    )
                                    Log.d(TAG, "Camera bound to lifecycle")
                                } catch (e: Exception) {
                                    Log.e(TAG, "Error setting up camera: ${e.message}", e)
                                    (ctx as? MainActivity)?.runOnUiThread {
                                        Toast.makeText(ctx, "Camera setup failed: ${e.message}", Toast.LENGTH_LONG).show()
                                    }
                                }
                            }, ContextCompat.getMainExecutor(context))

                            previewView
                        }
                    )
                    Text(
                        text = "Original",
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(8.dp)
                    )
                    Text(
                        text = blurScoreState.value,
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(8.dp)
                    )
                }

                Box(modifier = Modifier
                    .weight(1f)
                    .fillMaxSize()) {
                    rotatedSuperResolvedBitmap.value?.let { bitmap ->
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = "Super-Resolved Image",
                            modifier = Modifier.fillMaxSize()
                        )
                    }
                    Text(
                        text = if (isSuperResolutionActive.value) "Super-Resolution" else "No SR Output",
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(8.dp)
                    )
                }
            }
        }
    }
}

private fun ImageProxy.toBitmap(): Bitmap {
    try {
        val yBuffer: ByteBuffer = planes[0].buffer
        val uBuffer: ByteBuffer = planes[1].buffer
        val vBuffer: ByteBuffer = planes[2].buffer

        val ySize: Int = yBuffer.remaining()
        val uSize: Int = uBuffer.remaining()
        val vSize: Int = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    } catch (e: Exception) {
        Log.e("ImageProxy", "Error converting to bitmap: ${e.message}", e)
        throw e
    }
}