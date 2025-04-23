package com.example.superresolutionapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
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
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Core
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Инициализация OpenCV
        if (OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV loaded successfully", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_LONG).show()
            return
        }

        // Проверка разрешения камеры
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        setContent {
            // Создаём состояние внутри Composable
            val blurScoreState = remember { mutableStateOf("Blur Score: N/A") }
            CameraPreview(blurScoreState = blurScoreState)
        }
    }

    // Убрали private, чтобы функция была доступна
    fun calculateBlurScore(frame: Mat): Double {
        // Преобразуем в оттенки серого
        val gray = Mat()
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGBA2GRAY)

        // Применяем оператор Лапласа
        val laplacian = Mat()
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F)

        // Вычисляем дисперсию
        val mean = MatOfDouble()
        val stddev = MatOfDouble()
        Core.meanStdDev(laplacian, mean, stddev)
        val variance = stddev.toArray()[0] * stddev.toArray()[0] // Вместо .pow(2)

        // Освобождаем память
        gray.release()
        laplacian.release()
        mean.release()
        stddev.release()

        return variance
    }
}

@Composable
fun CameraPreview(blurScoreState: MutableState<String>) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                val executor = Executors.newSingleThreadExecutor()
                val mainActivity = ctx as? MainActivity

                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    // Настройка предпросмотра
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    // Настройка анализа изображений
                    val imageAnalysis = ImageAnalysis.Builder()
                        .setTargetResolution(Size(1280, 720))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    // Обработка кадров
                    imageAnalysis.setAnalyzer(executor) { imageProxy ->
                        // Получаем изображение и преобразуем в Bitmap
                        val bitmap = imageProxy.toBitmap()

                        // Преобразуем Bitmap в Mat для OpenCV
                        val mat = Mat()
                        Utils.bitmapToMat(bitmap, mat)

                        // Вычисляем уровень размытости
                        val blurScore = mainActivity?.calculateBlurScore(mat) ?: 0.0
                        blurScoreState.value = "Blur Score: %.2f".format(blurScore)

                        // Освобождаем ресурсы
                        mat.release()
                        imageProxy.close()
                    }

                    // Выбор задней камеры
                    val cameraSelector = CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build()

                    // Привязка к жизненному циклу
                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            cameraSelector,
                            preview,
                            imageAnalysis
                        )
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }
                }, ContextCompat.getMainExecutor(context))

                previewView
            }
        )

        // Отображение уровня размытости
        Text(
            text = blurScoreState.value,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(16.dp)
        )
    }
}

// Вспомогательная функция для преобразования ImageProxy в Bitmap
private fun ImageProxy.toBitmap(): Bitmap {
    val yBuffer: ByteBuffer = planes[0].buffer // Y
    val uBuffer: ByteBuffer = planes[1].buffer // U
    val vBuffer: ByteBuffer = planes[2].buffer // V

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
}