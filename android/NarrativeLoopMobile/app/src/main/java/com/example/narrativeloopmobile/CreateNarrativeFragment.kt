package com.example.narrativeloopmobile

import android.app.Activity.RESULT_OK
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.findNavController
import com.example.narrativeloopmobile.network.NarrativeRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class CreateNarrativeFragment : Fragment() {

    private lateinit var narrativeEditText: EditText
    private lateinit var saveNarrativeButton: Button
    private lateinit var aiRefineButton: Button
    private lateinit var cameraButton: Button
    private lateinit var progressBar: ProgressBar
    private val narrativeRepository = NarrativeRepository()

    private val takePictureLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            val imageBitmap = result.data?.extras?.get("data") as Bitmap
            val imageFile = saveBitmapToFile(imageBitmap)
            imageFile?.let { uploadImage(it) }
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
        if (isGranted) {
            dispatchTakePictureIntent()
        } else {
            Toast.makeText(requireContext(), "Camera permission is required to use this feature.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_create_narrative, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        narrativeEditText = view.findViewById(R.id.edit_text_narrative)
        saveNarrativeButton = view.findViewById(R.id.button_save_narrative)
        aiRefineButton = view.findViewById(R.id.button_ai_refine)
        cameraButton = view.findViewById(R.id.button_camera)
        progressBar = view.findViewById(R.id.progress_bar)

        cameraButton.setOnClickListener {
            when {
                ContextCompat.checkSelfPermission(
                    requireContext(),
                    android.Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED -> {
                    dispatchTakePictureIntent()
                }
                else -> {
                    requestPermissionLauncher.launch(android.Manifest.permission.CAMERA)
                }
            }
        }

        aiRefineButton.setOnClickListener { 
            // ... (existing AI Refine logic) 
        }

        saveNarrativeButton.setOnClickListener { 
            // ... (existing Save Narrative logic) 
        }
    }

    private fun dispatchTakePictureIntent() {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            takePictureIntent.resolveActivity(requireActivity().packageManager)?.also {
                takePictureLauncher.launch(takePictureIntent)
            }
        }
    }

    private fun saveBitmapToFile(bitmap: Bitmap): File? {
        val file = File(requireContext().cacheDir, "temp_image.jpg")
        return try {
            val fos = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)
            fos.flush()
            fos.close()
            file
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun uploadImage(file: File) {
        lifecycleScope.launch {
            withContext(Dispatchers.Main) {
                progressBar.visibility = View.VISIBLE
            }
            try {
                val response = narrativeRepository.uploadImageForVision(file)
                if (response.isSuccessful && response.body() != null) {
                    withContext(Dispatchers.Main) {
                        narrativeEditText.setText(response.body()!!.refined_text)
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(requireContext(), "Error: ${response.code()}", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(requireContext(), "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            } finally {
                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                }
            }
        }
    }
}
