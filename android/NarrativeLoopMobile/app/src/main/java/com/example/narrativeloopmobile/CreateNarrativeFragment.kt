package com.example.narrativeloopmobile

import android.app.Activity.RESULT_OK
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.example.narrativeloopmobile.network.NarrativeRepository
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream

class CreateNarrativeFragment : Fragment() {
    private companion object {
        private const val TAG = "CreateNarrativeFragment"
    }

    private enum class ExecutionMode {
        PLAN_FIRST,
        FOCUS_FIRST
    }

    private data class FlowRunResult(
        val sessionId: String,
        val finalStage: String,
        val linkedEvidenceCount: Int
    )

    private lateinit var narrativeEditText: EditText
    private lateinit var saveNarrativeButton: Button
    private lateinit var aiRefineButton: Button
    private lateinit var cameraButton: Button
    private lateinit var modeRadioGroup: RadioGroup
    private lateinit var modePlanRadio: RadioButton
    private lateinit var modeFocusRadio: RadioButton
    private lateinit var stageStateText: TextView
    private lateinit var actionStatusText: TextView
    private lateinit var evidenceStateText: TextView
    private lateinit var progressBar: ProgressBar
    private val narrativeRepository = NarrativeRepository()
    private var lastImageEventId: String? = null

    private val takePictureLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK && result.data?.extras?.get("data") is Bitmap) {
            val imageBitmap = result.data?.extras?.get("data") as Bitmap
            val imageFile = saveBitmapToFile(imageBitmap)
            imageFile?.let { uploadImage(it) }
        } else {
            updateActionStatus("Camera capture was cancelled.")
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
        modeRadioGroup = view.findViewById(R.id.radio_group_mode)
        modePlanRadio = view.findViewById(R.id.radio_mode_plan)
        modeFocusRadio = view.findViewById(R.id.radio_mode_focus)
        stageStateText = view.findViewById(R.id.text_stage_state)
        actionStatusText = view.findViewById(R.id.text_action_status)
        evidenceStateText = view.findViewById(R.id.text_evidence_state)
        progressBar = view.findViewById(R.id.progress_bar)

        modePlanRadio.isChecked = true
        updateStageState("idle")
        updateActionStatus("Ready")
        updateEvidenceState("No OCR evidence linked yet.")

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
            val text = narrativeEditText.text.toString().trim()
            if (text.isBlank()) {
                Toast.makeText(requireContext(), "Narrative text is required for AI refine.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            runAiRefine(text)
        }

        saveNarrativeButton.setOnClickListener {
            val text = narrativeEditText.text.toString().trim()
            if (text.isBlank()) {
                Toast.makeText(requireContext(), "Narrative text is required for save.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            runSaveAndE2eFlow(text)
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
            bitmap.compress(Bitmap.CompressFormat.JPEG, 92, fos)
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
            setLoading(true)
            try {
                updateActionStatus("Uploading OCR evidence...")
                val response = narrativeRepository.uploadImageForVision(file)
                val body = response.body()
                if (response.isSuccessful && body != null) {
                    lastImageEventId = body.image_event_id
                    Log.i(
                        TAG,
                        "ocr_upload status=${body.status} image_event_id=${body.image_event_id} session_id=${body.session_id} link_rule=${body.link_rule}"
                    )
                    val resolvedText = body.refined_text ?: body.ocr_text_normalized.orEmpty()
                    narrativeEditText.setText(resolvedText)
                    updateEvidenceState(
                        "image_event_id=${body.image_event_id ?: "n/a"} / session_id=${body.session_id ?: "n/a"} / link_rule=${body.link_rule ?: "n/a"}"
                    )
                    updateActionStatus("OCR upload accepted.")
                } else {
                    throw IllegalStateException("OCR upload failed with HTTP ${response.code()}")
                }
            } catch (e: Exception) {
                updateActionStatus("OCR upload failed: ${e.message}")
                showRetryDialog(
                    title = "OCR upload failed",
                    message = "Failed to upload OCR evidence. Retry now?",
                    onRetry = { uploadImage(file) }
                )
            } finally {
                setLoading(false)
            }
        }
    }

    private fun runAiRefine(text: String) {
        lifecycleScope.launch {
            setLoading(true)
            try {
                updateActionStatus("Requesting AI refine...")
                val response = narrativeRepository.refineNarrative(text)
                val body = requireBody(response, "narrative/refine")
                narrativeEditText.setText(body.refined_text)
                updateActionStatus("AI refine complete.")
            } catch (e: Exception) {
                updateActionStatus("AI refine failed: ${e.message}")
                showRetryDialog(
                    title = "AI refine failed",
                    message = "Refine request failed. Retry with same text?",
                    onRetry = { runAiRefine(text) }
                )
            } finally {
                setLoading(false)
            }
        }
    }

    private fun runSaveAndE2eFlow(narrativeText: String) {
        lifecycleScope.launch {
            setLoading(true)
            try {
                updateActionStatus("Saving narrative log...")
                val saveResponse = narrativeRepository.saveNarrativeLog(narrativeText)
                val saveBody = requireBody(saveResponse, "narrative/save")
                val logId = saveBody.log_id ?: "n/a"

                ensureEvidenceReady(narrativeText)

                val mode = selectedExecutionMode()
                val flowResult = when (mode) {
                    ExecutionMode.PLAN_FIRST -> executePlanFirstFlow(narrativeText)
                    ExecutionMode.FOCUS_FIRST -> executeFocusFirstFlow(narrativeText)
                }

                updateStageState(flowResult.finalStage)
                updateActionStatus(
                    "E2E save completed (${modeLabel(mode)}). log_id=$logId session_id=${flowResult.sessionId}"
                )
                updateEvidenceState(
                    "image_event_id=${lastImageEventId ?: "n/a"} / linked_count=${flowResult.linkedEvidenceCount}"
                )
            } catch (e: Exception) {
                updateActionStatus("E2E save flow failed: ${e.message}")
                showRetryDialog(
                    title = "Save flow failed",
                    message = "E2E flow failed. Retry save + stage chain?",
                    onRetry = { runSaveAndE2eFlow(narrativeText) }
                )
            } finally {
                setLoading(false)
            }
        }
    }

    private suspend fun executePlanFirstFlow(narrativeText: String): FlowRunResult {
        updateStageState("start(plan)")
        val startBody = requireBody(
            narrativeRepository.startSession(entryMode = "plan"),
            "execution/session/start(plan)"
        )
        val sessionId = startBody.session_id ?: throw IllegalStateException("start(plan) returned empty session_id")

        updateStageState("frog")
        requireBody(
            narrativeRepository.saveFrog(
                sessionId = sessionId,
                title = deriveFrogTitle(narrativeText),
                why = "Android plan-first e2e run"
            ),
            "execution/session/frog"
        )

        updateStageState("timebox/draft")
        requireBody(
            narrativeRepository.saveTimeboxDraft(
                sessionId = sessionId,
                title = "Android Plan Block",
                goal = "Complete one focused narrative cycle"
            ),
            "execution/session/timebox/draft"
        )

        updateStageState("commit")
        requireBody(narrativeRepository.commitSession(sessionId), "execution/session/commit")

        updateStageState("focus/start")
        requireBody(narrativeRepository.startFocus(sessionId), "execution/session/focus/start")

        updateStageState("focus/end")
        val focusEnd = requireBody(narrativeRepository.endFocus(sessionId), "execution/session/focus/end")

        updateStageState("reflect")
        val reflect = requireBody(
            narrativeRepository.reflectSession(
                sessionId = sessionId,
                narrativeText = narrativeText,
                imageEventId = lastImageEventId,
                modeLabel = "Plan-first"
            ),
            "execution/session/reflect(plan)"
        )
        return FlowRunResult(
            sessionId = sessionId,
            finalStage = reflect.flow_stage ?: focusEnd.flow_stage ?: "done",
            linkedEvidenceCount = reflect.evidence_link_summary?.linked ?: 0
        )
    }

    private suspend fun executeFocusFirstFlow(narrativeText: String): FlowRunResult {
        updateStageState("start(focus_now)")
        val startBody = requireBody(
            narrativeRepository.startSession(entryMode = "focus_now"),
            "execution/session/start(focus_now)"
        )
        val sessionId = startBody.session_id ?: throw IllegalStateException("start(focus_now) returned empty session_id")

        updateStageState("focus/start")
        requireBody(narrativeRepository.startFocus(sessionId), "execution/session/focus/start")

        updateStageState("focus/end")
        requireBody(narrativeRepository.endFocus(sessionId), "execution/session/focus/end")

        updateStageState("timebox/retro")
        requireBody(
            narrativeRepository.saveTimeboxRetro(
                sessionId = sessionId,
                title = "Android Retro Block"
            ),
            "execution/session/timebox/retro"
        )

        updateStageState("reflect")
        val reflect = requireBody(
            narrativeRepository.reflectSession(
                sessionId = sessionId,
                narrativeText = narrativeText,
                imageEventId = lastImageEventId,
                modeLabel = "Focus-first"
            ),
            "execution/session/reflect(focus_now)"
        )
        return FlowRunResult(
            sessionId = sessionId,
            finalStage = reflect.flow_stage ?: "done",
            linkedEvidenceCount = reflect.evidence_link_summary?.linked ?: 0
        )
    }

    private suspend fun ensureEvidenceReady(narrativeText: String) {
        if (!lastImageEventId.isNullOrBlank()) {
            return
        }
        updateActionStatus("No OCR evidence found. Generating synthetic evidence upload...")
        val syntheticFile = createSyntheticEvidenceFile(narrativeText)
            ?: throw IllegalStateException("Failed to generate synthetic OCR evidence image")
        try {
            val response = narrativeRepository.uploadImageForVision(syntheticFile)
            val body = requireBody(response, "ocr/ingest(synthetic)")
            if (body.image_event_id.isNullOrBlank()) {
                throw IllegalStateException("Synthetic OCR upload returned empty image_event_id")
            }
            lastImageEventId = body.image_event_id
            updateEvidenceState(
                "image_event_id=${body.image_event_id} / session_id=${body.session_id ?: "n/a"} / link_rule=${body.link_rule ?: "n/a"}"
            )
        } finally {
            syntheticFile.delete()
        }
    }

    private fun createSyntheticEvidenceFile(text: String): File? {
        return try {
            val bitmap = Bitmap.createBitmap(1080, 720, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            canvas.drawColor(Color.WHITE)

            val paint = Paint().apply {
                color = Color.BLACK
                textSize = 36f
                isAntiAlias = true
            }
            canvas.drawText("Android Phase2.5 OCR Evidence", 40f, 70f, paint)
            canvas.drawText("Generated: ${narrativeRepository.nowIsoUtc()}", 40f, 120f, paint)
            val lines = text.chunked(48).take(8)
            lines.forEachIndexed { index, line ->
                canvas.drawText(line, 40f, 190f + (index * 52f), paint)
            }

            val file = File(requireContext().cacheDir, "synthetic_evidence_${System.currentTimeMillis()}.jpg")
            FileOutputStream(file).use { fos ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 92, fos)
            }
            file
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create synthetic evidence image", e)
            null
        }
    }

    private fun selectedExecutionMode(): ExecutionMode {
        return if (modeRadioGroup.checkedRadioButtonId == R.id.radio_mode_focus) {
            ExecutionMode.FOCUS_FIRST
        } else {
            ExecutionMode.PLAN_FIRST
        }
    }

    private fun modeLabel(mode: ExecutionMode): String {
        return when (mode) {
            ExecutionMode.PLAN_FIRST -> "Plan-first"
            ExecutionMode.FOCUS_FIRST -> "Focus-first"
        }
    }

    private fun deriveFrogTitle(narrativeText: String): String {
        val firstLine = narrativeText.lineSequence().firstOrNull()?.trim().orEmpty()
        if (firstLine.isBlank()) {
            return "Android E2E narrative session"
        }
        return if (firstLine.length > 42) {
            firstLine.take(42)
        } else {
            firstLine
        }
    }

    private fun updateStageState(stage: String) {
        stageStateText.text = "Stage: $stage"
        Log.i(TAG, "stage_update=$stage")
    }

    private fun updateActionStatus(status: String) {
        actionStatusText.text = "Status: $status"
        Log.i(TAG, "action_status=$status")
    }

    private fun updateEvidenceState(detail: String) {
        evidenceStateText.text = "Evidence: $detail"
        Log.i(TAG, "evidence_state=$detail")
    }

    private fun setLoading(isLoading: Boolean) {
        progressBar.visibility = if (isLoading) View.VISIBLE else View.GONE
        saveNarrativeButton.isEnabled = !isLoading
        aiRefineButton.isEnabled = !isLoading
        cameraButton.isEnabled = !isLoading
    }

    private fun showRetryDialog(title: String, message: String, onRetry: () -> Unit) {
        if (!isAdded) return
        AlertDialog.Builder(requireContext())
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("Retry") { _, _ -> onRetry() }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun <T> requireBody(response: retrofit2.Response<T>, step: String): T {
        if (!response.isSuccessful) {
            throw IllegalStateException("$step failed with HTTP ${response.code()}")
        }
        return response.body() ?: throw IllegalStateException("$step returned empty body")
    }

    override fun onDestroyView() {
        super.onDestroyView()
        lastImageEventId = null
        modeRadioGroup.clearCheck()
        modePlanRadio.isChecked = true
        modeFocusRadio.isChecked = false
    }
}
