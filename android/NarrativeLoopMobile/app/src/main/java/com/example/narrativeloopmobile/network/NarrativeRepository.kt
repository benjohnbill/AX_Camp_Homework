package com.example.narrativeloopmobile.network

import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.Response
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

class NarrativeRepository {

    private val narrativeApiService = ApiClient.narrativeApiService
    private val textPlain = "text/plain".toMediaTypeOrNull()

    fun nowIsoUtc(): String {
        return SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).format(Date())
    }

    private fun isoUtcFromMillis(millis: Long): String {
        return SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).format(Date(millis))
    }

    suspend fun saveNarrative(narrativeText: String): Response<Unit> {
        val requestBody = IngestRequestBody(
            user_id = "android-e2e-user",
            image_base64 = "",
            client_ts = nowIsoUtc(),
            session_id = UUID.randomUUID().toString(),
            mode_hint = "stream",
            manual_override_text = narrativeText
        )
        return narrativeApiService.saveNarrative(requestBody)
    }

    suspend fun saveNarrativeLog(narrativeText: String): Response<SaveNarrativeLogResponse> {
        return narrativeApiService.saveNarrativeLog(SaveNarrativeLogRequest(text = narrativeText))
    }

    suspend fun refineNarrative(text: String): Response<RefineResponse> {
        return narrativeApiService.refineNarrative(RefineRequest(text))
    }

    suspend fun getTodaySession(): Response<TodaySessionResponse> {
        return narrativeApiService.getTodaySession()
    }

    suspend fun getSessionInsights(sessionId: String): Response<SessionInsightsResponse> {
        return narrativeApiService.getSessionInsights(sessionId)
    }

    suspend fun startSession(entryMode: String): Response<StartSessionResponse> {
        return narrativeApiService.startSession(StartSessionRequest(entry_mode = entryMode))
    }

    suspend fun saveFrog(sessionId: String, title: String, why: String): Response<StageActionResponse> {
        return narrativeApiService.saveFrog(sessionId, FrogRequest(frog_title = title, frog_why = why))
    }

    suspend fun saveTimeboxDraft(sessionId: String, title: String, goal: String): Response<StageActionResponse> {
        val nowMillis = System.currentTimeMillis()
        val startAt = isoUtcFromMillis(nowMillis)
        val endAt = isoUtcFromMillis(nowMillis + (25L * 60L * 1000L))
        val block = TimeboxBlockRequest(
            id = "blk_${UUID.randomUUID().toString().take(8)}",
            title = title,
            goal = goal,
            why = "Android plan-first stage execution",
            inbox_note = "[[android]] [[phase25]]",
            starts_at = startAt,
            ends_at = endAt,
            order_index = 0
        )
        return narrativeApiService.saveTimeboxDraft(
            sessionId,
            TimeboxDraftRequest(blocks = listOf(block), manual_tags = listOf("android", "phase25"))
        )
    }

    suspend fun saveTimeboxRetro(sessionId: String, title: String): Response<StageActionResponse> {
        val block = TimeboxBlockRequest(
            id = "rblk_${UUID.randomUUID().toString().take(8)}",
            title = title,
            goal = "Retro capture from Android focus-first path",
            why = "Phase2.5 focus-first verification",
            starts_at = "2026-03-05T09:00:00Z",
            ends_at = "2026-03-05T09:25:00Z",
            order_index = 0
        )
        return narrativeApiService.saveTimeboxRetro(
            sessionId,
            TimeboxRetroRequest(skip = false, blocks = listOf(block))
        )
    }

    suspend fun commitSession(sessionId: String): Response<StageActionResponse> {
        return narrativeApiService.commitSession(sessionId)
    }

    suspend fun startFocus(sessionId: String): Response<StageActionResponse> {
        return narrativeApiService.startFocus(sessionId)
    }

    suspend fun endFocus(sessionId: String): Response<StageActionResponse> {
        return narrativeApiService.endFocus(sessionId)
    }

    suspend fun reflectSession(
        sessionId: String,
        narrativeText: String,
        imageEventId: String?,
        modeLabel: String
    ): Response<StageActionResponse> {
        val links = if (imageEventId.isNullOrBlank()) {
            emptyList()
        } else {
            listOf(
                EvidenceLink(
                    image_event_id = imageEventId,
                    decision = "linked",
                    user_meaning = "Android $modeLabel evidence linked from OCR upload"
                )
            )
        }
        return narrativeApiService.reflectSession(
            sessionId,
            ReflectRequest(
                reflection_good = "$modeLabel path completed with narrative save.",
                reflection_hard = "Mobile context switch and camera flow required careful sequencing.",
                reflection_next_action = "Repeat $modeLabel flow with one focused block and immediate evidence curation.",
                reflection_free_text = narrativeText,
                evidence_links = links
            )
        )
    }

    suspend fun uploadImageForVision(imageFile: File): Response<VisionResponse> {
        val sessionLinkId = resolveSessionLinkId()
        val requestFile = imageFile.asRequestBody("image/*".toMediaTypeOrNull())
        val body = MultipartBody.Part.createFormData("image", imageFile.name, requestFile)
        val sessionIdPart = sessionLinkId?.toRequestBody(textPlain)
        val sessionLinkPart = sessionLinkId?.toRequestBody(textPlain)
        return narrativeApiService.uploadImageForVision(
            image = body,
            sessionId = sessionIdPart,
            sessionLink = sessionLinkPart
        )
    }

    private suspend fun resolveSessionLinkId(): String? {
        return try {
            val todayResponse = getTodaySession()
            if (todayResponse.isSuccessful) {
                val todayPayload = todayResponse.body()
                val todaySessionId = todayPayload?.session?.id?.trim().orEmpty()
                if (todayPayload?.status == "ok" && todaySessionId.isNotBlank()) {
                    return todaySessionId
                }
            }

            val startResponse = startSession(entryMode = "plan")
            if (!startResponse.isSuccessful) {
                null
            } else {
                startResponse.body()?.session_id?.trim()?.ifBlank { null }
            }
        } catch (_: Exception) {
            null
        }
    }
}
