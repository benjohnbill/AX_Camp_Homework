package com.example.narrativeloopmobile.network

import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

data class IngestRequestBody(
    val user_id: String,
    val image_base64: String,
    val client_ts: String,
    val session_id: String,
    val mode_hint: String,
    val manual_override_text: String
)

data class RefineRequest(val text: String)

data class RefineResponse(val refined_text: String)

/**
 * VisionResponse aligned with agent.md Section 6
 */
data class VisionResponse(
    val request_id: String,
    val ocr_text_normalized: String,
    val ocr_text_raw: String? = null,
    val confidence: Double? = null,
    val saved_log_id: String? = null,
    val ai_response: String? = null,
    val related_log_ids: List<String>? = null,
    val warnings: List<String>? = null,
    val refined_text: String? = null // Kept for legacy compatibility during transition
)

interface NarrativeApiService {
    /**
     * Standard Ingest API (Section 6 of agent.md)
     */
    @POST("v1/ocr/ingest")
    suspend fun saveNarrative(@Body requestBody: IngestRequestBody): Response<Unit>

    @POST("v1/narrative/refine")
    suspend fun refineNarrative(@Body requestBody: RefineRequest): Response<RefineResponse>

    /**
     * Upload image for OCR processing. 
     * Fixed route to match gateway_fastapi.py standard.
     */
    @Multipart
    @POST("v1/ocr/ingest")
    suspend fun uploadImageForVision(@Part image: MultipartBody.Part): Response<VisionResponse>
}
