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

data class VisionResponse(
    val request_id: String,
    val ocr_text_normalized: String,
    val ai_response: String?,
    val refined_text: String? // Legacy compatibility
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
