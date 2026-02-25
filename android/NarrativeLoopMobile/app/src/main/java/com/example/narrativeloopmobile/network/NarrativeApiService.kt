package com.example.narrativeloopmobile.network

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

data class IngestRequestBody(
    val user_id: String,
    val image_base64: String,
    val client_ts: String,
    val session_id: String,
    val mode_hint: String,
    val manual_override_text: String
)

interface NarrativeApiService {
    @POST("v1/ocr/ingest")
    suspend fun saveNarrative(@Body requestBody: IngestRequestBody): Response<Unit>
}
