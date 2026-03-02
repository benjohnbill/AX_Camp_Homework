package com.example.narrativeloopmobile.network

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.Header
import retrofit2.http.POST

data class DebugTokenIssueRequest(
    val user_id: String,
    val aud: String,
    val ttl_minutes: Int,
)

data class DebugTokenIssueResponse(
    val status: Int,
    val code: String?,
    val token: String?,
    val token_type: String?,
    val message: String?,
    val expires_at_utc: String?,
)

interface DebugTokenApiService {
    @POST("debug/token")
    suspend fun issueToken(
        @Header("X-Debug-Admin-Key") adminKey: String,
        @Body requestBody: DebugTokenIssueRequest,
    ): Response<DebugTokenIssueResponse>
}
