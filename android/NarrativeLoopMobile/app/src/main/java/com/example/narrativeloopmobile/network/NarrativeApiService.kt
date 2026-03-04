package com.example.narrativeloopmobile.network

import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path

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

data class SaveNarrativeLogRequest(
    val text: String
)

data class SaveNarrativeLogResponse(
    val status: String? = null,
    val log_id: String? = null
)

data class StartSessionRequest(
    val entry_mode: String = "plan"
)

data class StartSessionResponse(
    val status: String? = null,
    val session_id: String? = null,
    val flow_stage: String? = null,
    val entry_mode: String? = null
)

data class SessionSummary(
    val id: String,
    val session_date: String? = null,
    val flow_stage: String? = null,
    val plan_status: String? = null,
    val entry_mode: String? = null
)

data class TodaySessionResponse(
    val status: String? = null,
    val session: SessionSummary? = null
)

data class TimeboxBlockRequest(
    val id: String? = null,
    val title: String,
    val goal: String? = null,
    val why: String? = null,
    val inbox_note: String? = null,
    val starts_at: String? = null,
    val ends_at: String? = null,
    val order_index: Int? = null
)

data class FrogRequest(
    val frog_title: String,
    val frog_why: String
)

data class TimeboxDraftRequest(
    val blocks: List<TimeboxBlockRequest>,
    val manual_tags: List<String> = emptyList()
)

data class TimeboxRetroRequest(
    val skip: Boolean = false,
    val blocks: List<TimeboxBlockRequest> = emptyList()
)

data class EvidenceLink(
    val image_event_id: String,
    val decision: String = "linked",
    val user_meaning: String? = null
)

data class ReflectRequest(
    val reflection_good: String,
    val reflection_hard: String,
    val reflection_next_action: String,
    val reflection_free_text: String,
    val evidence_links: List<EvidenceLink> = emptyList()
)

data class EvidenceLinkSummary(
    val linked: Int? = null,
    val skipped: Int? = null,
    val missing: Int? = null
)

data class StageActionResponse(
    val status: String? = null,
    val session_id: String? = null,
    val flow_stage: String? = null,
    val plan_status: String? = null,
    val focus_total_minutes: Int? = null,
    val blocks_count: Int? = null,
    val skip: Boolean? = null,
    val queued_jobs: List<String>? = null,
    val evidence_link_summary: EvidenceLinkSummary? = null
)

data class InsightPayload(
    val similar_pattern: String? = null,
    val next_action: String? = null
)

data class SessionInsightsResponse(
    val status: String? = null,
    val insight_source: String? = null,
    val auto_tags: List<String>? = null,
    val insights: InsightPayload? = null
)

/**
 * VisionResponse aligned with agent.md Section 6
 */
data class VisionResponse(
    val status: String? = null,
    val image_event_id: String? = null,
    val ocr_status: String? = null,
    val session_id: String? = null,
    val link_rule: String? = null,
    val request_id: String? = null,
    val ocr_text_normalized: String? = null,
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

    @POST("v1/narrative")
    suspend fun saveNarrativeLog(@Body requestBody: SaveNarrativeLogRequest): Response<SaveNarrativeLogResponse>

    @POST("v1/narrative/refine")
    suspend fun refineNarrative(@Body requestBody: RefineRequest): Response<RefineResponse>

    @GET("v1/execution/session/today")
    suspend fun getTodaySession(): Response<TodaySessionResponse>

    @POST("v1/execution/session/start")
    suspend fun startSession(@Body requestBody: StartSessionRequest = StartSessionRequest()): Response<StartSessionResponse>

    @POST("v1/execution/session/{session_id}/frog")
    suspend fun saveFrog(
        @Path("session_id") sessionId: String,
        @Body requestBody: FrogRequest
    ): Response<StageActionResponse>

    @POST("v1/execution/session/{session_id}/timebox/draft")
    suspend fun saveTimeboxDraft(
        @Path("session_id") sessionId: String,
        @Body requestBody: TimeboxDraftRequest
    ): Response<StageActionResponse>

    @POST("v1/execution/session/{session_id}/commit")
    suspend fun commitSession(@Path("session_id") sessionId: String): Response<StageActionResponse>

    @POST("v1/execution/session/{session_id}/focus/start")
    suspend fun startFocus(@Path("session_id") sessionId: String): Response<StageActionResponse>

    @POST("v1/execution/session/{session_id}/focus/end")
    suspend fun endFocus(@Path("session_id") sessionId: String): Response<StageActionResponse>

    @POST("v1/execution/session/{session_id}/timebox/retro")
    suspend fun saveTimeboxRetro(
        @Path("session_id") sessionId: String,
        @Body requestBody: TimeboxRetroRequest
    ): Response<StageActionResponse>

    @POST("v1/execution/session/{session_id}/reflect")
    suspend fun reflectSession(
        @Path("session_id") sessionId: String,
        @Body requestBody: ReflectRequest
    ): Response<StageActionResponse>

    @GET("v1/execution/session/{session_id}/insights")
    suspend fun getSessionInsights(@Path("session_id") sessionId: String): Response<SessionInsightsResponse>

    /**
     * Upload image for OCR processing. 
     * Fixed route to match gateway_fastapi.py standard.
     */
    @Multipart
    @POST("v1/ocr/ingest")
    suspend fun uploadImageForVision(
        @Part image: MultipartBody.Part,
        @Part("session_id") sessionId: RequestBody? = null,
        @Part("session-link") sessionLink: RequestBody? = null
    ): Response<VisionResponse>
}
