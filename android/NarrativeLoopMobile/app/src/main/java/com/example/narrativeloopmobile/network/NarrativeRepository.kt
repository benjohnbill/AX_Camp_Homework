package com.example.narrativeloopmobile.network

import retrofit2.Response
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

class NarrativeRepository {

    private val narrativeApiService = ApiClient.narrativeApiService

    suspend fun saveNarrative(narrativeText: String): Response<Unit> {
        val requestBody = IngestRequestBody(
            user_id = "android-e2e-user",
            image_base64 = "",
            client_ts = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).format(Date()),
            session_id = UUID.randomUUID().toString(),
            mode_hint = "stream",
            manual_override_text = narrativeText
        )
        return narrativeApiService.saveNarrative(requestBody)
    }
}
