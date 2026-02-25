package com.example.narrativeloopmobile.network

import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import retrofit2.Response
import java.io.File
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

    suspend fun refineNarrative(text: String): Response<RefineResponse> {
        return narrativeApiService.refineNarrative(RefineRequest(text))
    }

    suspend fun uploadImageForVision(imageFile: File): Response<VisionResponse> {
        val requestFile = imageFile.asRequestBody("image/*".toMediaTypeOrNull())
        val body = MultipartBody.Part.createFormData("image", imageFile.name, requestFile)
        return narrativeApiService.uploadImageForVision(body)
    }
}
