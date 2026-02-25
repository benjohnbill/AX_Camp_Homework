package com.example.narrativeloopmobile.network

import okhttp3.Interceptor
import okhttp3.Response

class AuthInterceptor : Interceptor {

    var token: String? = null

    override fun intercept(chain: Interceptor.Chain): Response {
        val originalRequest = chain.request()
        val requestBuilder = originalRequest.newBuilder()

        val currentToken = token?.trim()
        if (!currentToken.isNullOrBlank()) {
            requestBuilder.addHeader("Authorization", "Bearer $currentToken")
        }

        val request = requestBuilder.build()
        return chain.proceed(request)
    }
}
