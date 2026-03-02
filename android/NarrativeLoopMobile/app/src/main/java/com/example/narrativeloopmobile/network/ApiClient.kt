package com.example.narrativeloopmobile.network

import com.example.narrativeloopmobile.BuildConfig
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.converter.scalars.ScalarsConverterFactory

object ApiClient {

    private val authInterceptor = AuthInterceptor()

    private fun buildClient(disableRedirects: Boolean): OkHttpClient {
        val builder = OkHttpClient.Builder()
            .addInterceptor(authInterceptor)
        if (disableRedirects) {
            builder.followRedirects(false)
            builder.followSslRedirects(false)
        }
        if (BuildConfig.DEBUG) {
            val loggingInterceptor = HttpLoggingInterceptor()
            loggingInterceptor.setLevel(HttpLoggingInterceptor.Level.BODY)
            builder.addInterceptor(loggingInterceptor)
        }
        return builder.build()
    }

    private fun buildPlainClient(): OkHttpClient {
        val builder = OkHttpClient.Builder()
        if (BuildConfig.DEBUG) {
            val loggingInterceptor = HttpLoggingInterceptor()
            loggingInterceptor.setLevel(HttpLoggingInterceptor.Level.BODY)
            builder.addInterceptor(loggingInterceptor)
        }
        return builder.build()
    }

    private val defaultClient: OkHttpClient by lazy {
        buildClient(disableRedirects = false)
    }

    private val debugClient: OkHttpClient by lazy {
        // Debug probe path needs raw 307/401/403 responses for contract verification.
        buildClient(disableRedirects = true)
    }

    val apiService: ApiService by lazy {
        Retrofit.Builder()
            .baseUrl("https://ax-camp-universe-gateway-staging.onrender.com/")
            .client(defaultClient)
            .addConverterFactory(ScalarsConverterFactory.create())
            .build()
            .create(ApiService::class.java)
    }

    val debugApiService: ApiService by lazy {
        Retrofit.Builder()
            .baseUrl("https://ax-camp-universe-gateway-staging.onrender.com/")
            .client(debugClient)
            .addConverterFactory(ScalarsConverterFactory.create())
            .build()
            .create(ApiService::class.java)
    }

    val narrativeApiService: NarrativeApiService by lazy {
        Retrofit.Builder()
            .baseUrl("https://ax-camp-universe-gateway-staging.onrender.com/")
            .client(defaultClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(NarrativeApiService::class.java)
    }

    val debugTokenApiService: DebugTokenApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BuildConfig.DEBUG_TOKEN_BASE_URL)
            .client(buildPlainClient())
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(DebugTokenApiService::class.java)
    }

    fun setAuthToken(token: String?) {
        authInterceptor.token = token
    }
}
