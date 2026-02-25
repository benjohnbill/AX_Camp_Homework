package com.example.narrativeloopmobile.network

import retrofit2.Response
import retrofit2.http.GET
import retrofit2.http.Url

interface ApiService {
    @GET
    suspend fun getUniverse(@Url url: String): Response<String> // We expect an HTML response, so we use String
}