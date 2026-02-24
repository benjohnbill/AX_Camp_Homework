package com.example.narrativeloopmobile

import android.content.Context

object TokenStore {
    private const val PREFS = "auth_prefs"
    private const val KEY_ACCESS_TOKEN = "access_token"

    fun saveAccessToken(context: Context, token: String) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_ACCESS_TOKEN, token)
            .apply()
    }

    fun getAccessToken(context: Context): String? {
        return context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(KEY_ACCESS_TOKEN, null)
    }

    fun clear(context: Context) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .remove(KEY_ACCESS_TOKEN)
            .apply()
    }
}