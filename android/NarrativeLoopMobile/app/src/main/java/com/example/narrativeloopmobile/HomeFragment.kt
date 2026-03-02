package com.example.narrativeloopmobile

import android.os.Bundle
import android.view.View
import android.webkit.CookieManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.example.narrativeloopmobile.network.ApiClient
import com.example.narrativeloopmobile.network.DebugTokenIssueRequest
import com.google.android.material.bottomnavigation.BottomNavigationView
import kotlinx.coroutines.launch

class HomeFragment : Fragment(R.layout.fragment_home) {

    private lateinit var writeNarrativeButton: Button
    private lateinit var debugToolsContainer: LinearLayout
    private lateinit var tokenStatusText: TextView
    private lateinit var tokenInput: EditText
    private lateinit var saveTokenButton: Button
    private lateinit var clearTokenButton: Button
    private lateinit var logoutButton: Button
    private lateinit var debugUserIdInput: EditText
    private lateinit var debugAudienceInput: EditText
    private lateinit var debugTtlInput: EditText
    private lateinit var debugAdminKeyInput: EditText
    private lateinit var issueDebugTokenButton: Button
    private lateinit var debugIssueResultText: TextView

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        writeNarrativeButton = view.findViewById(R.id.button_write_narrative)
        debugToolsContainer = view.findViewById(R.id.debug_tools_container)
        tokenStatusText = view.findViewById(R.id.token_status_text)
        tokenInput = view.findViewById(R.id.token_input_edittext)
        saveTokenButton = view.findViewById(R.id.save_token_button)
        clearTokenButton = view.findViewById(R.id.clear_token_button)
        logoutButton = view.findViewById(R.id.logout_button)
        debugUserIdInput = view.findViewById(R.id.debug_user_id_edittext)
        debugAudienceInput = view.findViewById(R.id.debug_audience_edittext)
        debugTtlInput = view.findViewById(R.id.debug_ttl_edittext)
        debugAdminKeyInput = view.findViewById(R.id.debug_admin_key_edittext)
        issueDebugTokenButton = view.findViewById(R.id.issue_debug_token_button)
        debugIssueResultText = view.findViewById(R.id.debug_issue_result_text)

        writeNarrativeButton.setOnClickListener {
            val bottomNav = requireActivity().findViewById<BottomNavigationView>(R.id.bottom_nav)
            bottomNav.selectedItemId = R.id.nav_create_narrative
        }

        if (BuildConfig.DEBUG) {
            setupDebugTools()
        } else {
            debugToolsContainer.visibility = View.GONE
        }
    }

    private fun setupDebugTools() {
        debugToolsContainer.visibility = View.VISIBLE
        updateTokenStatus()

        saveTokenButton.setOnClickListener {
            val token = normalizeToken(tokenInput.text.toString())
            if (token.isNotBlank()) {
                TokenStore.saveAccessToken(requireContext(), token)
                // [FIX] Update interceptor immediately after manual save
                ApiClient.setAuthToken(token) 
                tokenInput.text.clear()
                updateTokenStatus()
                Toast.makeText(requireContext(), "Token saved and applied.", Toast.LENGTH_SHORT).show()
            }
        }

        clearTokenButton.setOnClickListener {
            TokenStore.clear(requireContext())
            ApiClient.setAuthToken(null) // [FIX] Clear interceptor
            updateTokenStatus()
        }

        logoutButton.setOnClickListener {
            TokenStore.clear(requireContext())
            ApiClient.setAuthToken(null) // [FIX] Clear interceptor
            CookieManager.getInstance().removeAllCookies(null)
            CookieManager.getInstance().flush()
            updateTokenStatus()
            Toast.makeText(
                requireContext(),
                "Logged out: Token and cookies cleared",
                Toast.LENGTH_SHORT,
            ).show()
        }

        issueDebugTokenButton.setOnClickListener {
            issueAndSaveDebugToken()
        }
    }

    private fun updateTokenStatus() {
        val token = TokenStore.getAccessToken(requireContext())
        tokenStatusText.text = if (token.isNullOrBlank()) {
            "Status: No token"
        } else {
            "Status: Token is set"
        }
        // Ensure interceptor stays in sync with persistent store on UI refresh
        if (!token.isNullOrBlank()) ApiClient.setAuthToken(token)
    }

    private fun issueAndSaveDebugToken() {
        if (!BuildConfig.DEBUG) {
            Toast.makeText(requireContext(), "Debug build only.", Toast.LENGTH_SHORT).show()
            return
        }

        val adminKey = debugAdminKeyInput.text.toString().trim()
        val userId = debugUserIdInput.text.toString().trim().ifBlank { "android-e2e-user" }
        val audience = debugAudienceInput.text.toString().trim().ifBlank { "android-universe" }
        val ttl = debugTtlInput.text.toString().trim().toIntOrNull()?.coerceIn(1, 120) ?: 60

        if (adminKey.isBlank()) {
            debugIssueResultText.text = "Admin key is required."
            return
        }

        debugIssueResultText.text = "Issuing token..."
        issueDebugTokenButton.isEnabled = false

        lifecycleScope.launch {
            try {
                val response = ApiClient.debugTokenApiService.issueToken(
                    adminKey = adminKey,
                    requestBody = DebugTokenIssueRequest(
                        user_id = userId,
                        aud = audience,
                        ttl_minutes = ttl,
                    ),
                )
                if (response.isSuccessful) {
                    val token = normalizeToken(response.body()?.token.orEmpty())
                    if (token.isNotBlank()) {
                        TokenStore.saveAccessToken(requireContext(), token)
                        // [FIX] Update interceptor immediately after issue
                        ApiClient.setAuthToken(token) 
                        updateTokenStatus()
                        debugIssueResultText.text = "Issued and saved. aud=$audience ttl=${ttl}m"
                        Toast.makeText(requireContext(), "Debug token applied.", Toast.LENGTH_SHORT).show()
                    } else {
                        debugIssueResultText.text = "Issued response has empty token."
                    }
                } else {
                    val detail = response.errorBody()?.string()?.take(180) ?: "unknown error"
                    debugIssueResultText.text = "Issue failed: HTTP ${response.code()} $detail"
                }
            } catch (e: Exception) {
                debugIssueResultText.text = "Issue failed: ${e.message}"
            } finally {
                issueDebugTokenButton.isEnabled = true
            }
        }
    }

    private fun normalizeToken(raw: String): String {
        val trimmed = raw.trim()
        return if (trimmed.startsWith("Bearer ", ignoreCase = true)) {
            trimmed.substringAfter(" ").trim()
        } else {
            trimmed
        }
    }
}
