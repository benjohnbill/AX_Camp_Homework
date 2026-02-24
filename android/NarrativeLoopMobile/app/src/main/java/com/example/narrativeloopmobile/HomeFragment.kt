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

class HomeFragment : Fragment(R.layout.fragment_home) {

    private lateinit var debugToolsContainer: LinearLayout
    private lateinit var tokenStatusText: TextView
    private lateinit var tokenInput: EditText
    private lateinit var saveTokenButton: Button
    private lateinit var clearTokenButton: Button
    private lateinit var logoutButton: Button

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        debugToolsContainer = view.findViewById(R.id.debug_tools_container)
        tokenStatusText = view.findViewById(R.id.token_status_text)
        tokenInput = view.findViewById(R.id.token_input_edittext)
        saveTokenButton = view.findViewById(R.id.save_token_button)
        clearTokenButton = view.findViewById(R.id.clear_token_button)
        logoutButton = view.findViewById(R.id.logout_button)

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
            val token = tokenInput.text.toString()
            if (token.isNotBlank()) {
                TokenStore.saveAccessToken(requireContext(), token)
                tokenInput.text.clear()
                updateTokenStatus()
            }
        }

        clearTokenButton.setOnClickListener {
            TokenStore.clear(requireContext())
            updateTokenStatus()
        }

        logoutButton.setOnClickListener {
            // Clear token from SharedPreferences
            TokenStore.clear(requireContext())

            // Clear all WebView cookies
            CookieManager.getInstance().removeAllCookies(null)
            CookieManager.getInstance().flush()

            updateTokenStatus()
            Toast.makeText(requireContext(), "Logged out: Token and cookies cleared", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateTokenStatus() {
        val token = TokenStore.getAccessToken(requireContext())
        if (token.isNullOrBlank()) {
            tokenStatusText.text = "Status: No token"
        } else {
            tokenStatusText.text = "Status: Token is set"
        }
    }
}