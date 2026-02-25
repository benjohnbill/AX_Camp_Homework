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
import androidx.navigation.fragment.findNavController

class HomeFragment : Fragment(R.layout.fragment_home) {

    private lateinit var writeNarrativeButton: Button
    private lateinit var debugToolsContainer: LinearLayout
    private lateinit var tokenStatusText: TextView
    private lateinit var tokenInput: EditText
    private lateinit var saveTokenButton: Button
    private lateinit var clearTokenButton: Button
    private lateinit var logoutButton: Button

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        writeNarrativeButton = view.findViewById(R.id.button_write_narrative)
        debugToolsContainer = view.findViewById(R.id.debug_tools_container)
        tokenStatusText = view.findViewById(R.id.token_status_text)
        tokenInput = view.findViewById(R.id.token_input_edittext)
        saveTokenButton = view.findViewById(R.id.save_token_button)
        clearTokenButton = view.findViewById(R.id.clear_token_button)
        logoutButton = view.findViewById(R.id.logout_button)

        writeNarrativeButton.setOnClickListener {
            findNavController().navigate(R.id.action_homeFragment_to_createNarrativeFragment)
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
            TokenStore.clear(requireContext())
            CookieManager.getInstance().removeAllCookies(null)
            CookieManager.getInstance().flush()
            updateTokenStatus()
            Toast.makeText(
                requireContext(),
                "Logged out: Token and cookies cleared",
                Toast.LENGTH_SHORT,
            ).show()
        }
    }

    private fun updateTokenStatus() {
        val token = TokenStore.getAccessToken(requireContext())
        tokenStatusText.text = if (token.isNullOrBlank()) {
            "Status: No token"
        } else {
            "Status: Token is set"
        }
    }
}
