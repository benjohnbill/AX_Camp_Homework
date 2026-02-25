package com.example.narrativeloopmobile.ui.debug

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.example.narrativeloopmobile.BuildConfig
import com.example.narrativeloopmobile.R
import com.example.narrativeloopmobile.network.ApiClient
import kotlinx.coroutines.launch

class DebugFragment : Fragment() {

    private lateinit var tokenEditText: EditText
    private lateinit var sendRequestButton: Button
    private lateinit var responseTextView: TextView
    private lateinit var narrativeEditText: EditText
    private lateinit var saveNarrativeButton: Button

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_debug, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        tokenEditText = view.findViewById(R.id.edit_text_token)
        sendRequestButton = view.findViewById(R.id.button_send_request)
        responseTextView = view.findViewById(R.id.text_view_response)
        narrativeEditText = view.findViewById(R.id.edit_text_narrative)
        saveNarrativeButton = view.findViewById(R.id.button_save_narrative)

        sendRequestButton.setOnClickListener {
            val token = tokenEditText.text.toString()
            responseTextView.text = "Sending request..."

            lifecycleScope.launch {
                try {
                    ApiClient.setAuthToken(token)
                    val response = ApiClient.debugApiService.getUniverse(BuildConfig.UNIVERSE_URL)

                    val responseCode = response.code()
                    val resultText = if (response.isSuccessful || responseCode == 307) {
                        "Success: Received response with code $responseCode"
                    } else {
                        "Error: Received error response with code $responseCode"
                    }
                    responseTextView.text = resultText

                } catch (e: Exception) {
                    responseTextView.text = "Exception: ${e.message}"
                }
            }
        }

        saveNarrativeButton.setOnClickListener {
            val token = tokenEditText.text.toString()
            responseTextView.text = "Sending request..."

            lifecycleScope.launch {
                try {
                    ApiClient.setAuthToken(token)
                    val response = ApiClient.debugApiService.getUniverse(BuildConfig.UNIVERSE_URL)

                    val responseCode = response.code()
                    val resultText = if (response.isSuccessful || responseCode == 307) {
                        "Success: Received response with code $responseCode"
                    } else {
                        "Error: Received error response with code $responseCode"
                    }
                    responseTextView.text = resultText

                } catch (e: Exception) {
                    responseTextView.text = "Exception: ${e.message}"
                }
            }
        }
    }
}
