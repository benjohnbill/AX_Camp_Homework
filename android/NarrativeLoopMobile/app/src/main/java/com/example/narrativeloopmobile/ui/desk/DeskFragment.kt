package com.example.narrativeloopmobile.ui.desk

import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.example.narrativeloopmobile.R
import com.example.narrativeloopmobile.network.NarrativeRepository
import kotlinx.coroutines.launch

class DeskFragment : Fragment(R.layout.fragment_desk) {

    private lateinit var logsContainer: LinearLayout
    private val repository = NarrativeRepository()

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        logsContainer = view.findViewById(R.id.logs_container)
        loadLogs()
    }

    private fun loadLogs() {
        lifecycleScope.launch {
            try {
                // In a real app, we would add a getLogs() method to Repository
                // For now, let's show a placeholder if we haven't implemented getLogs yet
                val placeholder = TextView(requireContext()).apply {
                    text = "Loading past narratives..."
                    setPadding(20, 20, 20, 20)
                }
                logsContainer.addView(placeholder)
                
                // TODO: Implement list fetching from API
            } catch (e: Exception) {
                // Handle error
            }
        }
    }
}
