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
            logsContainer.removeAllViews()
            try {
                addLine("Desk", "Loading session snapshot...")
                val todayResponse = repository.getTodaySession()
                if (!todayResponse.isSuccessful) {
                    addLine("Error", "Failed to fetch today's session. HTTP ${todayResponse.code()}")
                    return@launch
                }

                val todayBody = todayResponse.body()
                if (todayBody?.status != "ok" || todayBody.session == null) {
                    addLine("Today Session", "No session found for today.")
                    return@launch
                }

                val session = todayBody.session
                addLine("Today Session", "id=${session.id}")
                addLine("Flow Stage", session.flow_stage ?: "n/a")
                addLine("Plan Status", session.plan_status ?: "n/a")
                addLine("Entry Mode", session.entry_mode ?: "n/a")

                val insightsResponse = repository.getSessionInsights(session.id)
                if (!insightsResponse.isSuccessful) {
                    addLine("Insights", "Failed to load insights. HTTP ${insightsResponse.code()}")
                    return@launch
                }

                val insightsBody = insightsResponse.body()
                if (insightsBody?.status != "ok") {
                    addLine("Insights", "No insight payload available.")
                    return@launch
                }

                addLine("Insight Source", insightsBody.insight_source ?: "n/a")
                addLine("Next Action", insightsBody.insights?.next_action ?: "n/a")
                addLine("Similar Pattern", insightsBody.insights?.similar_pattern ?: "n/a")
                addLine("Auto Tags", insightsBody.auto_tags?.joinToString(", ").orEmpty().ifBlank { "n/a" })
            } catch (e: Exception) {
                addLine("Exception", e.message ?: "Unknown error")
            }
        }
    }

    private fun addLine(label: String, value: String) {
        val textView = TextView(requireContext()).apply {
            text = "$label: $value"
            setPadding(20, 14, 20, 14)
            textSize = 14f
            setTextColor(0xFFFFFFFF.toInt())
        }
        logsContainer.addView(textView)
    }
}
