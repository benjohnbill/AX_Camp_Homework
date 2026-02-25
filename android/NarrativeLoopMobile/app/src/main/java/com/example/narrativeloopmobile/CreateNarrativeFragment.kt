package com.example.narrativeloopmobile

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.example.narrativeloopmobile.network.NarrativeRepository
import kotlinx.coroutines.launch

class CreateNarrativeFragment : Fragment() {

    private lateinit var narrativeEditText: EditText
    private lateinit var saveNarrativeButton: Button
    private val narrativeRepository = NarrativeRepository()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_create_narrative, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        narrativeEditText = view.findViewById(R.id.edit_text_narrative)
        saveNarrativeButton = view.findViewById(R.id.button_save_narrative)

        saveNarrativeButton.setOnClickListener {
            val narrativeText = narrativeEditText.text.toString()
            if (narrativeText.isNotBlank()) {
                lifecycleScope.launch {
                    try {
                        val response = narrativeRepository.saveNarrative(narrativeText)
                        if (response.isSuccessful) {
                            Toast.makeText(requireContext(), "Narrative saved!", Toast.LENGTH_SHORT).show()
                            parentFragmentManager.popBackStack()
                        } else {
                            Toast.makeText(requireContext(), "Error: ${response.code()}", Toast.LENGTH_SHORT).show()
                        }
                    } catch (e: Exception) {
                        Toast.makeText(requireContext(), "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            } else {
                Toast.makeText(requireContext(), "Narrative cannot be empty.", Toast.LENGTH_SHORT).show()
            }
        }
    }
}
