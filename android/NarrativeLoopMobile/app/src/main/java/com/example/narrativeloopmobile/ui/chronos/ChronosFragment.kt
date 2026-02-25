package com.example.narrativeloopmobile.ui.chronos

import android.os.Bundle
import android.os.CountDownTimer
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.fragment.app.Fragment
import com.example.narrativeloopmobile.R
import java.util.concurrent.TimeUnit

class ChronosFragment : Fragment(R.layout.fragment_chronos) {

    private lateinit var timerText: TextView
    private lateinit var start25Btn: Button
    private lateinit var start60Btn: Button
    private lateinit var cancelBtn: Button
    private var countDownTimer: CountDownTimer? = null

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        timerText = view.findViewById(R.id.text_timer)
        start25Btn = view.findViewById(R.id.button_start_25)
        start60Btn = view.findViewById(R.id.button_start_60)
        cancelBtn = view.findViewById(R.id.button_cancel_timer)

        start25Btn.setOnClickListener { startTimer(25) }
        start60Btn.setOnClickListener { startTimer(60) }
        cancelBtn.setOnClickListener { cancelTimer() }
    }

    private fun startTimer(minutes: Int) {
        cancelTimer()
        val millis = minutes * 60 * 1000L
        
        countDownTimer = object : CountDownTimer(millis, 1000) {
            override fun onTick(millisUntilFinished: Long) {
                val min = TimeUnit.MILLISECONDS.toMinutes(millisUntilFinished)
                val sec = TimeUnit.MILLISECONDS.toSeconds(millisUntilFinished) % 60
                timerText.text = String.format("%02d:%02d", min, sec)
            }

            override fun onFinish() {
                timerText.text = "FINISHED"
                // TODO: Trigger docking dialog
            }
        }.start()
    }

    private fun cancelTimer() {
        countDownTimer?.cancel()
        timerText.text = "00:00"
    }
}
