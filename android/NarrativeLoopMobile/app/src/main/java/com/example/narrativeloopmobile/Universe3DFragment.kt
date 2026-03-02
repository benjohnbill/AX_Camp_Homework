package com.example.narrativeloopmobile

import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.webkit.CookieManager
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import com.google.android.material.bottomnavigation.BottomNavigationView

class Universe3DFragment : Fragment(R.layout.fragment_universe_3d) {

    private lateinit var webView: WebView
    private lateinit var authEmptyState: LinearLayout
    private lateinit var authMessage: TextView
    private lateinit var goHomeButton: Button
    private val dashboardUrl = BuildConfig.UNIVERSE_URL

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        webView = view.findViewById(R.id.universe_webview)
        authEmptyState = view.findViewById(R.id.universe_auth_empty_state)
        authMessage = view.findViewById(R.id.universe_auth_message)
        goHomeButton = view.findViewById(R.id.universe_go_home_button)

        goHomeButton.setOnClickListener {
            navigateToHomeForTokenInput()
        }

        configureWebView()
        evaluateAccessAndLoad()
    }

    private fun configureWebView() {
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            loadWithOverviewMode = true
            useWideViewPort = true
            mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
        }

        val cookieManager = CookieManager.getInstance()
        cookieManager.setAcceptCookie(true)
        cookieManager.setAcceptThirdPartyCookies(webView, true)

        webView.webChromeClient = WebChromeClient()
        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                // Let the WebView handle the redirect
                return false
            }

            override fun onReceivedHttpError(
                view: WebView?,
                request: WebResourceRequest?,
                errorResponse: WebResourceResponse?
            ) {
                super.onReceivedHttpError(view, request, errorResponse)
                if (request?.isForMainFrame == true && errorResponse?.statusCode == 401) {
                    showMissingTokenState(
                        "Universe 인증이 만료되었거나 누락되었습니다. Home에서 토큰을 저장한 뒤 다시 시도해 주세요."
                    )
                }
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?
            ) {
                super.onReceivedError(view, request, error)
                if (request?.isForMainFrame == true) {
                    Toast.makeText(requireContext(), "Universe 로딩 실패: ${error?.description}", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun evaluateAccessAndLoad() {
        val token = TokenStore.getAccessToken(requireContext())?.trim()
        if (token.isNullOrBlank()) {
            showMissingTokenState(
                "Universe 접근 토큰이 없습니다. Home에서 토큰을 저장한 뒤 다시 시도해 주세요."
            )
            return
        }
        hideMissingTokenState()
        loadUrlWithAuth(token)
    }

    private fun loadUrlWithAuth(token: String) {
        val headers = mutableMapOf<String, String>()
        headers["Authorization"] = "Bearer $token"
        webView.loadUrl(dashboardUrl, headers)
    }

    private fun showMissingTokenState(message: String) {
        authMessage.text = message
        authEmptyState.visibility = View.VISIBLE
        webView.visibility = View.INVISIBLE
    }

    private fun hideMissingTokenState() {
        authEmptyState.visibility = View.GONE
        webView.visibility = View.VISIBLE
    }

    private fun navigateToHomeForTokenInput() {
        val bottomNav = requireActivity().findViewById<BottomNavigationView>(R.id.bottom_nav)
        bottomNav.selectedItemId = R.id.nav_home
        Toast.makeText(requireContext(), "Home에서 토큰을 입력해 주세요.", Toast.LENGTH_SHORT).show()
    }

    override fun onPause() {
        super.onPause()
        if (::webView.isInitialized) {
            webView.onPause()
            webView.pauseTimers()
        }
    }

    override fun onResume() {
        super.onResume()
        if (::webView.isInitialized) {
            webView.onResume()
            webView.resumeTimers()
            evaluateAccessAndLoad()
        }
    }

    override fun onDestroyView() {
        if (::webView.isInitialized) {
            webView.stopLoading()
            (webView.parent as? ViewGroup)?.removeView(webView)
            webView.destroy()
        }
        super.onDestroyView()
    }
}
