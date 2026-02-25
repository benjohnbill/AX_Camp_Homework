package com.example.narrativeloopmobile

import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.webkit.CookieManager
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import androidx.fragment.app.Fragment

class Universe3DFragment : Fragment(R.layout.fragment_universe_3d) {

    private lateinit var webView: WebView
    // Use the official staging gateway URL
    private val dashboardUrl = "https://ax-camp-universe-gateway-staging.onrender.com/gateway/universe_3d"

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        webView = view.findViewById(R.id.universe_webview)
        configureWebView()
        loadUrlWithAuth()
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
        }
    }

    private fun loadUrlWithAuth() {
        val token = TokenStore.getAccessToken(requireContext())
        val headers = mutableMapOf<String, String>()
        if (!token.isNullOrBlank()) {
            headers["Authorization"] = "Bearer $token"
        }
        webView.loadUrl(dashboardUrl, headers)
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
            // Re-validate auth/session state when returning to the fragment.
            webView.reload()
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
