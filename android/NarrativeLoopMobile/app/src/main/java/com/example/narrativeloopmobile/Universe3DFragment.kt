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
import android.widget.Toast
import androidx.fragment.app.Fragment

class Universe3DFragment : Fragment(R.layout.fragment_universe_3d) {

    private lateinit var webView: WebView

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        webView = view.findViewById(R.id.universe_webview)
        configureWebView()
        loadUniverseWithAuth()
    }

    private fun configureWebView() {
        val s = webView.settings
        s.javaScriptEnabled = true
        s.domStorageEnabled = true
        s.loadsImagesAutomatically = true
        s.mediaPlaybackRequiresUserGesture = false
        s.cacheMode = WebSettings.LOAD_DEFAULT
        s.mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW

        val cookieManager = CookieManager.getInstance()
        cookieManager.setAcceptCookie(true)
        cookieManager.setAcceptThirdPartyCookies(webView, true)

        webView.webChromeClient = WebChromeClient()
        webView.webViewClient = object : WebViewClient() {
            override fun onReceivedHttpError(
                view: WebView?,
                request: WebResourceRequest?,
                errorResponse: WebResourceResponse?
            ) {
                if (request?.isForMainFrame == true) {
                    when (errorResponse?.statusCode) {
                        401, 403 -> {
                            Toast.makeText(
                                requireContext(),
                                "Authentication failed. Please login again.",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    }
                }
                super.onReceivedHttpError(view, request, errorResponse)
            }
        }

        webView.setLayerType(View.LAYER_TYPE_HARDWARE, null)
    }

    private fun loadUniverseWithAuth() {
        val universeUrl = BuildConfig.UNIVERSE_URL
        if (universeUrl.isBlank()) {
            Toast.makeText(requireContext(), "Universe URL is not configured.", Toast.LENGTH_LONG).show()
            return
        }

        val token = TokenStore.getAccessToken(requireContext())

        if (token.isNullOrBlank()) {
            // No token, try loading with existing cookie session
            webView.loadUrl(universeUrl)
        } else {
            // Token exists, use it for initial authentication
            val headers = mapOf("Authorization" to "Bearer $token")
            webView.loadUrl(universeUrl, headers)
        }
    }

    override fun onPause() {
        super.onPause()
        webView.onPause()
        webView.pauseTimers()
    }

    override fun onResume() {
        super.onResume()
        webView.onResume()
        webView.resumeTimers()
        // Reload to check auth state on return
        webView.reload()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        webView.stopLoading()
        (webView.parent as? ViewGroup)?.removeView(webView)
        webView.destroy()
    }
}