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

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        webView = view.findViewById(R.id.universe_webview)
        configureWebView()
        loadUniverseWithAuth()
    }

    private fun configureWebView() {
        val settings = webView.settings
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true
        settings.loadsImagesAutomatically = true
        settings.mediaPlaybackRequiresUserGesture = false
        settings.cacheMode = WebSettings.LOAD_DEFAULT
        settings.mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW

        val cookieManager = CookieManager.getInstance()
        cookieManager.setAcceptCookie(true)
        cookieManager.setAcceptThirdPartyCookies(webView, true)

        webView.webChromeClient = WebChromeClient()
        webView.webViewClient = object : WebViewClient() {
            override fun onReceivedHttpError(
                view: WebView?,
                request: WebResourceRequest?,
                errorResponse: WebResourceResponse?,
            ) {
                if (request?.isForMainFrame == true) {
                    when (errorResponse?.statusCode) {
                        401, 403 -> {
                            Toast.makeText(
                                requireContext(),
                                "Authentication failed. Please login again.",
                                Toast.LENGTH_LONG,
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
            webView.loadUrl(universeUrl)
        } else {
            val headers = mapOf("Authorization" to "Bearer $token")
            webView.loadUrl(universeUrl, headers)
        }
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
