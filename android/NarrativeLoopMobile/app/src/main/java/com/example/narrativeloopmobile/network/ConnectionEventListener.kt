package com.example.narrativeloopmobile.network

import android.util.Log
import okhttp3.Call
import okhttp3.Connection
import okhttp3.EventListener
import okhttp3.Handshake
import okhttp3.HttpUrl
import okhttp3.Protocol
import java.io.IOException
import java.net.InetAddress
import java.net.Proxy

class ConnectionEventListener : EventListener() {
    private val TAG = "OkHttpEvent"

    override fun dnsStart(call: Call, domainName: String) {
        Log.d(TAG, "dnsStart: $domainName")
    }

    override fun dnsEnd(call: Call, domainName: String, inetAddressList: List<InetAddress>) {
        Log.d(TAG, "dnsEnd: $inetAddressList")
    }

    override fun connectStart(call: Call, inetSocketAddress: java.net.InetSocketAddress, proxy: Proxy) {
        Log.d(TAG, "connectStart: $inetSocketAddress proxy: $proxy")
    }

    override fun secureConnectStart(call: Call) {
        Log.d(TAG, "secureConnectStart")
    }

    override fun secureConnectEnd(call: Call, handshake: Handshake?) {
        Log.d(TAG, "secureConnectEnd: tlsVersion=${handshake?.tlsVersion} cipherSuite=${handshake?.cipherSuite}")
    }

    override fun connectEnd(call: Call, inetSocketAddress: java.net.InetSocketAddress, proxy: Proxy, protocol: Protocol?) {
        Log.d(TAG, "connectEnd: protocol=$protocol")
    }

    override fun connectFailed(call: Call, inetSocketAddress: java.net.InetSocketAddress, proxy: Proxy, protocol: Protocol?, ioe: IOException) {
        Log.e(TAG, "connectFailed: ${ioe.message}")
    }

    override fun connectionAcquired(call: Call, connection: Connection) {
        Log.d(TAG, "connectionAcquired")
    }

    override fun connectionReleased(call: Call, connection: Connection) {
        Log.d(TAG, "connectionReleased")
    }

    override fun requestHeadersStart(call: Call) {
        Log.d(TAG, "requestHeadersStart")
    }

    override fun responseHeadersStart(call: Call) {
        Log.d(TAG, "responseHeadersStart")
    }

    override fun responseFailed(call: Call, ioe: IOException) {
        Log.e(TAG, "responseFailed: ${ioe.message}")
    }
}
