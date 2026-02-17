import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent dir to path to import narrative_logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import narrative_logic as logic

class TestRefactor(unittest.TestCase):
    def test_policy_silence_logic(self):
        engine = logic.PolicyEngine()
        
        # Case A: 25 hours silent (UTC)
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=25)
        
        msg = engine.evaluate_silence(past, now)
        print(f"Case A (25h): {msg}")
        self.assertIsNotNone(msg)
        self.assertIn("25시간", msg)
        
        # Case B: 10 hours silent (No alert)
        past = now - timedelta(hours=10)
        msg = engine.evaluate_silence(past, now)
        self.assertIsNone(msg)
        
        # Case C: 73+ hours (Critical)
        past = now - timedelta(hours=75)
        msg = engine.evaluate_silence(past, now)
        print(f"Case C (75h): {msg}")
        self.assertIn("우주가 차갑게", msg)

    def test_gateway_sanitization(self):
        gateway = logic.EvidenceGateway()
        
        # Case: Trim
        self.assertEqual(gateway.sanitize_input("  hello  "), "hello")
        
        # Case: Max Length
        long_text = "a" * 10005
        sanitized = gateway.sanitize_input(long_text)
        self.assertEqual(len(sanitized), 10000)

    @patch('narrative_logic.get_client')
    def test_circuit_breaker(self, mock_get_client):
        gateway = logic.EvidenceGateway()
        
        # Mock API to fail always
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("API Down")
        mock_get_client.return_value = mock_client
        
        # 1. First Failure
        res = gateway.get_embedding_safe("test")
        self.assertIsNone(res)
        self.assertEqual(gateway.api_failure_count, 1)
        
        # 2. Trigger Breaker (Max failures = 3)
        gateway.api_failure_count = 3
        res = gateway.get_embedding_safe("test")
        self.assertTrue(gateway.is_degraded_mode)
        
        # 3. Degraded Mode (Should skip API call)
        # Reset mock to pass, but gateway should skip it
        mock_client.embeddings.create.side_effect = None
        res = gateway.get_embedding_safe("test")
        self.assertIsNone(res) # Still None because degraded

if __name__ == '__main__':
    unittest.main()
