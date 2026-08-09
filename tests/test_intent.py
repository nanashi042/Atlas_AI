import unittest

from app.agent.intent import detect_intent, Intent


class TestIntentDetection(unittest.TestCase):

    def test_greetings(self):
        self.assertEqual(detect_intent("hi"), Intent.GREETING)
        self.assertEqual(detect_intent("Hello!"), Intent.GREETING)
        self.assertEqual(detect_intent("hey"), Intent.GREETING)
        self.assertEqual(detect_intent("good morning"), Intent.GREETING)

    def test_company_research_queries(self):
        queries = [
            "Tell me about Nvidia",
            "What is Tesla?",
            "Research Microsoft",
            "How is Apple doing?",
            "Give me information about NVDA",
            "What is happening with TSLA?",
            "What is NVDA?",
            "Analyze AAPL",
            "Company info on MSFT",
            "Overview of GOOGL",
            "Tell me about $AMZN",
        ]
        for query in queries:
            intent = detect_intent(query)
            self.assertEqual(
                intent,
                Intent.COMPANY_RESEARCH,
                f"Failed for query: '{query}' (got '{intent}')"
            )

    def test_general_chat_queries(self):
        queries = [
            "What is inflation?",
            "How do stock options work?",
            "Explain compound interest",
            "Tell me a funny financial joke",
        ]
        for query in queries:
            intent = detect_intent(query)
            self.assertEqual(
                intent,
                Intent.GENERAL_CHAT,
                f"Failed for query: '{query}' (got '{intent}')"
            )


if __name__ == "__main__":
    unittest.main()
