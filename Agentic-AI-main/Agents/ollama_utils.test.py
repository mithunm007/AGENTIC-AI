import unittest
from unittest.mock import patch

from ollama_utils import process_with_groq, process_with_ollama

def speak(text):
    """Mock function to simulate speaking"""
    print(f"[Speaking]: {text}")

def handle_response_stream(response):
    """
    Handles the response stream, speaking the output in chunks.
    """
    paragraph_buffer = ""
    
    for chunk in response:
        yield chunk['message']['content']
        if ("." in paragraph_buffer) and (len(paragraph_buffer) > 15):
            print(paragraph_buffer)
            speak(paragraph_buffer)
            paragraph_buffer = ""
        
        paragraph_buffer += chunk['message']['content']

    if paragraph_buffer:
        print(paragraph_buffer)
        speak(paragraph_buffer)
        paragraph_buffer = ""







# testing files 
class TestHandleResponseStream(unittest.TestCase):
    
    @patch("builtins.print")
    @patch("__main__.speak")
    def test_handle_response_stream(self, mock_speak, mock_print):
        response = [
            {'message': {'content': "Hello, this is a test. "}},
            {'message': {'content': "It should be spoken in chunks. "}},
            {'message': {'content': "Final sentence."}}
        ]
        
        output = list(handle_response_stream(response))
        
        self.assertEqual(output, [
            "Hello, this is a test. ",
            "It should be spoken in chunks. ",
            "Final sentence."
        ])
        
        mock_speak.assert_called()
        mock_print.assert_called()


#GROQ test
class TestProcessWithGroq(unittest.TestCase):
    def test_process_with_groq(self):
        query = "what is the capital of france"
        
        response = process_with_groq(query)
        if "paris" in response.lower():
            self.answer = True
        else:
            self.answer = False
        self.assertEqual(self.answer, True)


#MISTRAL test
class TestProcessWithMistral(unittest.TestCase):
    def test_process_with_mistral(self):
        query = "what is the capital of india"
        
        response = process_with_ollama(query = query)
        result = list(handle_response_stream(response))
        if " Delhi" in result:
            self.answer = True
        else:
            self.answer = False
        self.assertEqual(self.answer, True)

if __name__ == "__main__":
    unittest.main()
