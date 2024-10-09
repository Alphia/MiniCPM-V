from unittest import TestCase

from message_utils import convert_to_cpm_v_26


class Test(TestCase):
    def test_convert_to_cpm_v_26(self):
        open_ai_messages = [
            {
                "role": "system",
                "content": "explain image"
            },
            {
                "role": "system",
                "content": ["explain image again", "explain image again 3"]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content": "i don't know how to"
            }
        ]
        v26_messages = convert_to_cpm_v_26(open_ai_messages)
        self.assertEqual(len(v26_messages[2]['content']), 2)
        print(v26_messages)
