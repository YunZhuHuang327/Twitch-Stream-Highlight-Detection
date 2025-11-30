"""
Test if OpenAI API key is valid
"""
import os
from openai import OpenAI

def test_api_key():
    """Test if the OpenAI API key works"""
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        return False

    print(f"Testing API key: {api_key[:8]}...{api_key[-4:]}")

    try:
        client = OpenAI(api_key=api_key)

        # Make a minimal test request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Say 'API works!' if you can read this."}
            ],
            max_tokens=10
        )

        result = response.choices[0].message.content
        print(f"[OK] API key is valid!")
        print(f"[OK] Response: {result}")
        print(f"[OK] Tokens used: {response.usage.total_tokens}")
        return True

    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "invalid" in error_str.lower():
            print(f"[ERROR] API key is invalid or expired")
            print(f"   Error: {e}")
            print("\nTo fix:")
            print("1. Go to https://platform.openai.com/account/api-keys")
            print("2. Generate a new API key")
            print("3. Set it: $env:OPENAI_API_KEY='your-new-key'")
        elif "429" in error_str:
            print(f"[ERROR] Rate limit exceeded")
            print(f"   Error: {e}")
        else:
            print(f"[ERROR] Error: {e}")
        return False

if __name__ == "__main__":
    test_api_key()
