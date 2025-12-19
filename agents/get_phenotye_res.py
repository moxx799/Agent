import anthropic
import os

def get_response():
# Set your API key (or use environment variable)
    API_KEY = os.environ.get("ANTHROPIC_API_KEY") or "your-api-key-here"

    # Initialize the client
    client = anthropic.Anthropic(api_key=API_KEY)

    # Read the system prompt from a file
    with open("./sysprompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # Send request to Claude
    print("Sending request to Claude...")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system_prompt,
        messages=[
            {"role": "user", "content": "Please respond based on the system prompt provided."}
        ]
    )

    # Extract the response text
    response_text = response.content[0].text

    # Save to output file
    with open("claude_response.txt", "w", encoding="utf-8") as f:
        f.write(response_text)
        
    return None