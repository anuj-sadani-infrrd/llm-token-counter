from google import genai
client = genai.Client(api_key="YOUR_API_KEY")

# This counts text, image, and video together
response = client.models.count_tokens(
    model="gemini-2.0-flash",
    contents=[
        "Describe this video and image:",
        genai.types.Part.from_uri(file_uri="gs://bucket/video.mp4", mime_type="video/mp4"),
        genai.types.Part.from_uri(file_uri="gs://bucket/image.jpg", mime_type="image/jpeg")
    ]
)
print(f"Total Tokens: {response.total_tokens}")