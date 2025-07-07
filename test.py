import google.generativeai as genai

genai.configure(api_key="AIzaSyCDkWv9Lw5FiIXTbjfZ4Q26VveoSeQvSgM")
model = genai.GenerativeModel('gemini-2.5-flash')
response = model.generate_content("Say hello from Gemini!")
print(response.text)