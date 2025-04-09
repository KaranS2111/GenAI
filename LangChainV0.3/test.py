from google.generativeai import list_models
import google.generativeai as genai

genai.configure(api_key="AIzaSyBhiDFDXQbUjCgVdLuztKXDsMsKSROUBp8")

for m in list_models():
    print(m.name)
