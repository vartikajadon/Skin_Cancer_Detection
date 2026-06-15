import requests, os

url = 'http://127.0.0.1:5000/api/predict'
image_path = r'C:\\Users\\varti\\.gemini\\antigravity-ide\\brain\\a0a934cc-4830-4ebf-bd7c-bfbefe125bcf\\test_skin_image_1781082999305.png'
if not os.path.exists(image_path):
    raise FileNotFoundError(f'Image not found: {image_path}')

with open(image_path, 'rb') as f:
    files = {'file': (os.path.basename(image_path), f, 'image/png')}
    response = requests.post(url, files=files)
    print('Status code:', response.status_code)
    print('Response JSON:', response.json())
