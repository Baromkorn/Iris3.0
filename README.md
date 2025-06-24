# 👁️ Iris Recognition API (FastAPI + Milvus)

A biometric authentication system using iris templates, powered by FastAPI and Milvus. Supports user enrollment, verification, and iris-based search.

---

## 🚀 Features

- Iris template extraction using OpenCV and open-iris pipeline
- Iris matching using Hamming distance matcher from open-iris
- Milvus vector database for fast vector search paired with open-iris matcher for higher accuracy
- REST API built with FastAPI
- Support for single/both-eye processing

---

## 📂 Project Structure

```
Iris3.0/
├── core/
│   └── iris_setup.py
├── services_logic/
│   └── iris_service.py
├── Database/
│   ├── DatabaseEnroll/
│   ├── DatabaseSearch/
│   ├── DatabaseVerify/
│   └── DatabaseCheck/
├── utils/
│   ├── file_utils.py
│   └── iris_utils.py
|   └── SaveImage.py
├── routers/
│   └── iris_endpoints.py
├── misc_utils/
|   └──AutoEnrollTest/
|   └──DatabaseCreate/
├── main.py
└── README.md
└── docker-compse.yml
└── requirements.txt
```

---

## 🧪 API Endpoints

| Method | Endpoint                          | Description                         |
|--------|-----------------------------------|-------------------------------------|
| POST   | `/`                               | Enroll user with iris image(s)      |
| POST   | `/verify/`                        | Verify user identity via iris       |
| POST   | `/search/`                        | Search top matches in database      |
| GET    | `/check_available/{cid_or_pcode}`| Check ID or code availability       |

---

## 🛠️ Installation & Setup

```bash
# Clone the repo
git clone https://github.com/Baromkorn/Iris3.0.git
cd Iris3.0

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Docker for Milvus Database (optional, if already running no need to start again)
# Make sure you have Docker and Docker Compose installed:
docker --version
docker-compose --version
# Start the containers
docker compose up -d
# After done using, for stopping containers use
docker-compose down

# Start FastAPI server
source .venv/bin/activate  # or .venv\Scripts\activate on Windows (Make sure virtual environment is activated!)
fastapi dev main.py --host 0.0.0.0 --port 8000 # or whatever port you are using

# for fastAPI docs visit Visit: http://localhost:8000/docs or whatever port you are using
```

---

## 🧠 Technologies Used

- FastAPI ⚡
- Milvus DB 🧠
- Python OpenCV
- NumPy
- Hamming Distance Matcher
- AsyncIO for fast IO
- Open-Iris from World Coin

---

## 📁 Saving Images

Uploaded images are saved in the format:
```
root_path/YYYY/MM/DD/{cid}/auto_generated_L.bmp / _R.bmp
```

---

## 📄 License

This project is open-source and licensed under the MIT License.

---

## 👤 Authors

- Baromkorn Wannasarnmaytha (Co-Developers)
- Teekatat Piriyapittaya (Co-Developers)
