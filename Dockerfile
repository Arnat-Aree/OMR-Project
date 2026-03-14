# ใช้ Python เวอร์ชันเล็กเพื่อความเบา
FROM python:3.10-slim

WORKDIR /app

# คัดลอกและติดตั้ง Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดทั้งหมดลง Container
COPY . .

# เปิดพอร์ต 5000 สำหรับ Flask
EXPOSE 5000

# รันแอป
CMD ["python", "app.py"]