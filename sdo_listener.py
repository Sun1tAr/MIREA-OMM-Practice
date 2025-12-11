import requests
import time
from datetime import datetime

URI = "https://online-edu.mirea.ru/"

# ANSI цвета
GREEN = '\033[92m'
ORANGE = '\033[33m'
PURPLE = '\033[35m'
RESET = '\033[0m'

def send_requests():
    while True:
        try:
            response = requests.get(URI, timeout=5)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            status = response.status_code
            if 200 <= status < 300:
                color = GREEN
                status_text = "Доступен"
            elif 500 <= status < 600:
                color = PURPLE
                status_text = "Сервер недоступен"
            else:
                color = ORANGE
                status_text = "Другая ошибка"

            print(f"[{timestamp}] GET {URI}")
            print(f"{color}Status: {status} - {status_text}{RESET}")
            # print(f"Response: {response.text[:200]}\n")
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {PURPLE}Error: {e}{RESET}")

        time.sleep(5)

if __name__ == "__main__":
    print("Запуск... (Ctrl+C для остановки)\n")
    send_requests()