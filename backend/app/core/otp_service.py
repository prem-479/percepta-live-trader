import random

def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp(email: str, otp: str):
    print(f"Sending OTP to {email}: {otp}")
    return True
