import tkinter as tk
from tkinter import scrolledtext
import subprocess
import os
import signal

running_process = None  # Untuk menyimpan proses luar

def log_output(message):
    result_box.insert(tk.END, message + "\n")
    result_box.see(tk.END)

def start_action():
    global running_process
    if running_process is None or running_process.poll() is not None:
        log_output("🟢 Menjalankan sistem...")
        running_process = subprocess.Popen(["python3", "system.py"])
        log_output(f"▶️ PID: {running_process.pid}")
    else:
        log_output("⚠️ Sistem sudah berjalan.")

def stop_action():
    global running_process
    if running_process and running_process.poll() is None:
        log_output("🛑 Menghentikan sistem...")
        running_process.terminate()
        running_process.wait()
        log_output("✅ Sistem dihentikan.")
        running_process = None
    else:
        log_output("⚠️ Tidak ada sistem yang berjalan.")

def kill_ui():
    stop_action()
    log_output("🧨 System Juga dimatikan.")
    log_output("🧨 UI dimatikan oleh pengguna.")
    root.destroy()

# Setup UI
root = tk.Tk()
root.title("Control Panel Sistem Kehadiran")
root.geometry("650x350")

# Kiri: Tombol-tombol
frame_left = tk.Frame(root, width=180)
frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

tk.Button(frame_left, text="Start System", command=start_action, height=2, bg="lightgreen").pack(pady=5, fill=tk.X)
tk.Button(frame_left, text="Stop System", command=stop_action, height=2, bg="lightblue").pack(pady=5, fill=tk.X)
tk.Button(frame_left, text="Stop UI", command=kill_ui, height=2, bg="red", fg="white").pack(pady=5, fill=tk.X)

# Kanan: Log
frame_right = tk.Frame(root)
frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

result_box = scrolledtext.ScrolledText(frame_right, wrap=tk.WORD, font=("Courier", 10))
result_box.pack(fill=tk.BOTH, expand=True)

root.protocol("WM_DELETE_WINDOW", kill_ui)
# Ignore Ctrl+Z (SIGTSTP)
signal.signal(signal.SIGTSTP, kill_ui)  # Hanya bekerja di Unix/Linux

root.mainloop()