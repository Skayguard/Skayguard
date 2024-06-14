import cv2
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import os
from collections import deque

# Configurações iniciais
azimuth = 0
elevation = 0
latitude = -23.5505
longitude = -46.6333
lens_type = "default"
fov = 90  # Campo de Visão padrão em graus
cap = None
save_dir = ""
video_writer = None
is_recording = False
azimuth_adjustment = 0
elevation_adjustment = 0
magnitude_threshold = 6.0
size_threshold = 500  # Tamanho mínimo para detecção
brightness_threshold = 150  # Brilho mínimo para detecção
current_frame = None
buffer_size = 20  # Buffer para 10 segundos antes e depois
frame_rate = 2  # Frame rate para 10 segundos de vídeo
frame_buffer = deque(maxlen=buffer_size)
selected_camera = 0

def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

def configure_camera(camera_index):
    global cap, video_writer
    print("Configuring camera...")  # Adiciona mensagem de log para diagnóstico
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        messagebox.showerror("Erro", "Não foi possível abrir o dispositivo de vídeo")
        print("Failed to open video device")  # Adiciona mensagem de log para diagnóstico
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camera configured successfully")  # Adiciona mensagem de log para diagnóstico
    return True

def capture_frames():
    global cap, video_writer, is_recording, azimuth_adjustment, elevation_adjustment, magnitude_threshold, size_threshold, brightness_threshold, current_frame, frame_buffer
    if cap is None or not cap.isOpened():
        print("Capture failed: camera not opened")  # Adiciona mensagem de log para diagnóstico
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera")  # Adiciona mensagem de log para diagnóstico
        return
    
    current_frame = frame.copy()  # Armazena o quadro atual para snapshot

    # Processar frame para detecção de objetos (exemplo simplificado)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, brightness_threshold, brightness_threshold * 3)  # Use brightness threshold
    
    # Detectar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected = False
    for contour in contours:
        if cv2.contourArea(contour) > size_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            object_brightness = np.mean(frame[y:y+h, x:x+w])
            if object_brightness > brightness_threshold:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Calcular posição do objeto no céu
                time_now = Time(datetime.utcnow())
                location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg)
                altaz = AltAz(location=location, obstime=time_now)
                
                x_center = x + w / 2
                y_center = y + h / 2
                altitude = ((y_center / frame.shape[0]) * fov - (fov / 2)) + elevation_adjustment
                azimuth = ((x_center / frame.shape[1]) * fov - (fov / 2)) + azimuth_adjustment
                
                sky_coord = SkyCoord(alt=altitude * u.deg, az=azimuth * u.deg, frame=altaz)
                
                # Detecção de magnitude (placeholder simplificado, substitua pela lógica de detecção real)
                magnitude = np.random.uniform(-27, 30)  # Magnitude aleatória para exemplo
                if magnitude <= magnitude_threshold:
                    detected = True
                    # Exibir dados do objeto
                    text = f"Lente: {lens_type}, Azimute: {sky_coord.az.deg:.2f}, Elevação: {sky_coord.alt.deg:.2f}, Magnitude: {magnitude:.2f}"
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Salvar parâmetros em arquivo TXT
                    save_parameters_to_file(sky_coord, magnitude)
    
    # Adicionar data e hora no frame
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Adicionar frame ao buffer
    frame_buffer.append(frame.copy())

    # Salvar frame em vídeo se estiver gravando
    if is_recording and video_writer is not None:
        video_writer.write(frame)
    
    # Converter frame para formato ImageTk
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl_video.imgtk = imgtk
    lbl_video.configure(image=imgtk)
    lbl_video.after(10, capture_frames)

    # Salvar snapshot automaticamente
    if detected:
        save_snapshot()
        save_video_segment()

def start_capture():
    if configure_camera(selected_camera):
        capture_frames()

def stop_capture():
    global cap, is_recording
    is_recording = False
    if cap is not None:
        cap.release()
        cap = None
    lbl_video.config(image='')

def save_parameters():
    if not save_dir:
        return
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'lens_type': lens_type,
        'fov': fov,
        'azimuth_adjustment': azimuth_adjustment,
        'elevation_adjustment': elevation_adjustment,
        'magnitude_threshold': magnitude_threshold,
        'size_threshold': size_threshold,
        'brightness_threshold': brightness_threshold
    }
    with open(os.path.join(save_dir, 'parameters.txt'), 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

def save_snapshot():
    if save_dir and current_frame is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot = current_frame.copy()
        cv2.putText(snapshot, timestamp, (10, snapshot.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(save_dir, f'snapshot_{timestamp}.png'), snapshot)

def save_video_segment():
    if save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        segment_writer = cv2.VideoWriter(os.path.join(save_dir, f'segment_{timestamp}.avi'), fourcc, frame_rate, (1280, 720))
        for frame in frame_buffer:
            segment_writer.write(frame)
        segment_writer.release()

def save_parameters_to_file(sky_coord, magnitude):
    if save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(save_dir, f'object_{timestamp}.txt'), 'w') as f:
            f.write(f"Data e Hora: {timestamp}\n")
            f.write(f"Latitude: {latitude}\n")
            f.write(f"Longitude: {longitude}\n")
            f.write(f"Tipo de Lente: {lens_type}\n")
            f.write(f"Campo de Visão: {fov}\n")
            f.write(f"Ajuste de Azimute: {azimuth_adjustment}\n")
            f.write(f"Ajuste de Elevação: {elevation_adjustment}\n")
            f.write(f"Limite de Magnitude: {magnitude_threshold}\n")
            f.write(f"Limite de Tamanho: {size_threshold}\n")
            f.write(f"Limite de Brilho: {brightness_threshold}\n")
            f.write(f"Azimute: {sky_coord.az.deg:.2f}\n")
            f.write(f"Elevação: {sky_coord.alt.deg:.2f}\n")
            f.write(f"Magnitude: {magnitude:.2f}\n")

def apply_parameters(preset):
    global magnitude_threshold, size_threshold, brightness_threshold
    if preset == "Objetos brilhantes (-27 a 0)":
        magnitude_threshold = 0
        size_threshold = 500
        brightness_threshold = 150
    elif preset == "Objetos moderadamente brilhantes (0 a +6)":
        magnitude_threshold = 6
        size_threshold = 300
        brightness_threshold = 100
    elif preset == "Objetos fracos (+6 a +10)":
        magnitude_threshold = 10
        size_threshold = 200
        brightness_threshold = 75
    elif preset == "Objetos muito fracos (+10 a +30)":
        magnitude_threshold = 30
        size_threshold = 100
        brightness_threshold = 50

def main():
    global lbl_video, latitude, longitude, lens_type, fov, save_dir, is_recording, selected_camera
    global azimuth_adjustment, elevation_adjustment, magnitude_threshold, size_threshold, brightness_threshold
    
    root = tk.Tk()
    root.title("OVNI Detector")
    
    def select_save_directory():
        global save_dir
        save_dir = filedialog.askdirectory()
        if save_dir:
            messagebox.showinfo("Diretório Selecionado", f"Salvando em: {save_dir}")

    def update_camera_selection(event):
        global selected_camera
        selected_camera = int(camera_combo.get())

    cameras = list_cameras()
    
    # Layout horizontal para campos de entrada e botões de controle
    top_frame = ttk.Frame(root)
    top_frame.pack(padx=10, pady=10)

    # Seleção de câmera
    lbl_camera = ttk.Label(top_frame, text="Selecione a Câmera:")
    lbl_camera.grid(row=0, column=0, padx=5, pady=5)
    camera_combo = ttk.Combobox(top_frame, values=cameras)
    camera_combo.grid(row=0, column=1, padx=5, pady=5)
    camera_combo.current(0)
    camera_combo.bind("<<ComboboxSelected>>", update_camera_selection)
    
    # Campos de entrada
    lbl_lat = ttk.Label(top_frame, text="Latitude:")
    lbl_lat.grid(row=1, column=0, padx=5, pady=5)
    ent_lat = ttk.Entry(top_frame)
    ent_lat.grid(row=1, column=1, padx=5, pady=5)
    ent_lat.insert(0, str(latitude))
    
    lbl_lon = ttk.Label(top_frame, text="Longitude:")
    lbl_lon.grid(row=1, column=2, padx=5, pady=5)
    ent_lon = ttk.Entry(top_frame)
    ent_lon.grid(row=1, column=3, padx=5, pady=5)
    ent_lon.insert(0, str(longitude))
    
    lbl_lens = ttk.Label(top_frame, text="Tipo de Lente:")
    lbl_lens.grid(row=1, column=4, padx=5, pady=5)
    ent_lens = ttk.Entry(top_frame)
    ent_lens.grid(row=1, column=5, padx=5, pady=5)
    ent_lens.insert(0, lens_type)
    
    lbl_fov = ttk.Label(top_frame, text="Campo de Visão (°):")
    lbl_fov.grid(row=1, column=6, padx=5, pady=5)
    ent_fov = ttk.Entry(top_frame)
    ent_fov.grid(row=1, column=7, padx=5, pady=5)
    ent_fov.insert(0, str(fov))
    
    lbl_az_adjust = ttk.Label(top_frame, text="Ajuste de Azimute:")
    lbl_az_adjust.grid(row=2, column=0, padx=5, pady=5)
    ent_az_adjust = ttk.Entry(top_frame)
    ent_az_adjust.grid(row=2, column=1, padx=5, pady=5)
    ent_az_adjust.insert(0, str(azimuth_adjustment))
    
    lbl_el_adjust = ttk.Label(top_frame, text="Ajuste de Elevação:")
    lbl_el_adjust.grid(row=2, column=2, padx=5, pady=5)
    ent_el_adjust = ttk.Entry(top_frame)
    ent_el_adjust.grid(row=2, column=3, padx=5, pady=5)
    ent_el_adjust.insert(0, str(elevation_adjustment))
    
    lbl_mag_threshold = ttk.Label(top_frame, text="Limite de Magnitude:")
    lbl_mag_threshold.grid(row=2, column=4, padx=5, pady=5)
    ent_mag_threshold = ttk.Entry(top_frame)
    ent_mag_threshold.grid(row=2, column=5, padx=5, pady=5)
    ent_mag_threshold.insert(0, str(magnitude_threshold))
    
    lbl_size_threshold = ttk.Label(top_frame, text="Limite de Tamanho:")
    lbl_size_threshold.grid(row=2, column=6, padx=5, pady=5)
    ent_size_threshold = ttk.Entry(top_frame)
    ent_size_threshold.grid(row=2, column=7, padx=5, pady=5)
    ent_size_threshold.insert(0, str(size_threshold))
    
    lbl_brightness_threshold = ttk.Label(top_frame, text="Limite de Brilho:")
    lbl_brightness_threshold.grid(row=2, column=8, padx=5, pady=5)
    ent_brightness_threshold = ttk.Entry(top_frame)
    ent_brightness_threshold.grid(row=2, column=9, padx=5, pady=5)
    ent_brightness_threshold.insert(0, str(brightness_threshold))
    
    # Botões de controle
    btn_select_save = ttk.Button(top_frame, text="Selecionar Diretório", command=select_save_directory)
    btn_select_save.grid(row=3, column=0, padx=5, pady=5)
    
    btn_start = ttk.Button(top_frame, text="Iniciar Captura", command=start_capture)
    btn_start.grid(row=3, column=1, padx=5, pady=5)
    
    btn_stop = ttk.Button(top_frame, text="Parar Captura", command=stop_capture)
    btn_stop.grid(row=3, column=2, padx=5, pady=5)
    
    btn_save_params = ttk.Button(top_frame, text="Salvar Parâmetros", command=save_parameters)
    btn_save_params.grid(row=3, column=3, padx=5, pady=5)
    
    # Menu de seleção de parâmetros
    lbl_preset = ttk.Label(top_frame, text="Selecione o Caso:")
    lbl_preset.grid(row=4, column=0, padx=5, pady=5)
    preset_combo = ttk.Combobox(top_frame, values=["Objetos brilhantes (-27 a 0)", "Objetos moderadamente brilhantes (0 a +6)", "Objetos fracos (+6 a +10)", "Objetos muito fracos (+10 a +30)"])
    preset_combo.grid(row=4, column=1, padx=5, pady=5)
    preset_combo.current(0)
    
    def update_parameters(event):
        apply_parameters(preset_combo.get())
    
    preset_combo.bind("<<ComboboxSelected>>", update_parameters)
    
    lbl_video = ttk.Label(root)
    lbl_video.pack(padx=10, pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()
