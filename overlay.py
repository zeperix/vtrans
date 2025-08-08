import tkinter as tk
import ctypes
from PIL import ImageFont, ImageTk, Image, ImageDraw
from screeninfo import get_monitors
import sys
def make_window_clickthrough(hwnd):
    # WS_EX_LAYERED | WS_EX_TRANSPARENT = 0x80000 | 0x20
    styles = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
    styles |= 0x80000 | 0x20
    ctypes.windll.user32.SetWindowLongW(hwnd, -20, styles)

def render_text_image(text, font_path="arial.ttf", font_size=48, text_color="red"):
    font = ImageFont.truetype(font_path, font_size)

    # Calculate bounding box for plain text
    dummy_img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Create image for plain text
    img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=text_color)

    return ImageTk.PhotoImage(img), img.width, img.height

def get_screen_resolution():
    monitor = get_monitors()[0]
    return monitor.width, monitor.height

def show_overlay(text, x, y, font_size):
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.overrideredirect(True)
    root.wm_attributes("-transparentcolor", "black")
    root.configure(bg="black")

    img, img_w, img_h = render_text_image(text, font_size=font_size)  # Pass font_size explicitly
    label = tk.Label(root, image=img, bg="black", bd=0, padx=0, pady=0, highlightthickness=0)
    label.pack()

    root.update_idletasks()

    screen_width, screen_height = get_screen_resolution()
    print(f"Kích thước màn hình thực tế: {screen_width}x{screen_height}")

    # Clamp để không ra ngoài màn hình
    x = min(max(x, 0), screen_width - img_w)
    y = min(max(y, 0), screen_height - img_h)

    root.geometry(f"{img_w}x{img_h}+{x}+{y}")

    hwnd = ctypes.windll.user32.FindWindowW(None, root.winfo_name())
    if hwnd != 0:
        make_window_clickthrough(hwnd)

    root.mainloop()

if __name__ == "__main__":
    screen_width, screen_height = get_screen_resolution()
    print(f"Kích thước màn hình của hệ thống: {screen_width}x{screen_height}")

    user_text = sys.argv[1].strip("[]")
    size = int(sys.argv[2])
    x = int(sys.argv[3])
    y = int(sys.argv[4])
    show_overlay(user_text, x, y, size)