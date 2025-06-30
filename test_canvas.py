import tkinter as tk

def paint(event):
    # Very simple drawing test
    canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, fill='black')
    print(f"Drew at {event.x}, {event.y}")

root = tk.Tk()
root.title("Canvas Test")

canvas = tk.Canvas(root, width=300, height=300, bg='white')
canvas.pack()

canvas.bind('<B1-Motion>', paint)
canvas.bind('<ButtonPress-1>', paint)

root.mainloop()
