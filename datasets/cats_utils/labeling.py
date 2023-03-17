import json
import os
import shutil
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox as mb
from PIL import ImageTk, Image, ImageOps

# WIN_WIDTH = 1280
# WIN_HEIGHT = 720
#
# window = Tk()
# window.title("Labeling app")
# window.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")
#
# frm_picture = LabelFrame(window, text='Image', width=WIN_WIDTH // 2, height=WIN_HEIGHT, cursor='dot')
# frm_picture.pack_propagate(False)
#
#
# def motion(event):
#     pass
#
#
# pic = ImageTk.PhotoImage(
#     ImageOps.contain(
#         Image.open('./unlabeled/7744bb38627f56c5048a2892ca99e6e7.jpg'),
#         (WIN_WIDTH // 2, WIN_HEIGHT)
#     )
# )
# lbl_pic = Label(frm_picture, image=pic)
# lbl_pic.bind('<Motion>', motion)
# lbl_pic.pack()
#
# frm_picture.pack(side=LEFT)
#
# frm_controls = LabelFrame(window, text='Controls', width=WIN_WIDTH // 2, height=WIN_HEIGHT)
# frm_controls.pack_propagate(False)
#
# frm_counter = Frame(frm_controls)
# # frm_counter.grid_propagate(False)
#
# lbl_kp_count = Label(frm_counter, text="Keypoints count:")
# lbl_kp_count.grid(row=0, column=0)
#
# sb_kp_count_def = IntVar()
# sb_kp_count_def.set(16)
# sb_kp_count = Spinbox(frm_counter, textvariable=sb_kp_count_def, from_=0, to=100, width=5)
# sb_kp_count.grid(row=0, column=1)
#
# frm_counter.pack()
#
# frm_keypoints = LabelFrame(frm_controls, text="Keypoint labels", width=WIN_WIDTH // 2 - 500, height=WIN_HEIGHT - 500)
# frm_keypoints.grid_propagate(False)
#
# for i in range(16):
#     lbl_kp_id = Label(frm_keypoints, text=str(i + 1))
#     lbl_kp_id.grid(row=i, column=0)
#     ent_kp_label = Entry(frm_keypoints, width=10)
#     ent_kp_label.grid(row=i, column=1)
#
# frm_keypoints.pack()
#
# frm_controls.pack(side=RIGHT)
#
# window.mainloop()

# Элементы:
# Надпись +
# Кнопка +
# Поле ввода
# Выпадающий список
# Кнопка-галочка
# Кнопка-кружок
# Поле ввода с прокруткой
# Информационное окно сообщения (предупреждения, ошибки, да/нет) +
# Поле выбора значения стрелками +
# Виджет загрузки
# Диалог открытия файла/директории +
# Панель инструментов
# Фрейм +
# Вкладки
# Паддинг элементов


class KeyPoint:
    def __init__(self, name, x=0, y=0, is_visible=True):
        self.name = name
        self.x = x,
        self.y = y,
        self.is_visible = is_visible

    def set_coord(self, x, y, is_visible):
        self.is_visible = is_visible
        self.x = x
        self.y = y

    def draw_kp(self, canvas):
        canvas.create_oval(
            self.x - 5, self.y - 5, self.x + 5, self.y + 5,
            tags='keypoint',
            fill='light green' if self.is_visible else 'orange',
            outline='black',
            width=1)


class Picture:

    def __init__(self, path):
        self.path = path
        self.name = self.path[self.path.rfind('/') + 1:]
        self.canvas_pic = Image.open(self.path)
        self.orig_height = self.canvas_pic.height
        self.orig_width = self.canvas_pic.width
        self.canvas_pic = ImageTk.PhotoImage(
            ImageOps.contain(
                self.canvas_pic,
                (1280, 720)
            )
        )

    def draw_picture(self, canvas):
        canvas.configure(width=self.canvas_pic.width(), height=self.canvas_pic.height())
        canvas.create_image(0, 0, anchor='nw', image=self.canvas_pic, tags='image')


class LabelingAppApp:
    def __init__(self, master=None):
        # build ui
        self.window = tk.Tk() if master is None else tk.Toplevel(master)
        self.window.title("Labeling app")
        self.window.configure(height=830, width=1620)
        self.window.minsize(1620, 830)


        self.frm_labeling = ttk.Labelframe(self.window)
        self.frm_labeling.configure(
            height=780,
            text='Image',
            width=1280)

        self.frm_picture = ttk.Frame(self.frm_labeling)
        self.frm_picture.configure(
            height=720,
            width=1280)

        # self.pic = ImageTk.PhotoImage(
        #     ImageOps.contain(
        #         Image.open('./unlabeled/1614520238_131-p-kot-na-belom-fone-208.jpg'),
        #         (1280, 720)
        #     )
        # )
        # self.lbl_pic = ttk.Label(self.frm_picture, image=self.pic, cursor="tcross")
        # self.lbl_pic.bind('<Motion>', self.pic_motion)
        # self.lbl_pic.grid(row=0, column=0)

        self.cnv_labeling = tk.Canvas(self.frm_picture, height=720, width=1280,
                                      cursor='tcross', state='disabled')
        self.cnv_labeling.bind('<Motion>', self.pic_motion, add='+')
        self.cnv_labeling.bind('<Button-1>', self.pic_lmb_pressed, add='+')
        self.cnv_labeling.bind('<Button-3>', self.pic_rmb_pressed, add='+')
        self.cnv_labeling.grid(row=0, column=0)

        # self.img_cnv_labeling = self.cnv_labeling.create_image(0, 0, anchor='nw', image=self.pic, tags='image')

        self.frm_picture.pack(side='top')
        # self.frm_picture.grid_propagate(False)

        self.lbl_cursor_coord = ttk.Label(self.frm_labeling, text="In standby")
        self.lbl_cursor_coord.pack(side='bottom', anchor='e', pady=10, padx=10)

        self.frm_labeling.pack(padx=20, pady=20, side="left")
        self.frm_labeling.pack_propagate(False)


        self.frm_controls = ttk.Frame(self.window)
        self.frm_controls.configure(height=200, width=200)

        self.frm_counter = ttk.Frame(self.frm_controls)
        self.frm_counter.configure(height=200, width=200)

        self.lbl_kp_count = ttk.Label(self.frm_counter)
        self.lbl_kp_count.configure(text='Keypoint count:')
        self.lbl_kp_count.pack(side="left")

        self.sb_kp_count = ttk.Spinbox(self.frm_counter)
        self.sb_kp_count_def = tk.IntVar()
        self.sb_kp_count_def.set(16)
        self.sb_kp_count.configure(
            from_=1,
            textvariable=self.sb_kp_count_def,
            to=100,
            width=10)
        self.sb_kp_count.pack(side="right")

        self.frm_counter.pack(anchor="w", fill="x", side="top")


        self.frm_keypoints = ttk.Frame(self.frm_controls)
        self.frm_keypoints.configure(height=340, width=265)

        self.lbl_kp_num = ttk.Label(self.frm_keypoints)
        self.lbl_kp_num.configure(text='Keypoint num')
        self.lbl_kp_num.grid(column=0, pady=10, row=0)

        self.lbl_kp_name = ttk.Label(self.frm_keypoints)
        self.lbl_kp_name.configure(text='Keypoint json key')
        self.lbl_kp_name.grid(column=1, pady=10, row=0)

        self.scr_frm_keypoints = ttk.Scrollbar(self.frm_keypoints)
        self.scr_frm_keypoints.configure(orient="vertical")
        self.scr_frm_keypoints.grid(
            column=3, ipady=130, padx=10, row=1, rowspan=16)

        self.ent_kp_label_def_values = [
            'head', 'upper_spine', 'left_shoulder', 'left_elbow', 'front_left_paw',
            'right_shoulder', 'right_elbow', 'front_right_paw',
            'center_spine', 'bottom_spine', 'left_knee', 'left_heel', 'rear_left_paw',
            'right_knee', 'right_heel', 'rear_right_paw'
        ]
        self.ent_kp_label_list = []
        self.kp_list = []
        self.kp_id = 0
        self.pic_id = 0
        self.pic_labeling = None
        for i in range(16):
            lbl_kp_id = ttk.Label(self.frm_keypoints, text=str(i))
            lbl_kp_id.grid(row=i + 1, column=0)
            ent_kp_label = ttk.Entry(self.frm_keypoints)
            ent_kp_label.insert(0, self.ent_kp_label_def_values[i])
            ent_kp_label.grid(row=i + 1, column=1, pady=2)
            self.ent_kp_label_list.append(ent_kp_label)

        self.frm_keypoints.pack(pady=15, side="top")
        self.frm_keypoints.grid_propagate(False)


        self.frm_buttons = ttk.Frame(self.frm_controls)
        self.frm_buttons.configure(height=200, width=200)

        self.btn_start = ttk.Button(self.frm_buttons)
        self.btn_start.configure(text='Start')
        self.btn_start.pack(padx=10, pady=10, side="left")
        self.btn_start.bind("<Button>", self.start_labeling, add="")
        self.labeling_started = False

        self.btn_cancel = ttk.Button(self.frm_buttons)
        self.btn_cancel.configure(text='Cancel')
        self.btn_cancel.pack(padx=10, pady=10, side="right")
        self.btn_cancel.bind("<Button>", self.cancel_pressed, add="")
        self.labeling_cancelled = False

        self.btn_choose = ttk.Button(self.frm_buttons)
        self.btn_choose.configure(text='Choose...')
        self.btn_choose.pack(side="left")
        self.btn_choose.bind("<Button>", self.choose_path, add="")

        self.frm_buttons.pack(anchor="e", side="bottom")

        self.frm_controls.pack(
            expand=True,
            fill="y",
            padx=20,
            pady=20,
            side="right")


        self.window.pack_propagate(False)

        # Main widget
        self.mainwindow = self.window

    def run(self):
        self.mainwindow.mainloop()


    def choose_path(self, event=None):
        pass

    def update_coordinates(self, x, y):
        if self.labeling_started:
            x = round(x / self.cnv_labeling.winfo_width(), 3)
            y = round(y / self.cnv_labeling.winfo_height(), 3)
            message = f'Now labeling: {self.kp_list[self.kp_id].name} ({self.kp_id})'
            if 0 <= x <= 1 and 0 <= y <= 1:
                self.lbl_cursor_coord.configure(
                    text=message + f' [{x}, {y}]')
            else:
                self.lbl_cursor_coord.configure(text=message + ' [out of bounds]')

    def pic_motion(self, event):
        self.update_coordinates(event.x, event.y)

    def get_unlabeled_pic(self):
        if len(os.listdir('./train/unlabeled')) > 0:
            self.kp_id = 0
            self.pic_labeling = Picture('./train/unlabeled/' + os.listdir('./train/unlabeled')[0])
            self.pic_labeling.draw_picture(self.cnv_labeling)
        else:
            self.cancel_labeling()
            mb.showinfo('Labeling done', 'All images from chosen directory were labeled.')


    def add_keypoint(self, x, y, is_visible):
        if self.labeling_started:
            self.update_coordinates(x, y)
            self.kp_list[self.kp_id].set_coord(x, y, is_visible)
            self.kp_list[self.kp_id].draw_kp(self.cnv_labeling)
            self.kp_id += 1
            if self.kp_id > 15:
                if os.path.isfile('./train/keypoints_annotations.json'):
                    with open('./train/keypoints_annotations.json') as f:
                        json_data = json.load(f)
                    self.pic_id = max(map(int, json_data['images'].keys())) + 1
                else:
                    json_data = {
                        'info': "",
                        'images': {},
                        'keypoints': {}
                    }
                    self.pic_id = 0
                json_data['images'][self.pic_id] = {
                    "file_name": self.pic_labeling.name,
                    "width": self.pic_labeling.orig_width,
                    "height": self.pic_labeling.orig_height
                }
                json_data['keypoints'][self.pic_id] = dict()
                for kp in self.kp_list:
                    json_data['keypoints'][self.pic_id][kp.name] = [
                        kp.x / self.cnv_labeling.winfo_width(),
                        kp.y / self.cnv_labeling.winfo_height(),
                        int(kp.is_visible)
                    ]
                with open('./train/keypoints_annotations.json', 'w+') as f:
                    json.dump(json_data, f, indent=4)
                if not os.path.isdir('./train/labeled'):
                    os.mkdir('./train/labeled')
                shutil.move(self.pic_labeling.path, './train/labeled')
                self.get_unlabeled_pic()
            else:
                self.update_coordinates(x, y)

    def pic_lmb_pressed(self, event):
        x, y = event.x, event.y
        self.add_keypoint(x, y, is_visible=True)

    def pic_rmb_pressed(self, event):
        x, y = event.x, event.y
        self.add_keypoint(x, y, is_visible=False)


    def start_labeling(self, event):
        if len(os.listdir('./train/unlabeled')) > 0:
            print("Labeling started")
            self.btn_start.configure(state='disabled')
            self.btn_choose.configure(state='disabled')
            self.sb_kp_count.configure(state='readonly')
            self.kp_list = []
            for ent in self.ent_kp_label_list:
                ent.configure(state='readonly')
                self.kp_list.append(KeyPoint(ent.get().lower().replace(' ', '_')))
            self.labeling_started = True
            self.get_unlabeled_pic()
            # self.pic = ImageTk.PhotoImage(
            #     ImageOps.contain(
            #         Image.open(self.pic_path),
            #         (1280, 720)
            #     )
            # )
            # self.cnv_labeling.configure(width = self.pic.width(), height = self.pic.height())
            # self.cnv_labeling.create_image(0, 0, anchor='nw', image=self.pic, tags='image')
        else:
            mb.showerror('No files', 'There are no files to label in chosen directory.')

    def cancel_labeling(self):
        if self.labeling_started:
            self.labeling_started = False
            print("Labeling ended")
            self.cnv_labeling.delete('keypoint')
            self.cnv_labeling.delete('image')
            self.pic_labeling = None
            self.btn_start.configure(state='normal')
            self.btn_choose.configure(state='normal')
            self.sb_kp_count.configure(state='normal')
            for ent in self.ent_kp_label_list:
                ent.configure(state='normal')
            self.lbl_cursor_coord.configure(text="In standby")

    def cancel_pressed(self, event):
        self.cancel_labeling()


if __name__ == "__main__":
    app = LabelingAppApp()
    app.run()
