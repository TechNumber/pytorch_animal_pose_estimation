import tkinter as tk
import tkinter.ttk as ttk
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

class LabelingAppApp:
    def __init__(self, master=None):
        # build ui
        self.window = tk.Tk() if master is None else tk.Toplevel(master)
        self.window.title("Labeling app")
        self.window.configure(height=720, width=1280)
        self.window.minsize(1000, 470)


        self.frm_picture = ttk.Labelframe(self.window)
        self.frm_picture.configure(
            height=360,
            text='Image',
            width=640)

        self.pic = ImageTk.PhotoImage(
            ImageOps.contain(
                Image.open('./unlabeled/1614520238_131-p-kot-na-belom-fone-208.jpg'),
                (640, 360)
            )
        )
        self.lbl_pic = ttk.Label(self.frm_picture, image=self.pic, cursor="tcross",)
        self.lbl_pic.bind('<Motion>', self.pic_motion)
        self.lbl_pic.pack(side="top")

        self.frm_picture.pack(padx=20, pady=20, side="left")
        self.frm_picture.pack_propagate(False)


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
        self.lbl_kp_name.configure(text='Keypoint name')
        self.lbl_kp_name.grid(column=1, pady=10, row=0)

        self.scr_frm_keypoints = ttk.Scrollbar(self.frm_keypoints)
        self.scr_frm_keypoints.configure(orient="vertical")
        self.scr_frm_keypoints.grid(
            column=3, ipady=130, padx=10, row=1, rowspan=16)

        self.ent_kp_label_def_values = [
            'head', 'upper_spine', 'left_shoulder', 'left_elbow', 'front_left_paw',
            'right_shoulder', 'right_elbow', 'front_right_paw',
            'center_spine', 'bottom_spine', 'right_knee', 'right_heel', 'rear_right_paw',
            'left_knee', 'left_heel', 'rear_left_paw'
        ]
        self.lbl_kp_id_list = []
        self.ent_kp_label_list = []
        for i in range(16):
            lbl_kp_id = ttk.Label(self.frm_keypoints, text=str(i))
            lbl_kp_id.grid(row=i + 1, column=0)
            self.lbl_kp_id_list.append(lbl_kp_id)
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

        self.btn_cancel = ttk.Button(self.frm_buttons)
        self.btn_cancel.configure(text='Cancel')
        self.btn_cancel.pack(padx=10, pady=10, side="right")

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

    def pic_motion(self, event):
        x, y = event.x, event.y
        print('{}, {}'.format(x, y))

    def start_labeling(self, event):
        print("Labeling started")

if __name__ == "__main__":
    app = LabelingAppApp()
    app.run()
