import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QTextEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import subprocess

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口标题
        self.setWindowTitle("Image Viewer")

        # 设置窗口大小
        self.setFixedSize(800, 600)

        # 创建布局
        self.layout = QHBoxLayout() # 使用水平布局

        # 创建左侧布局
        left_layout = QVBoxLayout() # 使用垂直布局

        # 创建图片显示区域
        self.image_label = QLabel()
        self.image_label.setMinimumSize(256, 384) # 设置图片标签的最小尺寸
        left_layout.addWidget(self.image_label, alignment=Qt.AlignLeft | Qt.AlignTop) # 设置对齐方式

        # 将左侧布局添加到主布局中
        self.layout.addLayout(left_layout)

        # 创建右侧布局
        right_layout = QVBoxLayout() # 使用垂直布局

        # 创建用于显示结果的文本编辑器
        self.result_textedit = QTextEdit()
        self.result_textedit.setMinimumSize(256, 580) # 设置文本编辑器的最小尺寸
        right_layout.addWidget(self.result_textedit, alignment=Qt.AlignRight | Qt.AlignTop) # 设置对齐方式

        # 将右侧布局添加到主布局中
        self.layout.addLayout(right_layout)

        # 创建按钮布局
        button_layout = QVBoxLayout() # 使用垂直布局

        select_button = QPushButton("Select Image")
        select_button.clicked.connect(self.select_image)
        button_layout.addWidget(select_button)

        execute_button = QPushButton("Face")
        execute_button.clicked.connect(self.execute_oneface)
        button_layout.addWidget(execute_button)

        execute_button = QPushButton("Pedestrain")
        execute_button.clicked.connect(self.execute_oneimage)
        button_layout.addWidget(execute_button)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear)
        button_layout.addWidget(clear_button)

        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.exit)
        button_layout.addWidget(exit_button)

        # 设置按钮布局的对齐方式为居中
        button_layout.setAlignment(Qt.AlignCenter)

        # 将按钮布局添加到主布局中
        self.layout.addLayout(button_layout)

        # 设置布局
        self.setLayout(self.layout)

    def select_image(self):
        # 设置文件对话框的默认目录为当前目录
        file_dialog = QFileDialog(self)
        file_dialog.setDirectory(os.getcwd())
        file_path, _ = file_dialog.getOpenFileName()
        if file_path:
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(240, 580, Qt.KeepAspectRatio)  # 缩放图片到固定大小
            self.image_label.setPixmap(pixmap)
            # # 选择图片后自动执行oneimage.py
            # self.execute_oneimage()


    def execute_oneimage(self):
        image_path = self.image_label.pixmap()
        if image_path is not None and not image_path.isNull():
            image_path = image_path.toImage()
            image_path.save("temp_image.png")  # 将QImage对象保存为临时图片文件
            image_path = os.path.abspath("temp_image.png")
            result = subprocess.run(['python', 'oneimage.py', f'--image_path={image_path}'], capture_output=True,
                                    text=True)
            output = result.stdout
            self.result_textedit.setPlainText(output)

    def execute_oneface(self):
        image_path = self.image_label.pixmap()
        if image_path is not None and not image_path.isNull():
            image_path = image_path.toImage()
            image_path.save("temp_image.png")  # 将QImage对象保存为临时图片文件
            image_path = os.path.abspath("temp_image.png")
            result = subprocess.run(['python', 'oneface.py', f'--image_path={image_path}'], capture_output=True,
                                    text=True)
            output = result.stdout
            self.result_textedit.setPlainText(output)

    def clear(self):
        self.image_label.clear()
        self.result_textedit.clear()

    def exit(self):
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageViewer()
    window.show()

    sys.exit(app.exec_())
