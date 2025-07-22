from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Color, Ellipse, Line
from kivy.graphics.texture import Texture
from kivy.core.text import Label as CoreLabel
from kivy.utils import platform
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.base import EventLoop
from plyer import camera

import cv2
import numpy as np
import math
import os
import sys
# ----------------Splash_Screen----------------
class SplashScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()
        self.logo = Image(source='assets/Splash.png', allow_stretch=True, keep_ratio=True,
                          size_hint=(1, 1), pos_hint={"center_x": 0.5, "center_y": 0.5})
        layout.add_widget(self.logo)
        self.add_widget(layout)

        # after 4 seconds 
        Clock.schedule_once(self.switch_to_main, 4)

    def switch_to_main(self, dt):
        self.manager.current = 'menu'
# ---------------- Main Menu ------------------
class MainMenu(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()

        with layout.canvas.before:
            Color(0, 1, 0, 1)  # سبز
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
        layout.bind(size=self._update_bg, pos=self._update_bg)


        btn_start = Button(text="Start", size_hint=(0.4, 0.15), pos_hint={"center_x": 0.5, "center_y": 0.7})
        btn_tutorial = Button(text="Tutorial", size_hint=(0.4, 0.15), pos_hint={"center_x": 0.5, "center_y": 0.5})
        btn_about = Button(text="About Us", size_hint=(0.4, 0.15), pos_hint={"center_x": 0.5, "center_y": 0.3})
        btn_exit = Button(text="Exit", size_hint=(0.4, 0.15), pos_hint={"center_x": 0.5, "center_y": 0.1})

        btn_start.bind(on_release=self.start_app)
        btn_tutorial.bind(on_release=self.show_tutorial)
        btn_about.bind(on_release=self.show_about)
        btn_exit.bind(on_release=self.exit_app)

        layout.add_widget(btn_start)
        layout.add_widget(btn_tutorial)
        layout.add_widget(btn_about)
        layout.add_widget(btn_exit)

        self.add_widget(layout)

    def _update_bg(self, *args):
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos
    def start_app(self, instance):
        self.manager.current = 'pose'

    def show_tutorial(self, instance):
        self.manager.current = 'tutorial'

    def show_about(self, instance):
        self.manager.current = 'about'

    def exit_app(self, instance):
        App.get_running_app().stop()
        sys.exit()


from kivy.uix.image import Image

class AboutScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()

        with layout.canvas.before:
            Color(0, 1, 0, 1)  # 
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
        layout.bind(size=self._update_bg, pos=self._update_bg)

        about_image = Image(source='assets/about.png', allow_stretch=True, keep_ratio=True,
                            size_hint=(0.9, 0.9), pos_hint={"center_x": 0.5, "center_y": 0.5})

        layout.add_widget(about_image)
        self.add_widget(layout)

    def _update_bg(self, *args):
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos


class TutorialScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()
        label = Label(text="To analyze joint flexibility, drag the points.", font_size=18,
                      size_hint=(0.9, 0.3), pos_hint={"center_x": 0.5, "center_y": 0.5})
        layout.add_widget(label)
        self.add_widget(layout)


# ---------------- Pose Screen with Camera ------------------
class PoseScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = FloatLayout()
        self.capture_btn = Button(text="Capture from Camera", size_hint=(0.4, 0.1),
                                  pos_hint={"center_x": 0.5, "y": 0.85})
        self.capture_btn.bind(on_release=self.capture_photo)
        self.layout.add_widget(self.capture_btn)
        self.joint_canvas = None
        self.add_widget(self.layout)

    def capture_photo(self, instance):
        filepath = "/sdcard/DCIM/captured_pose.jpg"
        camera.take_picture(filename=filepath, on_complete=self.on_photo_captured)

    def on_photo_captured(self, filepath):
        if filepath and os.path.exists(filepath):
            if self.joint_canvas:
                self.layout.remove_widget(self.joint_canvas)
            self.joint_canvas = JointDragger(filepath)
            self.layout.add_widget(self.joint_canvas)


# ---------------- Joint Dragger ------------------
class JointDragger(Widget):
    def __init__(self, image_path, **kwargs):
        super().__init__(**kwargs)
        self.joints = {
            "main_hip": [350, 500],
            "left_ankle": [200, 300],
            "right_ankle": [500, 300]
        }
        self.selected = None
        self.radius = 20

        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError("❌ Image not found!")
        self.img = cv2.resize(self.img, (640, 480))
        self.texture = self.cv2_to_texture(self.img)
        self.reference_length_pixels = np.linalg.norm(np.array([100, 100]) - np.array([100, 150]))

        with self.canvas:
            self.bg = Rectangle(texture=self.texture, pos=self.pos, size=self.size)
            self.draw_joints()

        save_btn = Button(text="Save", size_hint=(0.15, 0.1), pos_hint={"right": 0.98, "y": 0.01})

        save_btn.bind(on_release=self.save_result)
        self.add_widget(save_btn)

        self.bind(pos=self.update_canvas, size=self.update_canvas)

    def save_result(self, instance):
        path = "/sdcard/DCIM/flexibility_result.png"
        self.export_to_png(path)

    def cv2_to_texture(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 0)
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='rgb')
        texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        return texture

    def update_canvas(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size
        self.redraw()

    def on_touch_down(self, touch):
        for name, pos in self.joints.items():
            if (abs(touch.x - pos[0]) < self.radius) and (abs(touch.y - pos[1]) < self.radius):
                self.selected = name
                break

    def on_touch_move(self, touch):
        if self.selected:
            self.joints[self.selected] = [touch.x, touch.y]
            self.redraw()

    def on_touch_up(self, touch):
        self.selected = None

    def redraw(self):
        self.canvas.clear()
        with self.canvas:
            self.bg = Rectangle(texture=self.texture, pos=self.pos, size=self.size)
            self.draw_joints()

    def draw_joints(self):
        Color(1, 0, 0)
        for name, pos in self.joints.items():
            Ellipse(pos=(pos[0] - self.radius / 2, pos[1] - self.radius / 2), size=(self.radius, self.radius))
            label = CoreLabel(text=name, font_size=20)
            label.refresh()
            text_texture = label.texture
            Rectangle(texture=text_texture, size=text_texture.size,
                      pos=(pos[0] + self.radius / 2, pos[1] - self.radius / 2))

        A = np.array(self.joints["left_ankle"], dtype=np.float32)
        B = np.array(self.joints["right_ankle"], dtype=np.float32)
        P = np.array(self.joints["main_hip"], dtype=np.float32)

        AB = B - A
        AP = P - A
        AB_norm = AB / np.linalg.norm(AB)
        proj_len = np.dot(AP, AB_norm)
        foot_proj = A + proj_len * AB_norm

        def distance(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))

        a = distance(self.joints["left_ankle"], foot_proj) / self.reference_length_pixels
        b = distance(self.joints["right_ankle"], foot_proj) / self.reference_length_pixels
        h = distance(self.joints["main_hip"], foot_proj) / self.reference_length_pixels

        degree = 360 - math.degrees(math.atan(a / h) + math.atan(b / h))
        mid_y = (self.joints["left_ankle"][1] + self.joints["right_ankle"][1]) / 2
        if self.joints["main_hip"][1] > mid_y:
            degree = 360 - degree

        Color(0, 0, 1)
        label = CoreLabel(text=f"D = {degree:.2f}°", font_size=30)
        label.refresh()
        Rectangle(texture=label.texture, size=label.texture.size, pos=(10, 10))


# ---------------- Main App ------------------
class MainApp(App):
    def build(self):
        self.sm = ScreenManager()
        self.sm.add_widget(SplashScreen(name='splash'))
        self.sm.add_widget(MainMenu(name='menu'))
        self.sm.add_widget(PoseScreen(name='pose'))
        self.sm.add_widget(TutorialScreen(name='tutorial'))
        self.sm.add_widget(AboutScreen(name='about'))

        if platform == 'android':
            EventLoop.window.bind(on_keyboard=self.hook_keyboard)

        self.sm.current = 'splash'

        return self.sm

    def hook_keyboard(self, window, key, *args):
        if key == 27:  # Back button
            if self.sm.current != 'menu':
                self.sm.current = 'menu'
                return True
        return False


if __name__ == '__main__':
    MainApp().run()
