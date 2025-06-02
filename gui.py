import os
import sys
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from car import Car
from rbfn_pso import RBFN, PSO
import numpy as np
import threading


class gui():
    def __init__(self, app_name, app_width, app_height):
        self.track = []
        self.ax = None
        self.car_artists = []
        self.path_artists = []
        self.position_artists = []
        self.model = None

        # container initialization
        self.container = tk.Tk()
        self.container.config(bg='white', padx=10, pady=10)
        self.container.maxsize(app_width, app_height)
        self.container.title(app_name)
        self.container.geometry(f"{app_width}x{app_height}")

        # components initialization
        self.setting_frame = tk.Frame(self.container, width=500, height=480, bg='white')
        self.graph_frame = tk.Frame(self.container, width=1300, height=450, bg='white')
        self.track_graph = FigureCanvasTkAgg(master=self.graph_frame)
        self.track_graph.get_tk_widget().config(width=430, height=400)
        self.rbfn_graph = FigureCanvasTkAgg(master=self.container)
        self.rbfn_graph.get_tk_widget().config(width=800, height=750)

        # components placing
        self.setting_frame.place(x=5, y=5)
        self.graph_frame.place(x=5, y=120)
        self.track_graph.get_tk_widget().place(x=0, y=50)
        self.rbfn_graph.get_tk_widget().place(x=500, y=10)

        # Buttons
        self.train_btn = tk.Button(self.setting_frame, text="Train RBFN", command=self.train_rbfn_and_save, height=2, width=15, highlightbackground='white')
        self.sim_btn = tk.Button(self.setting_frame, text="Test Simulation", command=self.run_rbfn_test, height=2, width=20, highlightbackground='white')
        self.success_btn = tk.Button(self.setting_frame, text="Simulate with Success Param", command=self.run_success, height=2, width=22, highlightbackground='white')

        self.train_btn.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.sim_btn.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.success_btn.grid(row=1, column=0, padx=5, pady=5, sticky='w')

        self.draw_car_track()

    def open(self):
        self.container.mainloop()

    def draw_car_track(self):
        if hasattr(sys, '_MEIPASS'):
            trackFile = os.path.join(sys._MEIPASS, "track.txt")
        else:
            trackFile = os.path.join(os.path.abspath("."), "track.txt")
        with open(trackFile, 'r') as f:
            lines = f.readlines()
        
        # “起點座標”及“起點與水平線之的夾角”
        start_x, start_y, phi = [float(coord) for coord in lines[0].strip().split(',')]
        
        # “終點區域左上角座標”及“終點區域右下角座標”
        finish_top_left = [float(coord) for coord in lines[1].strip().split(',')]
        finish_bottom_right = [float(coord) for coord in lines[2].strip().split(',')]
        
        # “賽道邊界”
        boundaries = [[float(coord) for coord in line.strip().split(',')] for line in lines[3:]]
        
        # Extract x and y coordinates from boundaries
        boundary_x, boundary_y = zip(*boundaries)
        self.track = boundaries
        self.car = Car(start_x, start_y, phi, boundaries)
        
        
        self.figure = plt.Figure(figsize=(15, 15), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(-20, 40)
        self.ax.set_ylim(-5, 55)
        self.ax.set_aspect('equal') # 讓xy軸的單位長度相等
        self.ax.set_title("Track")
        
        # Plot track boundary
        self.ax.plot(boundary_x, boundary_y, 'k-', linewidth=2)
        
        # Draw start line
        self.ax.plot([-6, 6], [0, 0], 'b-', linewidth=2, label="Start Line")
        
        # Draw finishing line
        self.ax.plot([18, 30], [37, 37], 'k-', linewidth=2, label="Finishing Line")
        self.ax.plot([18, 30], [40, 40], 'k-', linewidth=2)
        
        # Drawing the racecar-contest-like finishing line
        num_squares = 10
        square_width = (finish_bottom_right[0] - finish_top_left[0]) / num_squares
        square_height = (finish_bottom_right[1] - finish_top_left[1]) / 2
        for row in range(2):
            for i in range(num_squares):
                color = 'black' if (i + row) % 2 == 0 else 'white'
                self.ax.add_patch(plt.Rectangle((finish_top_left[0] + i * square_width, finish_top_left[1] + row * square_height),
                        square_width, square_height,
                        edgecolor=color, facecolor=color))
        
        # Draw starting position and direction arrow
        car, text, path = self.car.draw_car(self.ax)
        self.position_artists.append(car)
        self.position_artists.append(text)
        self.path_artists.append(path)
        self.ax.plot(start_x, start_y, 'ro', label="Start Position")
        self.ax.scatter([], [], color='darkgrey', label='Path')
        self.ax.scatter([], [], marker=r'$\rightarrow$', label=f"Front Sensor", color='red', s=100)
        self.ax.scatter([], [], marker=r'$\rightarrow$', label=f"Right Sensor", color='blue', s=100)
        self.ax.scatter([], [], marker=r'$\rightarrow$', label=f"Left Sensor", color='green', s=100)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend()
        plt.grid(True)

        # Show plot
        self.track_graph.figure = self.figure
        self.track_graph.draw()

    def check_finish(self):
        return 18 <= self.car.currentX <= 30 and 37 <= self.car.currentY

    def clear_all_artists(self):
        self.clear_car_artists()
        self.clear_position_artists()
        self.clear_path_artists()
    
    def clear_car_artists(self):
        if len(self.car_artists) > 0:
            for artist in self.car_artists:
                if artist is not None:
                    artist.remove()
            self.car_artists = []

    def clear_position_artists(self):
        if len(self.position_artists) > 0:
            for artist in self.position_artists:
                if artist is not None:
                    artist.remove()
            self.position_artists = []

    def clear_path_artists(self):
        if len(self.path_artists) > 0:
            for artist in self.path_artists:
                if artist is not None:
                    artist.remove()
            self.path_artists = []
    
    def clear_loss_curve(self):
        if self.rbfn_graph.figure:
            self.rbfn_graph.figure.clf()
            self.rbfn_graph.draw_idle()
            
    def train_rbfn_and_save(self):
        self.clear_all_artists()
        self.clear_loss_curve()
        print('===== Start Training ====== ')
        def _train_loop():
            try:
                self.disable_buttons()
                num_particles = 20 # 粒子數量，代表候選的 RBFN 模型在同時進行訓練和競爭的數量
                max_iterations = 300 
                input_dim = 3
                num_rbf_units = 10 # 隱藏層 RBF 單元數量
                output_dim = 1
                pso = PSO(num_particles, max_iterations, input_dim, num_rbf_units, output_dim)
                loss_history = []
                def fitness_callback(loss):
                    loss_history.append(loss)
                    print(f"Iteration {len(loss_history)}: Loss = {loss}")  # 列印訓練資訊
                self.model, loss_history = pso.optimize(self.car, fitness_callback)
                np.save("train_param.npy", self.model.get_params())
                np.save("loss_history.npy", np.array(loss_history))
                messagebox.showinfo("Training", "Training finished and parameters saved.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                self.enable_buttons()
                self.plot_loss_curve()
        self.training_thread = threading.Thread(target=_train_loop)
        self.training_thread.start()

    def plot_loss_curve(self):
        try:
            loss_history = np.load("loss_history.npy")
            fig = plt.Figure(figsize=(8, 5), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(loss_history, label="Loss")
            ax.set_title("Training Loss Curve")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True)
            self.rbfn_graph.figure = fig
            self.rbfn_graph.draw()
            self.rbfn_graph.get_tk_widget().place(x=500, y=10)

        except Exception as e:
            messagebox.showerror("Error", f"Cannot plot loss: {e}")

    def run_rbfn_test(self):
        try:
            params = np.load("train_param.npy")
            self.model = RBFN(3, 10, 1)
            self.model.set_params(params)
            self.test_simulation(save_success=True)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load parameters: {e}")

    def test_simulation(self, save_success=False):

        self.clear_all_artists()
        self.car = Car(0, 0, 90, self.track)
        self.disable_buttons()

        try:
            done = False
            while not done:
                distances = self.car.get_distances()

                theta = self.model.predict(np.array(distances))[0]
                self.car.set_currentTHETA(theta)
                self.car.update_position()
                distances = self.car.get_distances()

                # 清除之前的位置資訊和車子及箭頭
                self.clear_car_artists()
                self.clear_position_artists()
                # 畫感測器箭頭
                self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Front', self.car.currentX, self.car.currentY, self.car.currentPHI, distances[0]))
                self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Left', self.car.currentX, self.car.currentY, self.car.currentPHI + 45, distances[1]))
                self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Right', self.car.currentX, self.car.currentY, self.car.currentPHI - 45, distances[2]))
                car, text, path = self.car.draw_car(self.ax)
                self.position_artists.append(car)
                self.position_artists.append(text)
                self.path_artists.append(path)
                
                self.track_graph.get_tk_widget().update()
                self.track_graph.draw()

                # 檢查終點或碰撞
                if self.check_finish():
                    if save_success:
                        np.save("success_param.npy", self.model.get_params())
                        messagebox.showinfo("Success", "Car reached finish! Parameters saved as success_param.npy")
                    else:
                        messagebox.showinfo("Success", "Car reached finish!")
                    done = True
                elif self.car.check_collision():
                    messagebox.showinfo("Collision", "Car hit the wall!")
                    done = True
            if not done:
                messagebox.showinfo("Simulation End", "Simulation ended without reaching finish.")
        except Exception as e:
            print(f"Error in run_: {e}")
        finally:
            self.enable_buttons()

    def run_success(self):
        try:
            params = np.load("success_param.npy")
            self.model = RBFN(3, 10, 1)
            self.model.set_params(params)
            self.test_simulation(save_success=False)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load success parameters: {e}")

    def disable_buttons(self):
        self.train_btn.config(state='disabled')
        self.sim_btn.config(state='disabled')
        self.success_btn.config(state='disabled')

    def enable_buttons(self):
        self.train_btn.config(state='normal')
        self.sim_btn.config(state='normal')
        self.success_btn.config(state='normal')