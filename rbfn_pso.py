import numpy as np

class RBFN:
    def __init__(self, input_dim, num_rbf_units, output_dim):
        self.input_dim = input_dim  # 輸入維度: 前方、左側、右側距離
        self.num_rbf_units = num_rbf_units # 隱藏層node數
        self.output_dim = output_dim # 輸出維度: 轉向角度
        self.centers = np.random.rand(num_rbf_units, input_dim) * 10 - 5
        self.sigmas = np.random.rand(num_rbf_units) + 0.1
        self.weights = np.random.rand(num_rbf_units, output_dim) * 2 - 1

    def gaussian(self, x, c, sigma):
        return np.exp(-np.sum((x - c)**2) / (2 * sigma**2))

    def predict(self, input_data):
        activations = np.zeros(self.num_rbf_units)
        for i in range(self.num_rbf_units):
            activations[i] = self.gaussian(input_data, self.centers[i], self.sigmas[i])
        output = np.dot(activations, self.weights)
        return np.clip(output, -40, 40)

    def get_params(self):
        return np.concatenate((self.centers.flatten(), self.sigmas.flatten(), self.weights.flatten()))

    def set_params(self, params):
        num_centers_params = self.num_rbf_units * self.input_dim
        num_sigmas_params = self.num_rbf_units
        num_weights_params = self.num_rbf_units * self.output_dim
        self.centers = params[0:num_centers_params].reshape(self.num_rbf_units, self.input_dim)
        self.sigmas = params[num_centers_params:num_centers_params + num_sigmas_params]
        self.weights = params[num_centers_params + num_sigmas_params:num_centers_params + num_sigmas_params + num_weights_params].reshape(self.num_rbf_units, self.output_dim)

class PSO:
    def __init__(self, num_particles, max_iter, input_dim, num_rbf_units, output_dim, c1=2, c2=2, w=0.7):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.input_dim = input_dim
        self.num_rbf_units = num_rbf_units
        self.output_dim = output_dim
        self.c1 = c1
        self.c2 = c2
        self.w = w

        # 用來記錄每個粒子的狀態，以便在粒子群優化過程中更新
        self.particles = []
        self.pbest_positions = []
        self.pbest_scores = []
        self.gbest_position = None
        self.gbest_score = float('inf')

        # 初始化粒子群
        for _ in range(num_particles):
            rbfn = RBFN(input_dim, num_rbf_units, output_dim)
            params = rbfn.get_params()
            self.particles.append({'rbfn': rbfn, 'position': params, 'velocity': np.random.rand(len(params)) * 0.1})
            self.pbest_positions.append(params)
            self.pbest_scores.append(float('inf'))

    def fitness_function(self, rbfn, car_instance):
        car = car_instance
        car.currentX = 0
        car.currentY = 0
        car.currentPHI = 90
        collision = False
        finished = False
        steps = 0
        max_steps = 1000
        while not finished and not collision and steps < max_steps:
            distances = car.get_distances()
            steering_angle = rbfn.predict(np.array(distances))[0]
            car.set_currentTHETA(steering_angle)
            car.update_position()
            if car.check_collision():
                collision = True
            if 18 <= car.currentX <= 30 and 37 <= car.currentY:
                finished = True
            steps += 1

        fitness = 0.0
        if finished:
            fitness -= 1000
        elif collision:
            fitness += 1000
            # 如果發生碰撞，計算到終點的歐式距離
            distance_to_finish_line = np.sqrt((car.currentX - 24)**2 + (car.currentY - 38.5)**2)
            fitness += distance_to_finish_line * 10

        return max(0, fitness)

    def optimize(self, car_instance, fitness_callback=None):
        loss_history = []
        best_rbfn = None

        for iteration in range(self.max_iter):
            # 評估所有粒子並更新個體與全域最佳
            for i, particle in enumerate(self.particles):
                current_fitness = self.fitness_function(particle['rbfn'], car_instance)
                if current_fitness < self.pbest_scores[i]:
                    self.pbest_scores[i] = current_fitness
                    self.pbest_positions[i] = particle['position'].copy()
                if current_fitness < self.gbest_score:
                    self.gbest_score = current_fitness
                    self.gbest_position = particle['position'].copy()
            # 更新粒子的速度和位置
            for i, particle in enumerate(self.particles):
                r1 = np.random.rand(len(particle['position']))
                r2 = np.random.rand(len(particle['position']))

                # 自身速度
                velocity_cognitive = self.c1 * r1 * (self.pbest_positions[i] - particle['position'])
                
                # 全局速度
                velocity_social = self.c2 * r2 * (self.gbest_position - particle['position'])
                
                # 速度更新公式
                particle['velocity'] = self.w * particle['velocity'] + velocity_cognitive + velocity_social
                
                # 位置更新公式
                particle['position'] += particle['velocity']
                particle['rbfn'].set_params(particle['position'])
            loss_history.append(self.gbest_score)
            if fitness_callback:
                fitness_callback(self.gbest_score)
        
        # 返回全部迭代完成後的最佳RBFN和損失歷史
        best_rbfn = RBFN(self.input_dim, self.num_rbf_units, self.output_dim)
        best_rbfn.set_params(self.gbest_position)
        return best_rbfn, loss_history