class data_gen:
    # discounted update weight
    def data_gen_fun(self):
        import random as rand
        # change those variable when a different test needed
        phi = 1
        q = 0.75
        sensor_size = 100
        iter = 10
        # change those variable when a different test needed
        print 'q:', q, 'phi:', phi, 'sensor_size:', sensor_size, 'iter:', iter 

        signals = []
        activity = []
        tmp_activity = []
        targets = []
        def gen_signal():
            for i in range(iter):
                tmp_signal = []
                for j in range(sensor_size):
                    if rand.randint(0, 1) is 0:
                        tmp_signal.extend([False, True])
                    else:
                        tmp_signal.extend([True, False])
                signals.append(tmp_signal)
    
        def gen_activity():
            for i in range(iter):
                if rand.randint(0, 1) is 0:
                    tmp_activity.append(0)
                    activity.append(False)
                else:
                    tmp_activity.append(1)
                    activity.append(True)
    
        def ind(row, col):
            return row * (row + 1) / 2 + col
    
        def gen_targets():
            for i in range(iter):
                tmp_target = []
                for j in range(2 * sensor_size):
                    for k in range(2 * sensor_size):
                        if j >= k:
                            last_t = targets[i - 1][ind(j, k)] if i > 0 else 0
                            t = last_t * q + (1 - q) * signals[i][j] * signals[i][k] * tmp_activity[i]
                            tmp_target.append(t)
                targets.append(tuple(tmp_target))
    
        gen_signal()
        gen_activity()
        gen_targets()

        return {'func_name': 'update_weights_discounted', 'iter': iter, 'sensor_size': sensor_size, 'q': q,
		 'phi': phi, 'signals': signals, 'activity': activity, 'targets': targets}