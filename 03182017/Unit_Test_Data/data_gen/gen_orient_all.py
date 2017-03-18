class data_gen:
    # orient all
    def data_gen_fun(self):
        import random as rand
        # change those variable when a different test needed
        q = 0.75
        sensor_size = 100
        total = 1
        threshold = 1
        # change those variable when a different test needed
        print 'q:', q, 'sensor_size:', sensor_size, 'total:', total 

        def ind(row, col):
            if row >= col:
                return row * (row + 1) / 2 + col
            else: 
                return col * (col + 1) / 2 + row

        def compi(x):
            if x % 2 == 0:
                return x + 1
            else:
                return x - 1

        def equivalent(row, col, weights):
            rc = weights[ind(row, col)]
            r_c = weights[ind(compi(row), col)]
            rc_ = weights[ind(row, compi(col))]
            r_c_ = weights[ind(compi(row), compi(col))]
            return rc_ == 0 and r_c == 0 and rc > 0 and r_c_ > 0

        def implies(row, col, weights, total, threshold):
            rc = weights[ind(row, col)]
            r_c = weights[ind(compi(row), col)]
            rc_ = weights[ind(row, compi(col))]
            r_c_ = weights[ind(compi(row), compi(col))]
            epsilon = total * threshold
            m = min(epsilon,min(rc, min(r_c, r_c_)))
            return rc_ < m

        weights = []
        tmp_weights = []
        targets = []
        tmp_targets = []

        def gen_weights():
            for i in range(sensor_size):
                tmp_weights.append([])
                tmp_weights.append([])
                for j in range(i + 1):
                    unit = 100.0
                    remain = unit
                    kvpair = {}
                    if i == j: # diagonal condition
                        index = [0, 1]
                        for k in range(1):
                            key = rand.randint(0, len(index) - 1)
                            value = rand.randint(0, remain)
                            kvpair[index[key]] = value
                            index.remove(index[key])
                            remain -= value
                        key = 0
                        value = remain
                        kvpair[index[key]] = value
                        index.remove(index[key])

                        tmp_weights[2 * i].append(kvpair[0] / unit)
                        tmp_weights[2 * i + 1].append(0)
                        tmp_weights[2 * i + 1].append(kvpair[1] / unit)
                    else:
                        index = [0, 1, 2, 3]
                        for k in range(3):
                            key = rand.randint(0, len(index) - 1)
                            value = rand.randint(0, remain)
                            kvpair[index[key]] = value
                            index.remove(index[key])
                            remain -= value
                        key = 0
                        value = remain
                        kvpair[index[key]] = value
                        index.remove(index[key])

                        tmp_weights[2 * i].append(kvpair[0] / unit)
                        tmp_weights[2 * i].append(kvpair[1] / unit)
                        tmp_weights[2 * i + 1].append(kvpair[2] / unit)
                        tmp_weights[2 * i + 1].append(kvpair[3] / unit)
                weights.extend(tmp_weights[2 * i])
                weights.extend(tmp_weights[2 * i + 1])
                        
    
        def gen_targets():
            for i in range(sensor_size):
                tmp_targets.append([])
                tmp_targets.append([])
                for j in range(i + 1):
                    if 2 * i >= 2 * j:
                        tmp_targets[2 * i].append(implies(2 * i, 2 * j, weights, total, threshold) or equivalent(2 * i, 2 * j, weights))
                    else:
                        tmp_targets[2 * i].append(implies(compi(2 * j), compi(2 * i), weights, total, threshold) or equivalent(compi(2 * j), compi(2 * i), weights))
                    if 2 * i >= 2 * j + 1:
                        tmp_targets[2 * i].append(implies(2 * i, 2 * j + 1, weights, total, threshold) or equivalent(2 * i, 2 * j + 1, weights))
                    if 2 * i + 1 >= 2 * j:
                        tmp_targets[2 * i + 1].append(implies(2 * i + 1, 2 * j, weights, total, threshold) or equivalent(2 * i + 1, 2 * j, weights))
                    else:
                        tmp_targets[2 * i + 1].append(implies(compi(2 * j), compi(2 * i + 1), weights, total, threshold) or equivalent(compi(2 * j), compi(2 * i + 1), weights))
                    if 2 * i + 1 >= 2 * j + 1:
                        tmp_targets[2 * i + 1].append(implies(2 * i + 1, 2 * j + 1, weights, total, threshold) or equivalent(2 * i + 1, 2 * j + 1, weights))
                    else:
                        tmp_targets[2 * i + 1].append(implies(compi(2 * j + 1), compi(2 * i + 1), weights, total, threshold) or equivalent(compi(2 * j + 1), compi(2 * i + 1), weights))
                targets.extend(tmp_targets[2 * i])
                targets.extend(tmp_targets[2 * i + 1])

        gen_weights()
        gen_targets()
        #print tmp_weights
        #exit()
        targets = [tuple(targets)]

        return {'func_name': 'orient_all', 'sensor_size': sensor_size, 'q': q, 'total': total, 'threshold': threshold,
		'weights': weights, 'targets': targets}