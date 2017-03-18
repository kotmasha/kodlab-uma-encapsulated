import UMA_NEW
import imp

class UMAPyUnitTest:
    def __init__(self):
        self.CPUTest = UMA_NEW.CPUTest()
        self.GPUTest = UMA_NEW.GPUTest()

    def result_report(self, targets, results):
        count = 0
        total = len(targets)
        fail_tests = []
        for i in range(total):
            if targets[i] == results[i]:
                count += 1
            else:
                fail_res = [i, targets[i], results[i]]
                fail_tests.append(fail_res)
        print str(count) + " out of " + str(total) + " tests passed"
        if count == total:
            print "All test passed"
        else:
            print "Warning: Not all test cases passed"
            for fail_res in fail_tests:
                print str(fail_res[0]) + "th test result: " + str(fail_res[2]) + ", but target: " + str(fail_res[1])

    def test_wrapper(fun):
        def fun_wrapper(self, test_name, kwargs = None):
            kwargs = imp.load_source('UnitTestData', 'Unit_Test_Data/test_data.py').UnitTestData.data[test_name] 
            print "-----------------" + test_name + "--------------------"
            print "Function Dependency: " + kwargs['function_dependency']
            if 'use_data_gen' in kwargs:
                print "Use data generation in test, from file" + kwargs['use_data_gen']
                data_gen = imp.load_source('data_gen', kwargs['use_data_gen']).data_gen()
                dict = data_gen.data_gen_fun()
                kwargs.update(dict)
            targets, results = fun(self, test_name, kwargs)
            self.result_report(targets, results)
            print "-----------------" + test_name + "--------------------"
        return fun_wrapper

    @test_wrapper
    def test_ind_host(self, test_name, test_data_ind = None):
        print "Test is done on host"
        row = test_data_ind['row']
        col = test_data_ind['col']
        targets = test_data_ind['targets']

        results = [self.CPUTest.TEST_ind_host(row[i], col[i]) for i in range(len(targets))]
        return targets, results

    @test_wrapper
    def test_ind_device(self, test_name, test_data_ind = None):
        print "Test is done on device"
        row = test_data_ind['row']
        col = test_data_ind['col']
        targets = test_data_ind['targets']

        results = [self.GPUTest.TEST_ind_device(row[i], col[i]) for i in range(len(targets))]
        return targets, results
    #ind test

    @test_wrapper
    def test_compi_host(self, test_name, test_data_ind = None):
        print "Test is done on host"
        x = test_data_ind['x']
        targets = test_data_ind['targets']

        results = [self.CPUTest.TEST_compi_host(x[i]) for i in range(len(targets))]
        return targets, results

    @test_wrapper
    def test_compi_device(self, test_name, test_data_ind = None):
        print "Test is done on device"
        x = test_data_ind['x']
        targets = test_data_ind['targets']

        results = [self.GPUTest.TEST_compi_device(x[i]) for i in range(len(targets))]
        return targets, results

    @test_wrapper
    def test_subtraction_kernel(self, test_name, test_data_ind = None):
        print "Test is done on device"
        b1 = test_data_ind['b1']
        b2 = test_data_ind['b2']
        targets = test_data_ind['targets']

        results = [self.GPUTest.TEST_subtraction_kernel(b1[i], b2[i], len(b1[i])) for i in range(len(targets))]
        return targets, results

    @test_wrapper
    def test_implies_GPU(self, test_name, test_data_ind = None):
        print "Test is done on device"
        row = test_data_ind['row']
        col = test_data_ind['col']
        weights = test_data_ind['weights']
        total = test_data_ind['total']
        threshold = test_data_ind['threshold']
        targets = test_data_ind['targets']

        results = [self.GPUTest.TEST_implies_GPU(row[i], col[i], weights, total, threshold) for i in range(len(targets))]
        return targets, results

    @test_wrapper
    def test_equivalent_GPU(self, test_name, test_data_ind = None):
        print "Test is done on device"
        row = test_data_ind['row']
        col = test_data_ind['col']
        weights = test_data_ind['weights']
        targets = test_data_ind['targets']

        results = [self.GPUTest.TEST_equivalent_GPU(row[i], col[i], weights) for i in range(len(targets))]
        return targets, results

    @test_wrapper
    def test_multiply_kernel(self, test_name, test_data_ind = None):
        print "Test is done on device"
        x = test_data_ind['x']
        dir = test_data_ind['dir']
        targets = test_data_ind['targets']

        results = [self.GPUTest.TEST_multiply_kernel(x[i], dir) for i in range(len(targets))]
        return targets, results

    @test_wrapper
    def test_check_mask(self, test_name, test_data_ind = None):
        print "Test is done on device"
        mask = test_data_ind['mask']
        targets = test_data_ind['targets']

        results = [self.GPUTest.TEST_check_mask(mask[i]) for i in range(len(targets))]
        return targets, results

    @test_wrapper
    def test_mask_kernel(self, test_name, test_data = None):
        print "Test is done on device"
        mask = test_data['mask']
        mask_amper = test_data['mask_amper']
        current = test_data['current']
        targets = test_data['targets']

        results = [self.GPUTest.TEST_mask_kernel(mask_amper, mask, current[i]) for i in range(len(targets))]
        return targets, results

    @test_wrapper
    def test_update_weights_forgetful(self, test_name, test_data = None):
        print "Test is done on device"
        q = test_data['q']
        phi = test_data['phi']
        sensor_size = test_data['sensor_size']
        iter = test_data['iter']
        activity = test_data['activity']
        signals = test_data['signals']
        
        targets = test_data['targets']
        tmp_target = [0 for i in range((2 * sensor_size + 1) * sensor_size)]
        targets.insert(0, tmp_target)

        results = [self.GPUTest.TEST_update_weights_forgetful(signals[i-1], targets[i-1], activity[i-1], phi, q, sensor_size) for i in range(1, len(targets))]
        targets.pop(0)
        return targets, results

    @test_wrapper
    def test_orient_all_kernel(self, test_name, test_data = None):
        print "Test is done on device"
        q = test_data['q']
        total = test_data['total']
        sensor_size = test_data['sensor_size']
        weights = test_data['weights']
        threshold = 0 #not used for now
        targets = test_data['targets']

        results = [self.GPUTest.TEST_orient_all(weights, q, threshold, total, sensor_size) for i in range(len(targets))]
        return targets, results

    @test_wrapper
    def test_up_GPU(self, test_name, test_data_ind = None):
        print "Test is done on device"
        signal = test_data_ind['signal']
        dir = test_data_ind['dir']
        targets = test_data_ind['targets']

        results = [self.CPUTest.TEST_up_GPU(signal[i], dir) for i in range(len(targets))]
        return targets, results

    @test_wrapper
    def test_gen_mask(self, test_name, test_data = None):
        print "Test is done on device"
        mask_amper = test_data['mask_amper']
        base_sensor_size = test_data['base_sensor_size']
        current = test_data['current']
        targets = test_data['targets']

        results = [self.CPUTest.TEST_gen_mask(mask_amper, current[i], base_sensor_size) for i in range(len(targets))]
        return targets, results

    @test_wrapper
    def test_set_signal(self, test_name, test_data = None):
        print "Test is done on device"
        signal = test_data['signal']
        targets = test_data['targets']

        results = [self.CPUTest.TEST_set_signal(signal)]
        return targets, results

    @test_wrapper
    def test_init_weight(self, test_name, test_data = None):
        print "Test is done on device"
        sensor_size = test_data['sensor_size']
        targets = test_data['targets']

        results = [self.CPUTest.TEST_init_weight(sensor_size[i]) for i in range(len(targets))]
        return targets, results

    @test_wrapper
    def test_init_direction(self, test_name, test_data = None):
        print "Test is done on device"
        sensor_size = test_data['sensor_size']
        targets = test_data['targets']

        results = [self.CPUTest.TEST_init_direction(sensor_size[i]) for i in range(len(targets))]
        return targets, results