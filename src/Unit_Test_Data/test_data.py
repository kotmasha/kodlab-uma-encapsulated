# use the wrapper to wrapper the data in the specific namespace

class UnitTestData:
    data = {
        # index test data
        '[Unit] Index Test':
        {#test_data_ind
            'function_dependency': 'None',
            'row' : [0, 3, 5, 4, 5],
            'col' : [0, 3, 5, 3, 0],
            'targets' : [0, 9, 20, 13, 15]
        },
        # index test data
	    
        # compi test data
        '[Unit] Compi Test':
        {#test_data_compi
            'function_dependency': 'None',
            'x' : [0, 1, 2, 3],
            'targets' : [1, 0, 3, 2]
        },
        # compi test data
	    
        # subtraction kernel data
        '[Unit] Subtraction Kernel Test':
        {#test_data_subtraction_kernel
            'function_dependency': 'None',
            'b1' : [[True, False, False, False], [True, True], [False, False]],
            'b2' : [[True, True, True, True], [False, True], [True, False]],
            'targets' : [(False, False, False, False), (True, False), (False, False)]
        },
        # subtraction kernel data

        # implies data
        '[Unit] Implies Test':
        {
            'function_dependency': '[ind, compi]',
            'row': [0, 0, 0, 0, 5, 5, 3, 4, 4, 5, 5],
            'col': [2, 3, 4, 5, 3, 4, 4, 8, 9, 9, 8],
            'weights': [0.2, 0, 0.8, 0.2, 0.2, 0.4, 0, 0.6, 0, 0.6, 0.2, 0.4, 0.4, 0.2, 0.6, 0, 0.4, 0, 0.4, 0, 0.4, 
            0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0.2, 0.4, 0.4, 0.2, 0.4, 0, 0.6, 0, 0.6,
            0, 0.4, 0, 0.4, 0.0, 0.6, 0.2, 0.2, 0, 0.4],
            'total': 1,
            'threshold': 0.125,
            'targets' : [True, False, True, False, True, False, False, False, False, False, False]
        },
        # implies data

        # equivalent data
        '[Unit] Equivalent Test':
        {
            'function_dependency': '[ind, compi]',
            'row': [4, 5, 4, 5],
            'col': [8, 9, 9, 8],
            'weights': [0.2, 0, 0.8, 0.2, 0.2, 0.4, 0, 0.6, 0, 0.6, 0.2, 0.4, 0.4, 0.2, 0.6, 0, 0.4, 0, 0.4, 0, 0.4, 
            0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0.2, 0.4, 0.4, 0.2, 0.4, 0, 0.6, 0, 0.6,
            0, 0.4, 0, 0.4, 0, 0.6, 0.2, 0.2, 0, 0.4],
            'total': 1,
            'threshold': 0.125,
            'targets' : [True, True, False, False]
        },
        # equivalent data

        # multiply_kernel data
        '[Unit] Multiply Kernel Test':
        {
            'function_dependency': '[ind]',
            'x': [[False, False, False, False, False, False, False, False, False, False],
			[False, False, True, False, False, False, False, False, False, False],
			[False, False, False, False, False, True, False, False, False, False],
			[False, False, False, False, False, False, False, False, True, True],
			[False, True, False, True, False, False, False, False, False, False],
			[False, False, False, True, False, False, False, True, False, True]],

            'dir': [True, False, True, False, False, True, False, False, True, True, False, True, False, False, True, True, False, False, False, False, True,
			False, False, False, False, False, True, True, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, True,
			False, False, False, False, False, True, False, False, True, True],

            'targets' : [(False, False, False, False, False, False, False, False, False, False),
			(False, False, True, True, False, False, False, False, False, False),
			(True, False, False, False, False, True, True, False, True, True),
			(True, False, False, False, False, True, True, False, True, True),
			(False, True, True, True, True, False, False, True, False, False),
			(True, True, True, True, True, True, True, True, True, True)]
        },
        # multiply_kernel data

		# check mask
        '[Unit] Check Mask Test':
        {
            'function_dependency': 'None',
            'mask': [[True, True, False, True],[False, True, True, False],[True, True, True, True]],
            'targets' : [(True, False, False, True), (False, True, True, False), (True, False, True, False)]
        },
        # check mask

        # mask kernel
        '[Unit] Mask Kernel Test':
        {
            'function_dependency': '[ind]',
			'mask_amper': [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
			False,True,False,False,False,False,False,False,False,False,False,False,True,False,False,True,False,False,False,False,False,False,
			False,False,False,False,True,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,False,False],
            'mask': [False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True],
			'current': [[True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False],
			[False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True],
			[True,False,True,False,False,True,False,True,False,True,False,True,True,False,True,False],
			[False,True,False,True,True,False,False,True,True,False,True,False,True,False,False,True]],
            'targets' : [(False,False,False,False,False,False,False,False,False,True,False,True,False,True,False,True),
			(False,False,False,False,False,False,False,False,True,True,False,True,False,True,False,True),
			(False,False,False,False,False,False,False,False,False,True,True,True,False,True,True,True),
			(False,False,False,False,False,False,False,False,True,True,False,True,True,True,False,True)]
        },
        # mask kernel

        # check update weights discounted
        '[Unit] Discounted Update Weights Test':
        {
            'function_dependency': '[ind]',
            'use_data_gen': 'Unit_Test_Data/data_gen/gen_update_weights.py'
        },
        # check update weights discounted

        # check orient all
        '[Integrate] Orient All Kernel Test':
        {
            'function_dependency': '[ind, compi]',
            'use_data_gen': 'Unit_Test_Data/data_gen/gen_orient_all.py'
        },
        # check orient all

        # up_GPU data
        '[Unit] Up GPU Test':
        {
            'function_dependency': '[multiply_kernel]',
            'signal': [[False, False, False, False, False, False, False, False, False, False],
			[False, False, True, False, False, False, False, False, False, False],
			[False, False, False, False, False, True, False, False, False, False],
			[False, False, False, False, False, False, False, False, True, True],
			[False, True, False, True, False, False, False, False, False, False],
			[False, False, False, True, False, False, False, True, False, True]],

            'dir': [True, False, True, False, False, True, False, False, True, True, False, True, False, False, True, True, False, False, False, False, True,
			False, False, False, False, False, True, True, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, True,
			False, False, False, False, False, True, False, False, True, True],

            'targets' : [(False, False, False, False, False, False, False, False, False, False),
			(False, False, True, True, False, False, False, False, False, False),
			(True, False, False, False, False, True, True, False, True, True),
			(True, False, False, False, False, True, True, False, True, True),
			(False, True, True, True, True, False, False, True, False, False),
			(True, True, True, True, True, True, True, True, True, True)]
        },
        # up_GPU data

        # gen mask data
        '[Unit] Generate Mask Test':
        {
            'function_dependency': '[check_mask, mask_kernel]',
			'mask_amper': [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
			False,True,False,False,False,False,False,False,False,False,False,False,True,False,False,True,False,False,False,False,False,False,
			False,False,False,False,True,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,False,False],
            
            'base_sensor_size': 4,

			'current': [[True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False],
			[False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True],
			[True,False,True,False,False,True,False,True,False,True,False,True,True,False,True,False],
			[False,True,False,True,True,False,False,True,True,False,True,False,True,False,False,True]],

            'targets' : [(False,False,False,False,False,False,False,False,False,True,False,True,False,True,False,True),
			(False,False,False,False,False,False,False,False,True,False,False,True,False,True,False,True),
			(False,False,False,False,False,False,False,False,False,True,True,False,False,True,True,False),
			(False,False,False,False,False,False,False,False,True,False,False,True,True,False,False,True)]
        },
        # gen mask data

        # set signal data
        '[Unit] Set Signal Test':
        {
            'function_dependency': 'None',
            'signal': [True, False, True, False, False, False, True, True],
            'targets' : [(True, False, True, False, False, False, True, True)]
        },
        # set signal data

        # init weight data
        '[Unit] Init Weight Test':
        {
            'function_dependency': 'None',
            'sensor_size': [2, 5, 8],
            'targets' : [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
        },
        # init weight data

        # init direction data
        '[Unit] Init Direction Test':
        {
            'function_dependency': 'None',
            'sensor_size': [2, 5, 8],
            'targets' : [(True, False, True, False, False, True, False, False, False, True),
			(True, False, True, False, False, True, False, False, False, True, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False,
			False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True),
			(True, False, True, False, False, True, False, False, False, True, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False,
			False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False,
			False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False,
			True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True,
			False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True)]
        },
        # init direction data

        # init mask amper
        '[Unit] init mask amper':
        {
            'function_dependency': 'None',
            'sensor_size': [2, 5],
            'targets': [(False, False, False, False, False, False),
			(False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
			False, False, False, False, False, False, False, False, False, False, False, False, False, False, False)]
        },
        # init mask amper

        # delay data
        '[Unit] Delay Test':
        {
            'function_dependency': '[copy_weights, amperand]',
            'id': [0],
            'weights': [[0.2], [0, 0.8], [0.2, 0.2, 0.4], [0, 0.6, 0, 0.6], [0.2, 0.4, 0.4, 0.2, 0.6], [0, 0.4, 0, 0.4, 0, 0.4], 
            [0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8], [0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2]],
            'last_total': 1,
            'measurable': [0.2, 0.8, 0.4, 0.6, 0.6, 0.4, 0.8, 0.2],
            'measurable_old': [0.2, 0.8, 0.4, 0.6, 0.6, 0.4, 0.8, 0.2],
            'targets' : [(0.04, 0.16, 0.08, 0.12, 0.12, 0.08, 0.16, 0.04, 0.2, 0.16, 0.64, 0.32, 0.48, 0.48, 0.32, 0.64,
            0.16, 0.0, 0.8)],
            'compare_fun': lambda target, result: all([abs(t - r) < 1e-5 for (t, r) in zip(target, result)]) 
        },
        # delay data

    }
	    