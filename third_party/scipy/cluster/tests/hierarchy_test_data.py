from numpy import array


Q_X = array([[5.26563660e-01, 3.14160190e-01, 8.00656370e-02],
             [7.50205180e-01, 4.60299830e-01, 8.98696460e-01],
             [6.65461230e-01, 6.94011420e-01, 9.10465700e-01],
             [9.64047590e-01, 1.43082200e-03, 7.39874220e-01],
             [1.08159060e-01, 5.53028790e-01, 6.63804780e-02],
             [9.31359130e-01, 8.25424910e-01, 9.52315440e-01],
             [6.78086960e-01, 3.41903970e-01, 5.61481950e-01],
             [9.82730940e-01, 7.04605210e-01, 8.70978630e-02],
             [6.14691610e-01, 4.69989230e-02, 6.02406450e-01],
             [5.80161260e-01, 9.17354970e-01, 5.88163850e-01],
             [1.38246310e+00, 1.96358160e+00, 1.94437880e+00],
             [2.10675860e+00, 1.67148730e+00, 1.34854480e+00],
             [1.39880070e+00, 1.66142050e+00, 1.32224550e+00],
             [1.71410460e+00, 1.49176380e+00, 1.45432170e+00],
             [1.54102340e+00, 1.84374950e+00, 1.64658950e+00],
             [2.08512480e+00, 1.84524350e+00, 2.17340850e+00],
             [1.30748740e+00, 1.53801650e+00, 2.16007740e+00],
             [1.41447700e+00, 1.99329070e+00, 1.99107420e+00],
             [1.61943490e+00, 1.47703280e+00, 1.89788160e+00],
             [1.59880600e+00, 1.54988980e+00, 1.57563350e+00],
             [3.37247380e+00, 2.69635310e+00, 3.39981700e+00],
             [3.13705120e+00, 3.36528090e+00, 3.06089070e+00],
             [3.29413250e+00, 3.19619500e+00, 2.90700170e+00],
             [2.65510510e+00, 3.06785900e+00, 2.97198540e+00],
             [3.30941040e+00, 2.59283970e+00, 2.57714110e+00],
             [2.59557220e+00, 3.33477370e+00, 3.08793190e+00],
             [2.58206180e+00, 3.41615670e+00, 3.26441990e+00],
             [2.71127000e+00, 2.77032450e+00, 2.63466500e+00],
             [2.79617850e+00, 3.25473720e+00, 3.41801560e+00],
             [2.64741750e+00, 2.54538040e+00, 3.25354110e+00]])

ytdist = array([662., 877., 255., 412., 996., 295., 468., 268., 400., 754.,
                564., 138., 219., 869., 669.])

linkage_ytdist_single = array([[2., 5., 138., 2.],
                               [3., 4., 219., 2.],
                               [0., 7., 255., 3.],
                               [1., 8., 268., 4.],
                               [6., 9., 295., 6.]])

linkage_ytdist_complete = array([[2., 5., 138., 2.],
                                 [3., 4., 219., 2.],
                                 [1., 6., 400., 3.],
                                 [0., 7., 412., 3.],
                                 [8., 9., 996., 6.]])

linkage_ytdist_average = array([[2., 5., 138., 2.],
                                [3., 4., 219., 2.],
                                [0., 7., 333.5, 3.],
                                [1., 6., 347.5, 3.],
                                [8., 9., 680.77777778, 6.]])

linkage_ytdist_weighted = array([[2., 5., 138., 2.],
                                 [3., 4., 219., 2.],
                                 [0., 7., 333.5, 3.],
                                 [1., 6., 347.5, 3.],
                                 [8., 9., 670.125, 6.]])

# the optimal leaf ordering of linkage_ytdist_single
linkage_ytdist_single_olo = array([[5., 2., 138., 2.],
                                   [4., 3., 219., 2.],
                                   [7., 0., 255., 3.],
                                   [1., 8., 268., 4.],
                                   [6., 9., 295., 6.]])

X = array([[1.43054825, -7.5693489],
           [6.95887839, 6.82293382],
           [2.87137846, -9.68248579],
           [7.87974764, -6.05485803],
           [8.24018364, -6.09495602],
           [7.39020262, 8.54004355]])
 
linkage_X_centroid = array([[3., 4., 0.36265956, 2.],
                            [1., 5., 1.77045373, 2.],
                            [0., 2., 2.55760419, 2.],
                            [6., 8., 6.43614494, 4.],
                            [7., 9., 15.17363237, 6.]])

linkage_X_median = array([[3., 4., 0.36265956, 2.],
                          [1., 5., 1.77045373, 2.],
                          [0., 2., 2.55760419, 2.],
                          [6., 8., 6.43614494, 4.],
                          [7., 9., 15.17363237, 6.]])

linkage_X_ward = array([[3., 4., 0.36265956, 2.],
                        [1., 5., 1.77045373, 2.],
                        [0., 2., 2.55760419, 2.],
                        [6., 8., 9.10208346, 4.],
                        [7., 9., 24.7784379, 6.]])

# the optimal leaf ordering of linkage_X_ward
linkage_X_ward_olo = array([[4., 3., 0.36265956, 2.],
                            [5., 1., 1.77045373, 2.],
                            [2., 0., 2.55760419, 2.],
                            [6., 8., 9.10208346, 4.],
                            [7., 9., 24.7784379, 6.]])

inconsistent_ytdist = {
    1: array([[138., 0., 1., 0.],
              [219., 0., 1., 0.],
              [255., 0., 1., 0.],
              [268., 0., 1., 0.],
              [295., 0., 1., 0.]]),
    2: array([[138., 0., 1., 0.],
              [219., 0., 1., 0.],
              [237., 25.45584412, 2., 0.70710678],
              [261.5, 9.19238816, 2., 0.70710678],
              [233.66666667, 83.9424406, 3., 0.7306594]]),
    3: array([[138., 0., 1., 0.],
              [219., 0., 1., 0.],
              [237., 25.45584412, 2., 0.70710678],
              [247.33333333, 25.38372182, 3., 0.81417007],
              [239., 69.36377537, 4., 0.80733783]]),
    4: array([[138., 0., 1., 0.],
              [219., 0., 1., 0.],
              [237., 25.45584412, 2., 0.70710678],
              [247.33333333, 25.38372182, 3., 0.81417007],
              [235., 60.73302232, 5., 0.98793042]])}

fcluster_inconsistent = {
    0.8: array([6, 2, 2, 4, 6, 2, 3, 7, 3, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1]),
    1.0: array([6, 2, 2, 4, 6, 2, 3, 7, 3, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1]),
    2.0: array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1])}

fcluster_distance = {
    0.6: array([4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 3,
                1, 1, 1, 2, 1, 1, 1, 1, 1]),
    1.0: array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1]),
    2.0: array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1])}

fcluster_maxclust = {
    8.0: array([5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 7, 7, 7, 7, 7, 8, 7, 7, 7, 7, 4,
                1, 1, 1, 3, 1, 1, 1, 1, 2]),
    4.0: array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1]),
    1.0: array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1])}
