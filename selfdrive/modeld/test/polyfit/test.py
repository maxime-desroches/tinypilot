import numpy as np

pts = np.array([3.1261718, 3.1642578, 3.0548828, 3.1125, 3.190625, 3.01875, 2.9816406, 3.1222656, 2.9728515, 2.9826171, 3.034375, 3.0392578, 3.1642578, 3.0792968, 3.0011718, 3.0705078, 2.9904296, 3.0089843, 3.0597656, 3.0978515, 2.9210937, 2.9992187, 2.9474609, 2.9621093, 2.9289062, 2.89375, 2.7975585, 2.9015625, 2.8175781, 2.9132812, 2.8175781, 2.7501953, 2.8332031, 2.8166015, 2.7638671, 2.8878906, 2.7599609, 2.6999023, 2.6720703, 2.6398437, 2.7243164, 2.6120117, 2.6588867, 2.5558593, 2.5978515, 2.5485351, 2.4269531, 2.5001953, 2.4855468, 2.4367187, 2.2973144, 2.2812011, 2.2890136, 2.39375, 2.2836425, 2.3815429, 2.2138183, 2.1964843, 2.1840332, 2.1759765, 2.0421875, 2.1034667, 2.0281494, 2.0880859, 1.9706542, 1.9276855, 1.8522155, 1.8991821, 1.7780273, 1.8180053, 1.8326843, 1.8270385, 1.7182128, 1.6439941, 1.5360839, 1.68385, 1.4584472, 1.5955322, 1.6002929, 1.4157226, 1.4704101, 1.2936523, 1.2990234, 1.4281738, 1.4357421, 1.409375, 1.2511718, 1.2194335, 1.1554687, 1.043164, 1.0954101, 1.0392578, 1.0895507, 1.0880859, 0.897168, 0.83369142, 0.86494142, 0.87763673, 0.85322267, 0.72968751, 0.57832032, 0.73066407, 0.78828126, 0.69160157, 0.64375, 0.5919922, 0.5529297, 0.52070314, 0.60957032, 0.51093751, 0.3576172, 0.49921876, 0.284375, 0.21992187, 0.25214845, 0.30683595, 0.30976564, 0.2716797, 0.22089843, 0.25507814, 0.084179685, 0.071484372, 0.1828125, 0.15644531, 0.13789062, 0.054882813, 0.021679688, -0.091601565, -0.0203125, -0.13359375, -0.037890624, -0.29765624, -0.15605469, -0.30351561, -0.055468749, -0.22148438, -0.246875, -0.31718749, -0.25468749, -0.35234374, -0.16484375, -0.56523436, -0.56523436, -0.39921874, -0.58671874, -0.45585936, -0.50859374, -0.44023436, -0.42656249, -0.56328124, -0.70195311, -0.403125, -0.76445311, -0.98710936, -0.7625, -0.75273436, -0.825, -0.996875, -0.86210936, -0.99492186, -0.85625, -0.88359374, -0.97148436, -1.0320313, -1.1609375, -1.1296875, -1.0203125, -1.0691407, -1.2371094, -1.1277344, -1.2214844, -1.1921875, -1.2996094, -1.2917969, -1.3699219, -1.434375, -1.3699219, -1.3601563, -1.5730469, -1.3152344, -1.4851563, -1.48125, -1.5925782, -1.746875, -1.5847657, -1.6003907, -1.5984375, -1.7703125, -1.8328125, -1.8152344, -1.9714844, -1.9421875])

stds = np.array([1.0945262, 1.156862, 1.0777057, 1.1501777, 1.234844, 1.0140595, 1.2004665, 1.1926303, 1.1269455, 1.0362904, 0.98873031, 0.88530254, 1.0078473, 0.93637651, 0.90959895, 0.86409503, 0.86353016, 0.74534553, 0.78025728, 0.88014913, 0.75756663, 0.77129823, 0.75581717, 0.79222, 0.84098673, 0.79402477, 0.85648865, 0.80315614, 0.77346581, 0.73097658, 0.72557795, 0.72930044, 0.666103, 0.77142948, 0.704379, 0.6806078, 0.67680347, 0.71318036, 0.72244918, 0.66123307, 0.62547487, 0.67786956, 0.68404138, 0.70508122, 0.62400025, 0.72325015, 0.73942852, 0.67811751, 0.70370805, 0.65040058, 0.6870054, 0.66093785, 0.666103, 0.70040214, 0.65300173, 0.69714534, 0.65825552, 0.64833081, 0.6464982, 0.75850725, 0.69627059, 0.71659386, 0.69307244, 0.61554217, 0.62015557, 0.61998636, 0.67650336, 0.68142927, 0.6278621, 0.612294, 0.62592906, 0.63736153, 0.74233508, 0.69297016, 0.69621509, 0.67229682, 0.64879686, 0.72361159, 0.70229048, 0.60928106, 0.62712252, 0.66923952, 0.65802008, 0.68361813, 0.61587888, 0.63348651, 0.60727841, 0.64873856, 0.68847752, 0.58432156, 0.61683363, 0.63311422, 0.64981711, 0.57369792, 0.62604266, 0.62162364, 0.62066346, 0.62808979, 0.58524042, 0.63537884, 0.65367514, 0.63900274, 0.61089778, 0.62513435, 0.6470505, 0.63952166, 0.5937764, 0.64310449, 0.64330715, 0.64322031, 0.64632386, 0.60827911, 0.58887208, 0.61959165, 0.70725286, 0.64287293, 0.62326396, 0.65896219, 0.55610275, 0.6658656, 0.65681434, 0.583188, 0.6311124, 0.559652, 0.71419227, 0.62490743, 0.66699386, 0.62032485, 0.663036, 0.61414057, 0.66179425, 0.59399503, 0.65203643, 0.67839557, 0.63698763, 0.617452, 0.61022842, 0.7398752, 0.65657932, 0.68718743, 0.67901206, 0.66126263, 0.69949967, 0.70709819, 0.713336, 0.68130863, 0.68652785, 0.67028236, 0.7626031, 0.65259206, 0.72977453, 0.66049516, 0.64261246, 0.66906089, 0.69762796, 0.73719794, 0.69081914, 0.69849437, 0.72435051, 0.62354708, 0.68812829, 0.7193296, 0.66211933, 0.69278532, 0.7518425, 0.69661695, 0.672491, 0.71539241, 0.7369433, 0.66120356, 0.79088491, 0.77491313, 0.79442614, 0.7878198, 0.78881842, 0.70690477, 0.80707121, 0.78768665, 0.7215547, 0.75226194, 0.72196257, 0.765799, 0.77267712, 0.75844234, 0.81038833, 0.81188059, 0.79864907, 0.816436, 0.845298, 0.85074174, 0.73668873, 0.83516812])

order = 3

x = np.arange(0, len(pts))
print(np.polyfit(x, pts, order, w=1/stds))

# Do polyfit manually
w = 1.0 / stds
lhs = np.vander(x, order+1).astype(np.float64)
rhs = pts

lhs *= np.atleast_2d(w).T
rhs *= w

scale = np.sqrt((lhs*lhs).sum(axis=0))
lhs = lhs / scale
c, resids, rank, s = np.linalg.lstsq(lhs, rhs, rcond=None)
c = (c.T/scale).T
print(c)
