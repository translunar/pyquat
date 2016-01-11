from pyquat import Quat

z = Quat(4,3,2,1) * Quat(1,2,3,4)
z.normalize()

z.to_matrix()

