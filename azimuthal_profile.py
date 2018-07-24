px, py = np.mgrid()

# either px, py are relative to center or write line that finds their relative coordinates to center

radius = (px**2 + py**2)**0.5
angle = np.abs(np.pi - np.arctan(np.abs(py / px)))