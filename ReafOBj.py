from vispy import scene, io

canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

x= io.read_mesh("test/glasses/Glasses.obj")
print(x)

mesh = scene.visuals.Mesh(vertices=x[0], faces=faces[1], shading='smooth')

view.add(mesh)

view.camera = scene.TurntableCamera()
view.camera.depth_value = 10


if __name__ == '__main__':
    canvas.app.run()